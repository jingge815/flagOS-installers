#!/usr/bin/env python3
"""Run causal LM inference under FlagGems."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import os
import time
from pathlib import Path

import torch

# 无 GPU 时必须在 import flag_gems 之前注入编译期 driver，否则 import 期就抛
# "0 active drivers" / "No device were detected"。安装脚本会通过
# CPU_HOST_DRIVER_PY 指到 flagOS-installers/cpu-host-driver.py；直接手工运行本
# 脚本时该变量可能没设，此时按同目录的相对位置找一次。
if not torch.cuda.is_available():
    _shim = os.environ.get("CPU_HOST_DRIVER_PY") or str(
        Path(__file__).resolve().parents[2] / "cpu-host-driver.py"
    )
    if Path(_shim).is_file():
        exec(open(_shim).read())
    else:
        raise SystemExit(
            f"没有 GPU，且找不到 cpu-host-driver.py（试过 {_shim}）。"
            "请设 CPU_HOST_DRIVER_PY 指向 flagOS-installers/cpu-host-driver.py。"
        )

import flag_gems
import triton
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)

# 有 GPU 就用 GPU，没有就用 CPU。模型推理在 CPU 上是完整可用的。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _sync() -> None:
    """只在有 GPU 时同步；CPU 上执行本来就是同步的。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


BUILTIN_GPT2_MODEL = "builtin-gpt2-random"
BUILTIN_GPT2_VOCAB_SIZE = 50257
BUILTIN_GPT2_EOS_TOKEN_ID = 50256


def distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-path",
        help="Local HuggingFace model directory",
    )
    model_group.add_argument(
        "--builtin-model",
        choices=[BUILTIN_GPT2_MODEL],
        help="Use the legacy randomly initialized GPT-2 smoke model; not the default target",
    )
    parser.add_argument(
        "--prompt",
        default="Explain in one sentence what FlagGems does for PyTorch.",
        help="Prompt text",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seq", type=int, default=128, help="Builtin model context length")
    parser.add_argument("--compare-baseline", action="store_true")
    return parser.parse_args()


class SimpleBatchEncoding(dict):
    """Tiny BatchEncoding subset used by SimpleTokenizer."""

    def to(self, device: str) -> "SimpleBatchEncoding":
        return SimpleBatchEncoding({key: value.to(device) for key, value in self.items()})


class SimpleTokenizer:
    """Offline tokenizer for builtin random GPT-2 smoke tests.

    This is intentionally simple: it maps UTF-8 bytes to stable token IDs and
    decodes unknown generated IDs as tok<N>. It validates the inference stack,
    not language quality.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self) -> None:
        self.bos_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|endoftext|>"
        self.bos_token_id = BUILTIN_GPT2_EOS_TOKEN_ID
        self.eos_token_id = BUILTIN_GPT2_EOS_TOKEN_ID
        self.pad_token_id = BUILTIN_GPT2_EOS_TOKEN_ID

    @property
    def vocab_size(self) -> int:
        return BUILTIN_GPT2_VOCAB_SIZE

    def __call__(self, text: str, return_tensors: str | None = None, **_: object) -> SimpleBatchEncoding:
        if return_tensors != "pt":
            raise ValueError("SimpleTokenizer only supports return_tensors='pt'")
        token_ids = self.encode(text)
        if not token_ids:
            token_ids = [self.eos_token_id]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return SimpleBatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})

    def encode(self, text: str, **_: object) -> list[int]:
        return [byte + 1 for byte in text.encode("utf-8")]

    def decode(self, token_ids, skip_special_tokens: bool = False, **_: object) -> str:
        decoded: list[str] = []
        for token_id in token_ids:
            value = int(token_id)
            if skip_special_tokens and value == self.eos_token_id:
                continue
            if 1 <= value <= 256:
                decoded.append(bytes([value - 1]).decode("utf-8", errors="replace"))
            else:
                decoded.append(f"tok{value}")
        return "".join(decoded)


def create_builtin_gpt2_components(max_seq: int) -> tuple[SimpleTokenizer, GPT2LMHeadModel]:
    if max_seq < 1:
        raise ValueError("--max-seq must be a positive integer")
    config = GPT2Config(
        vocab_size=BUILTIN_GPT2_VOCAB_SIZE,
        n_positions=max_seq,
        n_ctx=max_seq,
        n_embd=512,
        n_layer=4,
        n_head=8,
        bos_token_id=BUILTIN_GPT2_EOS_TOKEN_ID,
        eos_token_id=BUILTIN_GPT2_EOS_TOKEN_ID,
        pad_token_id=BUILTIN_GPT2_EOS_TOKEN_ID,
    )
    return SimpleTokenizer(), GPT2LMHeadModel(config)


def validate_generation(
    text: str,
    elapsed: float,
    prompt_length: int,
    generated_tokens: int,
    label: str,
) -> None:
    if elapsed <= 0:
        raise SystemExit(f"{label} elapsed time is not positive: {elapsed}")
    if generated_tokens < 1:
        raise SystemExit(f"{label} generated no new tokens")
    if prompt_length < 1:
        raise SystemExit(f"{label} prompt token length is empty")
    if not text.strip():
        raise SystemExit(f"{label} generated empty text")


def validate_generation_request(
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    max_seq: int | None = None,
) -> None:
    if max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be a positive integer")
    if max_seq is not None and max_seq < 1:
        raise SystemExit("--max-seq must be a positive integer")
    prompt_tokens = len(tokenizer.encode(prompt))
    if prompt_tokens < 1:
        raise SystemExit("prompt token length is empty")
    if max_seq is not None and prompt_tokens + max_new_tokens > max_seq:
        raise SystemExit(
            "prompt tokens plus --max-new-tokens exceeds --max-seq: "
            f"{prompt_tokens} + {max_new_tokens} > {max_seq}"
        )


def generate_once(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    use_gems: bool,
) -> tuple[str, float, int, int]:
    inputs = {key: value.to(DEVICE) for key, value in tokenizer(prompt, return_tensors="pt").items()}
    _sync()
    start = time.perf_counter()
    # 无 GPU 时不能启用 FlagGems：它的算子是 Triton kernel，纯 CPU 上无处可跑
    # （autotune 还要实测计时）。此时退回 PyTorch 原生实现——推理结果照样正确，
    # 只是不经过 FlagGems 的算子。
    enable_gems = use_gems and torch.cuda.is_available()
    if use_gems and not enable_gems:
        print("  note: 未检测到 GPU，本轮用 PyTorch 原生算子（FlagGems 的 Triton "
              "kernel 需要 GPU 才能执行）")
    context = flag_gems.use_gems() if enable_gems else contextlib.nullcontext()
    with torch.inference_mode():
        with context:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    _sync()
    prompt_length = inputs["input_ids"].shape[-1]
    new_token_ids = output_ids[0, prompt_length:]
    generated_tokens = new_token_ids.shape[-1]
    return (
        tokenizer.decode(new_token_ids, skip_special_tokens=True),
        time.perf_counter() - start,
        prompt_length,
        generated_tokens,
    )


def main() -> int:
    args = parse_args()
    if args.model_path:
        model_path = Path(args.model_path).resolve()
        if not model_path.is_dir():
            raise SystemExit(f"model path is not a directory: {model_path}")
    else:
        model_path = None
    # GPU 可选：无卡时在 CPU 上跑，算子编译与数值对拍都不需要 GPU 硬件。

    print("runtime:")
    if args.builtin_model:
        print(f"  model_backend: {args.builtin_model}")
        print("  model_source: builtin random initialization, no download")
        print("  model_config: GPT2 n_layer=4 n_head=8 n_embd=512 max_seq=%d" % args.max_seq)
    else:
        print(f"  model_path: {model_path}")
    print(f"  torch: {torch.__version__}")
    print(f"  torch_cuda: {torch.version.cuda}")
    print(f"  triton_import: {triton.__version__} ({triton.__file__})")
    print(f"  flagtree: {distribution_version('flagtree')}")
    print(f"  flag_gems: {distribution_version('flag_gems')}")
    print(f"  transformers: {distribution_version('transformers')}")
    print(f"  gpu: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available()
          else "  gpu: none（纯 CPU 模式）")

    if args.builtin_model:
        tokenizer, model = create_builtin_gpt2_components(args.max_seq)
        validate_generation_request(tokenizer, args.prompt, args.max_new_tokens, args.max_seq)
        model.to(dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        validate_generation_request(tokenizer, args.prompt, args.max_new_tokens)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
    model.eval()
    model.to(DEVICE)

    if args.compare_baseline:
        baseline_text, baseline_elapsed, baseline_prompt_tokens, baseline_generated_tokens = generate_once(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            use_gems=False,
        )
        validate_generation(
            baseline_text,
            baseline_elapsed,
            baseline_prompt_tokens,
            baseline_generated_tokens,
            "baseline",
        )
        print("baseline_elapsed_sec:", f"{baseline_elapsed:.4f}")
        print("baseline_prompt_tokens:", baseline_prompt_tokens)
        print("baseline_generated_tokens:", baseline_generated_tokens)
        print("baseline_text:")
        print(baseline_text)

    gems_text, gems_elapsed, gems_prompt_tokens, gems_generated_tokens = generate_once(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        use_gems=True,
    )
    validate_generation(
        gems_text,
        gems_elapsed,
        gems_prompt_tokens,
        gems_generated_tokens,
        "flaggems",
    )
    print("flaggems_elapsed_sec:", f"{gems_elapsed:.4f}")
    print("flaggems_prompt_tokens:", gems_prompt_tokens)
    print("flaggems_generated_tokens:", gems_generated_tokens)
    print("flaggems_text:")
    print(gems_text)
    print(
        "inference_status: ok "
        f"(backend={args.builtin_model or 'huggingface'}, generated_tokens={gems_generated_tokens})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
