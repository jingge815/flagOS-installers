#!/usr/bin/env python3
"""Run causal LM inference under FlagGems."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import time
from pathlib import Path

import flag_gems
import torch
import triton
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)


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
        help="Use a randomly initialized builtin model that does not download weights",
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
    inputs = {key: value.to("cuda") for key, value in tokenizer(prompt, return_tensors="pt").items()}
    torch.cuda.synchronize()
    start = time.perf_counter()
    context = flag_gems.use_gems() if use_gems else contextlib.nullcontext()
    with torch.inference_mode():
        with context:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    torch.cuda.synchronize()
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
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is False")

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
    print(f"  gpu: {torch.cuda.get_device_name(0)}")

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
    model.to("cuda")

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
