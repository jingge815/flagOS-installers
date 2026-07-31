#!/usr/bin/env python3
"""Run HuggingFace causal LM inference under FlagGems."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import time
from pathlib import Path

import flag_gems
import torch
import triton
from transformers import AutoModelForCausalLM, AutoTokenizer


def distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local HuggingFace model directory")
    parser.add_argument(
        "--prompt",
        default="Explain in one sentence what FlagGems does for PyTorch.",
        help="Prompt text",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--compare-baseline", action="store_true")
    return parser.parse_args()


def generate_once(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    use_gems: bool,
) -> tuple[str, float]:
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
    return (
        tokenizer.decode(new_token_ids, skip_special_tokens=True),
        time.perf_counter() - start,
    )


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"model path is not a directory: {model_path}")
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is False")

    print("runtime:")
    print(f"  model_path: {model_path}")
    print(f"  torch: {torch.__version__}")
    print(f"  torch_cuda: {torch.version.cuda}")
    print(f"  triton_import: {triton.__version__} ({triton.__file__})")
    print(f"  flagtree: {distribution_version('flagtree')}")
    print(f"  flag_gems: {distribution_version('flag_gems')}")
    print(f"  transformers: {distribution_version('transformers')}")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    model.to("cuda")

    if args.compare_baseline:
        baseline_text, baseline_elapsed = generate_once(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            use_gems=False,
        )
        print("baseline_elapsed_sec:", f"{baseline_elapsed:.4f}")
        print("baseline_text:")
        print(baseline_text)

    gems_text, gems_elapsed = generate_once(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        use_gems=True,
    )
    print("flaggems_elapsed_sec:", f"{gems_elapsed:.4f}")
    print("flaggems_text:")
    print(gems_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
