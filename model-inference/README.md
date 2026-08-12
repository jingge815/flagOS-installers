# Local Model Inference Snapshot

This directory is a local source snapshot committed with `flagOS-installers`.
The inference entrypoint is intentionally self-contained and does not require
the unavailable external model-inference repository.

Use it through the top-level installer after FlagTree and FlagGems are ready:

```bash
bash 0-install-flagtree.sh
bash 1-install-flaggems.sh
# optional: build PyTorch from source
bash 2-install-pytorch.sh
bash 3-install-model-inference.sh
```

`3-install-model-inference.sh` defaults to `builtin-gpt2-random`, a Python
constructed GPT-2 with `n_layer=4`, `n_head=8`, `n_embd=512`, and
`max_seq=128`. The default path does not download model weights. The installer
still installs missing inference Python packages into the selected runtime and
runs one FlagGems-backed generation.

PyTorch selection is automatic by default. If `../flagOS-installed/pytorch`
contains a usable CUDA PyTorch from `2-install-pytorch.sh`, the installer uses
it. Otherwise it uses the FlagTree Python and installs the CUDA PyTorch wheel.
Use `--pytorch-mode compiled` or `--pytorch-mode wheel` to force either path.

Useful installer commands:

```bash
bash 3-install-model-inference.sh
bash 3-install-model-inference.sh --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0
bash 3-install-model-inference.sh --skip-inference
bash 3-install-model-inference.sh --pytorch-mode wheel
source ../flagOS-installed/model-inference/env-model-inference.sh
```

Successful inference is validated from the log. The installer requires
`inference_status: ok`, a positive `flaggems_generated_tokens` value, and
non-empty text after `flaggems_text:`.

Run with a previously downloaded local Hugging Face model directory:

```bash
bash 3-install-model-inference.sh \
  --model-path /path/to/model \
  --prompt "Explain FlagGems in one sentence." \
  --max-new-tokens 32
```

The installer writes logs to
`../flagOS-installed/model-inference/logs/inference-YYYYMMDD_HHMMSS.log` and
Triton artifacts to
`../flagOS-installed/model-inference/artifacts/triton-dumps/<timestamp>/`.

Model weights are downloaded only when `--model-id` is used. Downloaded weights
go into the configured installation prefix. Only small model files may be
committed under `model-inference/models/`; large model weights must remain
outside Git.

For direct debugging with the builtin model:

```bash
python3 examples/run_llm_with_flaggems.py \
  --builtin-model builtin-gpt2-random \
  --max-seq 128 \
  --prompt "Explain FlagGems in one sentence." \
  --max-new-tokens 32
```

For direct debugging with a local Hugging Face model directory:

```bash
python3 examples/run_llm_with_flaggems.py \
  --model-path /path/to/model \
  --prompt "Explain FlagGems in one sentence." \
  --max-new-tokens 32
```
