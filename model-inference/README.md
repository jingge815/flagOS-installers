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
export HF_TOKEN=<your-token>
bash 3-install-model-inference.sh --skip-inference
bash 3-install-model-inference.sh --skip-download --skip-inference
bash 3-install-model-inference.sh
```

`3-install-model-inference.sh` defaults to the official Hugging Face model
`meta-llama/Llama-2-7b-hf`. Its local model directory is
`/media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf`.
The installer reuses a complete directory and otherwise downloads the model.
The official LLaMA2 repository requires HuggingFace authorization: set
`HF_TOKEN` or run `huggingface-cli login`. Failed download or access does not
fall back to GPT-2, TinyLlama, or random weights.

PyTorch selection is automatic by default. If `../flagOS-installed/pytorch`
contains a usable CUDA PyTorch from `2-install-pytorch.sh`, the installer uses
it. Otherwise it uses the FlagTree Python and installs the CUDA PyTorch wheel.
Use `--pytorch-mode compiled` or `--pytorch-mode wheel` to force either path.

Useful installer commands:

```bash
bash 3-install-model-inference.sh
bash 3-install-model-inference.sh --skip-inference
bash 3-install-model-inference.sh --skip-download --skip-inference
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

Downloaded weights go into the configured installation prefix. Only small model files may be
committed under `model-inference/models/`; large model weights must remain
outside Git.

Legacy/debug smoke: the builtin GPT-2 model is only for explicit debugging and
requires `--builtin-model builtin-gpt2-random`; it is never the default or a
fallback for the official model. For direct debugging with it:

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
