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

`3-install-model-inference.sh` defaults to
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, downloads or reuses the model under
`../flagOS-installed/model-inference/models/`, installs missing inference
Python packages into the selected runtime, and runs one FlagGems-backed
generation.

PyTorch selection is automatic by default. If `../flagOS-installed/pytorch`
contains a usable CUDA PyTorch from `2-install-pytorch.sh`, the installer uses
it. Otherwise it uses the FlagTree Python and installs the CUDA PyTorch wheel.
Use `--pytorch-mode compiled` or `--pytorch-mode wheel` to force either path.

Useful installer commands:

```bash
bash 3-install-model-inference.sh --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0
bash 3-install-model-inference.sh --skip-inference
bash 3-install-model-inference.sh --pytorch-mode wheel
source ../flagOS-installed/model-inference/env-model-inference.sh
```

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

Model weights are downloaded by the installer into its configured installation
prefix. Only small model files may be committed under `model-inference/models/`;
large model weights must remain outside Git.

For direct debugging, run the entrypoint with a local Hugging Face model
directory:

```bash
python3 examples/run_llm_with_flaggems.py \
  --model-path /path/to/model \
  --prompt "Explain FlagGems in one sentence." \
  --max-new-tokens 32
```
