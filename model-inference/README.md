# Local Model Inference Snapshot

This directory is a local source snapshot committed with `flagOS-installers`.
The inference entrypoint is intentionally self-contained and does not require
the unavailable external model-inference repository.

Model weights are downloaded by the installer into its configured installation
prefix. Only small model files may be committed under `model-inference/models/`;
large model weights must remain outside Git.

Run the entrypoint with a local Hugging Face model directory:

```bash
python3 examples/run_llm_with_flaggems.py \
  --model-path /path/to/model \
  --prompt "Explain FlagGems in one sentence." \
  --max-new-tokens 32
```
