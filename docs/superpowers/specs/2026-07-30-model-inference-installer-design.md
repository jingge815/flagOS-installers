# Model Inference Installer Design

## Goal

Add a standalone `3-install-model-inference.sh` that runs after the FlagTree
and FlagGems installers, optionally reuses the source-built PyTorch
installation, otherwise uses the existing FlagTree PyTorch wheel, downloads a
HuggingFace model when needed, and completes a real FlagGems-backed LLM
inference run without root privileges.

## Constraints

- Target platform is Ubuntu 22.04 x86_64 with a working NVIDIA driver and GPU.
- The script must not require root access or system-wide package installation.
- The script must work after only `0-install-flagtree.sh` and
  `1-install-flaggems.sh`.
- If `2-install-pytorch.sh` has produced a usable installation, its Python and
  source-built PyTorch must be preferred automatically.
- If the source-built PyTorch installation is absent or unusable, the script
  must use the FlagTree Python and install/use the existing CUDA PyTorch wheel
  as needed.
- The public `flagos-ai/model-inference` repository is unavailable, so the
  required inference source is a local snapshot committed under this
  repository's `model-inference/` directory.
- Small model assets may be committed under `model-inference/models/`.
  Large HuggingFace weights must be downloaded on demand into the selected
  user-owned installation prefix and must not be committed.

## File Layout

The implementation will contain:

- `3-install-model-inference.sh`
  - The only installation and execution entry point.
  - Checks platform and prerequisite installation prefixes.
  - Selects the PyTorch runtime.
  - Installs missing Python inference dependencies into that runtime.
  - Downloads or reuses the HuggingFace model.
  - Runs the real inference entry point and records logs/Triton artifacts.
  - Writes `env-model-inference.sh` for later use.
- `model-inference/examples/run_llm_with_flaggems.py`
  - Minimal local inference entry point adapted from the existing reference
    implementation.
  - Loads a local causal language model with Transformers.
  - Runs generation inside `flag_gems.use_gems()`.
  - Prints PyTorch, Triton, FlagTree, FlagGems, GPU, and output information.
- `model-inference/README.md`
  - Documents the local source snapshot and model asset policy.
- `tests/test-model-inference.sh`
  - Always performs static checks.
  - Performs the complete GPU-backed installation/model/inference flow when
    `RUN_MODEL_INFERENCE_INTEGRATION=1` is set.
- `README.md`
  - Documents the fourth script's prerequisites, options, runtime-selection
    behavior, model handling, and verification commands.

The reference shell scripts that depend on
`/media/disk/fengjingge/software/flagos-nvidia` will not be copied into the
installer snapshot. Their responsibilities are implemented directly by the
new standalone installer.

## Installation Prefixes

Defaults are relative to the installer directory:

```text
FlagTree:          ../flagOS-installed/flagTree
FlagGems:          ../flagOS-installed/flagGems
PyTorch:           ../flagOS-installed/pytorch
Model inference:   ../flagOS-installed/model-inference
Source snapshot:   ./model-inference
```

All prefixes are configurable through command-line options. The source
snapshot is verified to contain the inference entry point and is never fetched
from the unavailable remote repository.

## Runtime Selection

The installer uses automatic mode by default:

1. Validate `flagGems/env-flaggems.sh` and the FlagTree environment it sources.
2. If `pytorch/env-pytorch.sh` and its Python can import CUDA-enabled PyTorch,
   select that Python.
3. In source-built PyTorch mode, install the latest FlagTree wheel from the
   FlagTree prefix and install FlagGems from its pinned source checkout into
   the selected PyTorch Python without replacing PyTorch dependencies.
4. Otherwise select the FlagTree Python. If it cannot import PyTorch with CUDA,
   install the pinned CUDA PyTorch wheel used by the FlagTree installer.
5. Source the selected environment plus the FlagTree/FlagGems Triton paths and
   write a combined model-inference environment file.

The script will expose an explicit mode option for reproducibility:

- `auto`: prefer source-built PyTorch when usable.
- `compiled`: require the PyTorch prefix and fail if it is unusable.
- `wheel`: force the FlagTree Python and wheel path.

## Dependency Installation

The selected runtime will be checked for these packages:

```text
transformers>=4.43,<5
huggingface_hub>=0.23,<1
accelerate>=0.33,<2
safetensors>=0.4,<1
sentencepiece>=0.2,<1
```

Already-satisfied requirements are reused. Missing or incompatible packages
are installed with the selected runtime's `pip`, using a cache under the
model-inference prefix. No system package manager is used by this script.

## Model Download and Inference

The default model is `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Model ID,
revision, local model path, prompt, and maximum generated tokens are
configurable.

If a local model path is supplied, no download occurs. Otherwise the installer
uses `huggingface_hub.snapshot_download` and stores the model under the
model-inference prefix. Small checked-in model assets under
`model-inference/models/` are allowed, but downloaded large weights remain
outside the Git repository.

The default command completes all of these stages:

1. Validate the selected runtime and CUDA.
2. Install missing inference dependencies.
3. Download or reuse the model.
4. Run the local inference entry point with FlagGems enabled.
5. Save a timestamped log and Triton dump directory.

`--skip-inference` allows environment preparation without generation.
`--skip-test` skips the CUDA stack preflight but does not imply that the
inference run is safe.

## Verification

Static verification always runs:

- `bash -n 3-install-model-inference.sh`
- `bash 3-install-model-inference.sh --help`
- Python bytecode compilation for the local inference entry point
- checks for required source files and documented options

The integration test is opt-in because it requires a GPU and may download a
model:

```bash
RUN_MODEL_INFERENCE_INTEGRATION=1 bash tests/test-model-inference.sh
```

The integration path executes the actual installer and inference flow, then
asserts that:

- the selected PyTorch reports CUDA availability;
- FlagGems and Triton/FlagTree import successfully;
- the inference output is non-empty;
- the run log identifies PyTorch, Triton, FlagTree, FlagGems, and the GPU;
- Triton produced at least one non-empty compilation artifact.
