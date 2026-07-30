# Model Inference Installer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained non-root model-inference installer that uses the existing FlagTree/FlagGems stack, prefers a usable source-built PyTorch installation, downloads a HuggingFace model, and completes real FlagGems-backed inference.

**Architecture:** Keep the inference entry point as a small local source snapshot under `model-inference/`. Keep installation logic in one Bash script. The script selects either the compiled-PyTorch Python or the FlagTree Python, installs only missing HuggingFace packages into that Python, creates a combined environment file, and runs the local inference entry point while recording logs and Triton artifacts.

**Tech Stack:** Bash, Python 3.10, PyTorch CUDA, FlagTree/Triton, FlagGems, Transformers, HuggingFace Hub, Git, Ubuntu 22.04 x86_64.

## Global Constraints

- No root privileges or system-wide package installation.
- Default FlagTree prefix: `../flagOS-installed/flagTree`.
- Default FlagGems prefix: `../flagOS-installed/flagGems`.
- Default PyTorch prefix: `../flagOS-installed/pytorch`.
- Default model-inference prefix: `../flagOS-installed/model-inference`.
- Source snapshot lives at `./model-inference` and is committed with this repository.
- Prefer `../flagOS-installed/pytorch/env-pytorch.sh` when its Python imports CUDA-enabled PyTorch.
- Otherwise use the FlagTree Python and CUDA PyTorch wheel.
- Default model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
- Large model weights must stay outside Git under the model-inference installation prefix.
- The default installer command runs the complete inference flow.

---

### Task 1: Add Static and Integration Test Harness

**Files:**
- Create: `tests/test-model-inference.sh`

**Interfaces:**
- Consumes: repository root, `3-install-model-inference.sh`, and
  `model-inference/examples/run_llm_with_flaggems.py`.
- Produces: a static test command that always runs and a complete GPU-backed
  integration command enabled by `RUN_MODEL_INFERENCE_INTEGRATION=1`.

- [ ] **Step 1: Write the failing test**

Create the test harness with this behavior:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
INSTALLER="$ROOT_DIR/3-install-model-inference.sh"
ENTRYPOINT="$ROOT_DIR/model-inference/examples/run_llm_with_flaggems.py"

[[ -f "$INSTALLER" ]] || { echo "missing installer: $INSTALLER" >&2; exit 1; }
[[ -f "$ENTRYPOINT" ]] || { echo "missing inference entrypoint: $ENTRYPOINT" >&2; exit 1; }

bash -n "$INSTALLER"
HELP_OUTPUT=$(bash "$INSTALLER" --help)
grep -F -- '--pytorch-mode' <<<"$HELP_OUTPUT"
grep -F -- '--model-id' <<<"$HELP_OUTPUT"
grep -F -- '--skip-inference' <<<"$HELP_OUTPUT"

python3 -m py_compile "$ENTRYPOINT"
grep -F -- 'flag_gems.use_gems' "$ENTRYPOINT"
grep -F -- 'AutoModelForCausalLM' "$ENTRYPOINT"

if [[ "${RUN_MODEL_INFERENCE_INTEGRATION:-0}" == 1 ]]; then
  PREFIX="${MODEL_INFERENCE_TEST_PREFIX:-$ROOT_DIR/../flagOS-installed/model-inference}"
  MODEL_ID="${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
  bash "$INSTALLER" \
    --prefix "$PREFIX" \
    --model-id "$MODEL_ID" \
    --max-new-tokens "${MAX_NEW_TOKENS:-8}"

  LOG_FILE=$(find "$PREFIX/logs" -maxdepth 1 -type f -name 'inference-*.log' \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  [[ -n "$LOG_FILE" && -s "$LOG_FILE" ]] || {
    echo "missing non-empty inference log" >&2
    exit 1
  }
  grep -F -- 'flag_gems:' "$LOG_FILE"
  grep -F -- 'flagtree:' "$LOG_FILE"
  grep -F -- 'triton_import:' "$LOG_FILE"
  grep -F -- 'torch:' "$LOG_FILE"
  grep -F -- 'flaggems_text:' "$LOG_FILE"
  awk '/flaggems_text:/{seen=1; next} seen && NF {found=1; exit} END{exit !found}' "$LOG_FILE"
  find "$PREFIX/artifacts/triton-dumps" -type f -size +0c -print -quit | grep -q .
fi

printf 'model-inference static checks passed\n'
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
bash tests/test-model-inference.sh
```

Expected: FAIL because the placeholder installer has no required help options
and the local inference entry point does not yet exist.

- [ ] **Step 3: Commit the test harness**

```bash
git add tests/test-model-inference.sh
git commit -m "test: add model inference installer checks"
```

### Task 2: Add the Local Inference Source Snapshot

**Files:**
- Create: `model-inference/examples/run_llm_with_flaggems.py`
- Create: `model-inference/README.md`

**Interfaces:**
- Consumes: a local model directory, prompt text, and token limit.
- Produces: stdout containing runtime identity, generated text, and timing.
- Required entrypoint arguments:
  `--model-path DIR`, `--prompt TEXT`, `--max-new-tokens N`,
  `--compare-baseline`.

- [ ] **Step 1: Add the minimal inference entrypoint**

Implement the entrypoint with these required behaviors:

```python
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


def generate_once(model, tokenizer, prompt: str, max_new_tokens: int,
                  use_gems: bool) -> tuple[str, float]:
    inputs = {key: value.to("cuda")
              for key, value in tokenizer(prompt, return_tensors="pt").items()}
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
    return tokenizer.decode(output_ids[0], skip_special_tokens=True), time.perf_counter() - start
```

The `main()` function must validate the model path and CUDA availability,
print `torch:`, `torch_cuda:`, `triton_import:`, `flagtree:`, `flag_gems:`,
`transformers:`, and `gpu:`, load the tokenizer/model with
`trust_remote_code=False`, use `torch_dtype=torch.float16`, and print a
non-empty `flaggems_text:` result after calling `generate_once(..., True)`.

- [ ] **Step 2: Add snapshot documentation**

Document that this is a local snapshot committed with `flagOS-installers`,
that the unavailable external repository is intentionally not required, and
that only small model files may be committed under `model-inference/models/`.

- [ ] **Step 3: Run static source checks**

Run:

```bash
python3 -m py_compile model-inference/examples/run_llm_with_flaggems.py
bash tests/test-model-inference.sh
```

Expected: Python compilation succeeds and the test now fails only because the
installer help options have not been implemented.

- [ ] **Step 4: Commit the source snapshot**

```bash
git add model-inference
git commit -m "feat: add local model inference snapshot"
```

### Task 3: Implement Installer Arguments, Paths, and Platform Checks

**Files:**
- Modify: `3-install-model-inference.sh`

**Interfaces:**
- `bash 3-install-model-inference.sh [options]`
- Options:
  `--prefix`, `--source-dir`, `--flagtree-prefix`, `--flaggems-prefix`,
  `--pytorch-prefix`, `--pytorch-mode auto|compiled|wheel`,
  `--model-id`, `--revision`, `--model-path`, `--local-dir`, `--prompt`,
  `--max-new-tokens`, `--skip-test`, `--skip-download`,
  `--skip-inference`, `--compare-baseline`, `-h`, `--help`.
- Produces: canonicalized user-owned paths and a clear failure before any
  install action when prerequisites are missing.

- [ ] **Step 1: Implement the shell skeleton**

Use `set -euo pipefail`, resolve `SCRIPT_DIR`, define `die`, `note`,
`require_command`, safe path canonicalization, and a usage block that includes
every option above.

- [ ] **Step 2: Implement platform checks**

Require Ubuntu 22.04, x86_64, `git`, `tar`, `awk`, `sed`, `find`, `python3`,
and `curl` or `wget`. Require `nvidia-smi` unless `--skip-test` is set.
Reject `/` as any writable prefix and reject newline-containing paths.

- [ ] **Step 3: Implement prerequisite validation**

Require:

```text
<flagtree-prefix>/env-flagtree.sh
<flaggems-prefix>/env-flaggems.sh
<flagtree-prefix>/python/bin/python
<source-dir>/examples/run_llm_with_flaggems.py
```

Do not require the PyTorch prefix in `auto` or `wheel` mode.

- [ ] **Step 4: Run the static test**

Run:

```bash
bash tests/test-model-inference.sh
```

Expected: PASS for syntax, help options, and source checks.

- [ ] **Step 5: Commit the installer skeleton**

```bash
git add 3-install-model-inference.sh
git commit -m "feat: add model inference installer interface"
```

### Task 4: Implement Runtime Selection and Dependency Installation

**Files:**
- Modify: `3-install-model-inference.sh`

**Interfaces:**
- Produces `RUNTIME_PYTHON`, `RUNTIME_MODE`, and a generated
  `<prefix>/env-model-inference.sh`.
- `auto` selects compiled PyTorch only when
  `<pytorch-prefix>/env-pytorch.sh` exists and its Python imports CUDA-enabled
  PyTorch.
- `compiled` fails with an actionable error when that condition is false.
- `wheel` always selects `<flagtree-prefix>/python/bin/python`.

- [ ] **Step 1: Add runtime probes**

Implement a probe using the candidate Python:

```bash
"$candidate_python" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.version.cuda else 1)
PY
```

In compiled mode, source `env-flaggems.sh` and then `env-pytorch.sh`; install
the newest `flagtree-*.whl` from `<flagtree-prefix>/wheels` into the compiled
Python with `--force-reinstall --no-deps`, then install the FlagGems source
from the `FLAGGEMS_SOURCE` exported by `env-flaggems.sh` with
`--no-build-isolation --no-deps`.

In wheel mode, source `env-flaggems.sh`; if the FlagTree Python cannot import
CUDA PyTorch, install:

```text
torch==2.7.1+cu128
--index-url https://download.pytorch.org/whl/cu128
```

- [ ] **Step 2: Add dependency checks**

Check these requirements with `importlib.metadata` and
`packaging.requirements.Requirement`:

```text
transformers>=4.43,<5
huggingface_hub>=0.23,<1
accelerate>=0.33,<2
safetensors>=0.4,<1
sentencepiece>=0.2,<1
```

Install only missing or incompatible requirements using the selected runtime
Python and `PIP_CACHE_DIR=<prefix>/pip-cache`. Re-run the check after pip
finishes and fail if any requirement is still unsatisfied.

- [ ] **Step 3: Write the combined environment file**

The generated file must source FlagGems first and, in compiled mode, source
PyTorch second so the compiled Python remains first on `PATH`. It must export:

```text
MODEL_INFERENCE_PREFIX
MODEL_INFERENCE_ROOT
MODEL_INFERENCE_PYTHON
HF_HOME
HF_HUB_CACHE
TRANSFORMERS_CACHE
PIP_CACHE_DIR
MODEL_INFERENCE_ARTIFACTS
TRITON_DUMP_DIR
TRITON_ALWAYS_COMPILE
TRITON_KERNEL_DUMP
```

It must be source-only, create cache/artifact directories, and preserve
FlagTree Triton/LLVM/CUDA path variables.

- [ ] **Step 4: Run static and shell validation**

Run:

```bash
bash -n 3-install-model-inference.sh
bash tests/test-model-inference.sh
```

Expected: PASS without requiring a model download.

- [ ] **Step 5: Commit runtime selection**

```bash
git add 3-install-model-inference.sh
git commit -m "feat: select PyTorch runtime for model inference"
```

### Task 5: Implement Model Download, Verification, and Inference Execution

**Files:**
- Modify: `3-install-model-inference.sh`

**Interfaces:**
- `--model-path` bypasses download and uses an existing local model.
- Otherwise `--local-dir` controls the model directory; default is
  `<prefix>/models/<model-id with / replaced by ->`.
- `--skip-download` requires a usable local model path.
- Default execution downloads/reuses the model and runs inference.
- Produces `<prefix>/logs/inference-YYYYMMDD_HHMMSS.log` and
  `<prefix>/artifacts/triton-dumps/<timestamp>/`.

- [ ] **Step 1: Add model download**

Use the selected runtime Python and `huggingface_hub.snapshot_download` with
`repo_id`, `revision`, and `local_dir`. Require `config.json` after download.
Set `HF_HOME`, `HF_HUB_CACHE`, and `TRANSFORMERS_CACHE` under the model
inference prefix unless the user already exported them.

- [ ] **Step 2: Add stack preflight**

Unless `--skip-test` is set, run a Python check that imports `torch`, `triton`,
`flag_gems`, verifies `torch.cuda.is_available()`, prints package paths and
GPU name, and performs a small CUDA tensor operation inside
`flag_gems.use_gems()`.

- [ ] **Step 3: Add inference execution**

Set `TRITON_ALWAYS_COMPILE=1`, `TRITON_KERNEL_DUMP=1`, and a timestamped
`TRITON_DUMP_DIR`. Invoke:

```bash
"$RUNTIME_PYTHON" "$SOURCE_DIR/examples/run_llm_with_flaggems.py" \
  --model-path "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS"
```

Append `--compare-baseline` when requested. Pipe combined stdout/stderr
through `tee` to the timestamped log. Require `flaggems_text:` followed by a
non-empty generated line. Print the artifact directory and log path.

- [ ] **Step 4: Run the real integration test**

Run with a GPU and either network access or an existing HuggingFace cache:

```bash
RUN_MODEL_INFERENCE_INTEGRATION=1 \
MAX_NEW_TOKENS=8 \
bash tests/test-model-inference.sh
```

Expected: the selected runtime reports CUDA, the log contains PyTorch,
Triton, FlagTree, FlagGems, GPU, and non-empty generated text, and at least
one non-empty Triton artifact exists.

- [ ] **Step 5: Commit the complete execution flow**

```bash
git add 3-install-model-inference.sh
git commit -m "feat: run HuggingFace inference with FlagGems"
```

### Task 6: Update README and Model Asset Policy

**Files:**
- Modify: `README.md`
- Modify: `model-inference/README.md`

- [ ] **Step 1: Document prerequisites and command order**

Document:

```bash
bash 0-install-flagtree.sh
bash 1-install-flaggems.sh
# optional:
bash 2-install-pytorch.sh
bash 3-install-model-inference.sh
```

Explain that `2-install-pytorch.sh` is optional, that compiled PyTorch is
automatically preferred when usable, and that no root privileges are needed.

- [ ] **Step 2: Document useful commands**

Include examples for:

```bash
bash 3-install-model-inference.sh --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0
bash 3-install-model-inference.sh --skip-inference
bash 3-install-model-inference.sh --pytorch-mode wheel
source ../flagOS-installed/model-inference/env-model-inference.sh
```

Document model path, prompt, token limit, custom prefixes, logs, Triton dumps,
and the distinction between checked-in small assets and downloaded large
weights.

- [ ] **Step 3: Run documentation and static checks**

Run:

```bash
git diff --check
bash tests/test-model-inference.sh
```

Expected: no whitespace errors and static checks pass.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md model-inference/README.md
git commit -m "docs: document model inference installer"
```

### Task 7: Final Verification

**Files:**
- Verify all changed files; no new source changes unless a test exposes a
  concrete defect.

- [ ] **Step 1: Run shell and Python static checks**

```bash
bash -n 0-install-flagtree.sh
bash -n 1-install-flaggems.sh
bash -n 2-install-pytorch.sh
bash -n 3-install-model-inference.sh
bash tests/test-model-inference.sh
```

- [ ] **Step 2: Run the complete GPU-backed flow**

```bash
RUN_MODEL_INFERENCE_INTEGRATION=1 \
MAX_NEW_TOKENS=8 \
bash tests/test-model-inference.sh
```

If the model is already present, pass `--model-path` through the test
environment or reuse the prefix cache. Do not claim integration success unless
the generated text and Triton artifact assertions both pass.

- [ ] **Step 3: Review the final diff**

```bash
git diff --check HEAD~6..HEAD
git status --short --branch
git log --oneline -8
```

Confirm that no model weights, cache files, logs, or Triton dumps are tracked.

