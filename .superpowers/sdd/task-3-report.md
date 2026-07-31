# Task 3 Report: Model Inference Installer Skeleton

## Summary

Implemented `3-install-model-inference.sh` as a validation-only skeleton for the model inference installer.

The script now provides:
- `set -euo pipefail`, `SCRIPT_DIR`, `die`, `note`, and `require_command`.
- Help text covering every Task 3 option.
- CLI parsing for all required options.
- Canonicalized paths without creating install directories.
- Newline rejection for path options.
- Root (`/`) rejection for writable prefixes.
- User-owned ancestor validation for configured paths.
- Ubuntu 22.04 and `x86_64` platform checks.
- Required command validation for `git`, `tar`, `awk`, `sed`, `find`, `python3`, and `curl` or `wget`.
- `nvidia-smi` validation unless `--skip-test` is set.
- Prerequisite checks for:
  - `<flagtree-prefix>/env-flagtree.sh`
  - `<flaggems-prefix>/env-flaggems.sh`
  - `<flagtree-prefix>/python/bin/python`
  - `<source-dir>/examples/run_llm_with_flaggems.py`

The script intentionally does not implement runtime selection, dependency installation, model download, preflight, or inference execution.

## Verification

Red test before implementation:

```text
$ bash tests/test-model-inference.sh
/media/disk/fengjingge/src/flagOS/flagOS-installers/3-install-model-inference.sh: line 1: haha: command not found
```

Verification after implementation:

```text
$ bash tests/test-model-inference.sh
model-inference static checks passed
```

Additional syntax/help checks:

```text
$ bash -n 3-install-model-inference.sh
exit 0

$ bash 3-install-model-inference.sh --help
exit 0
```

## Self-Review

Checked the implementation against the Task 3 brief:
- All required options are accepted and documented in `--help`.
- Required platform checks are present.
- `nvidia-smi` is skipped only when `--skip-test` is used.
- PyTorch prefix is only required in `compiled` mode, not in `auto` or `wheel`.
- Required FlagTree, FlagGems, Python, and local entrypoint files are validated before any install action.
- No dependency install, download, preflight, runtime selection, or inference behavior was added.

## Concerns

No blocking concerns for Task 3 static expectations. The repository does not expose a subagent review tool in this session, so the review was performed manually.
