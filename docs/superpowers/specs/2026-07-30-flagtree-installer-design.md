# FlagTree Installer Design

Date: 2026-07-30

## Goal

Add a standalone `0-install-flagtree.sh` installer and update `README.md` so a fresh Ubuntu 22.04 machine can install and verify FlagTree without root privileges, assuming NVIDIA driver and GPU access are already available.

The installer must:

- Clone FlagTree source into a local `FlagTree/` directory.
- Pin the source to a fixed commit from `common-ir-triton35`.
- Install build/runtime dependencies into a user-owned prefix.
- Reuse already-downloaded or already-installed artifacts on later runs instead of rebuilding everything.
- Provide a simple validation path that can confirm FlagTree is usable and can emit intermediate compilation artifacts such as MLIR.

## Assumptions And Non-Goals

### Assumptions

- Target OS is Ubuntu 22.04 on `x86_64`.
- `nvidia-smi` is available and reports a usable NVIDIA GPU.
- Basic userland tools are present: `git`, `tar`, `dpkg-deb`, `apt-get`, `awk`, `sed`, `find`, `make`, `cc`, `c++`, `ar`, `ld`, and either `curl` or `wget`.
- Network access is available for downloading source and binary dependencies.

### Non-Goals

- The script will not install NVIDIA drivers.
- The script will not require or use `sudo`.
- The script will not build heavyweight dependencies like LLVM fully from source.

## Installation Layout

### Source Location

- Default source directory: `./FlagTree` relative to `flagOS-installers`.
- Source repository: `https://github.com/KernelLLM/FlagTree.git`
- Source branch: `common-ir-triton35`
- Pinned commit: `317f15a426466633c4f37f164b2c58ae9c31bd03`

The installer will clone into `./FlagTree` if missing, then check out the pinned commit in detached HEAD state.

If `./FlagTree` already exists:

- If it is not a Git repository, fail fast.
- If it has tracked or staged changes, fail fast to avoid overwriting user work.
- If it is already on the pinned commit, reuse it.

### Install Prefix

- Default prefix: `../flagOS-installed/flagTree`
- Allow override with `--prefix DIR`

The prefix contains all user-owned installation artifacts:

- Standalone Python
- Download cache
- Extracted Ubuntu `deb` sysroot headers/libs
- LLVM package
- Triton/NVIDIA helper binaries
- Build directory
- Built wheels
- Environment script
- Example and validation scripts

Later runs should reuse existing valid artifacts in the prefix instead of re-downloading or re-installing them.

## Installer Interface

The script will support:

- `--prefix DIR`: override the installation prefix
- `--source-dir DIR`: override the source checkout directory
- `--max-jobs N`: control parallel build jobs, default `8`
- `--skip-test`: skip post-install verification
- `-h` or `--help`: print usage

## Dependency Strategy

The installer uses a fully user-space dependency model.

### Python

Download a standalone CPython build and install Python packages with `pip` inside the prefix. This avoids system Python mutation and avoids root.

### LLVM

Download a prebuilt LLVM archive compatible with the Triton 3.5-based FlagTree flow, unpack it under the prefix, and expose it via the generated environment script.

### Triton/NVIDIA Tooling

Download the prepackaged Triton 3.5 NVIDIA build dependencies archive, then expose tools like `ptxas`, `cuobjdump`, `nvdisasm`, and `libdevice` from the prefix.

### Sysroot Headers And Libraries

Use `apt-get download` to fetch Ubuntu 22.04 `deb` packages such as:

- `zlib1g`
- `zlib1g-dev`
- `libxml2`
- `libxml2-dev`

Extract them into a prefix-owned sysroot with `dpkg-deb -x`, then point the FlagTree/Triton build at those headers and shared libraries.

## Execution Flow

The script runs these stages:

1. Validate host platform and required commands.
2. Normalize prefix and source paths.
3. Prepare download/cache directories.
4. Install or reuse standalone Python.
5. Install or reuse LLVM bundle.
6. Install or reuse Triton/NVIDIA dependency bundle.
7. Install or reuse extracted sysroot headers/libs from downloaded `deb` packages.
8. Clone or reuse `FlagTree`, then pin to commit `317f15a426466633c4f37f164b2c58ae9c31bd03`.
9. Build a wheel from FlagTree and install it into the prefix Python.
10. Generate an `env-flagtree.sh` environment script.
11. Install a simple Triton matmul validation program.
12. Run verification unless `--skip-test` is passed.

## Reuse And Idempotency Rules

First run assumes `../flagOS-installed/flagTree` is empty.

Subsequent runs should reuse existing artifacts when they pass lightweight validity checks:

- Existing tarballs are reused if they are present and extractable.
- Existing standalone Python is reused if the interpreter exists.
- Existing LLVM is reused if `llvm-config --version` matches expectation.
- Existing Triton/NVIDIA bundle is reused if the expected binaries and files exist.
- Existing sysroot is reused if the expected headers and libraries exist.
- Existing source checkout is reused if it is a clean Git repository.

The build directory itself may be recreated on each build to keep behavior simple and predictable.

## Validation Design

Validation has two levels.

### Basic Import Check

Run the prefix Python with imports for the installed package and confirm the environment script works.

### GPU Compilation Check

Run a small Triton matmul example similar to the existing NVIDIA reference flow. The example should:

- Create FP16 CUDA tensors
- Launch a Triton kernel
- Synchronize
- Compare with `torch.matmul`
- Print the active target

### Intermediate IR / MLIR Output

The validation path should also document how to enable compiler dumps for debugging intermediate stages. The implementation should prefer simple environment-variable-based dumping into a known directory under the prefix so the user can inspect generated IR files after a run.

If upstream variable names are version-sensitive, the script and README should describe the exact variables used by this installer rather than claiming generic support without evidence.

## README Changes

`README.md` should be rewritten from the current draft into a practical usage document for this repository section.

The FlagTree section should state:

- Supported platform: Ubuntu 22.04, `x86_64`
- Required precondition: NVIDIA driver and GPU access already work, and `nvidia-smi` must be available
- Default source directory: `./FlagTree`
- Default install prefix: `../flagOS-installed/flagTree`
- One-command install example
- Re-run behavior: existing valid downloads and installed dependencies are reused
- How to source the generated environment script
- How to run the validation example
- Where intermediate compiler artifacts are emitted

## Error Handling

The installer should use `set -euo pipefail` and fail early with short actionable messages.

Important failure cases:

- Unsupported OS or architecture
- Missing `nvidia-smi`
- Missing base tools like `git` or `dpkg-deb`
- Existing source directory that is not a Git repository
- Existing source repository with local modifications
- Downloaded archives that fail validation
- Build that does not produce a wheel

## Implementation Notes

- Keep the shell logic simple and linear.
- Prefer helper functions for repeated tasks such as command checks, tarball download, stage logging, and fatal exits.
- Avoid unnecessary configurability beyond `--prefix`, `--source-dir`, `--max-jobs`, and `--skip-test`.
- Mirror the proven structure from `/media/disk/fengjingge/software/flagos-nvidia/scripts/install_flagtree_triton35_nvidia.sh` where practical, but remove machine-specific hardcoded paths.

## Risks

- Upstream Triton/FlagTree debug dump knobs may differ across revisions; verification must confirm the chosen mechanism works with the pinned commit.
- Some target machines may lack enough baseline userland tools for a true one-command install without admin help.
- Prebuilt binary dependency URLs may change in the future; pinning exact URLs reduces ambiguity but not long-term availability risk.

## Open Questions Resolved

- Default install prefix is fixed to `../flagOS-installed/flagTree`.
- Existing valid installation artifacts should be reused on later runs instead of forcing a clean reinstall.
- README must explicitly state that NVIDIA driver installation is out of scope and `nvidia-smi` is required.
