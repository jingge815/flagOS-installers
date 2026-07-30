# FlagTree Installer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `0-install-flagtree.sh` installer plus README updates so Ubuntu 22.04 users can clone, build, install, and verify FlagTree without root, assuming NVIDIA driver access is already working.

**Architecture:** Keep the implementation shell-first and linear. The installer owns a user-space prefix, reuses previously validated artifacts inside that prefix, clones `FlagTree` into a separate source directory pinned to a fixed commit, then builds and installs a wheel into a standalone Python environment and writes a reusable environment script plus validation example.

**Tech Stack:** Bash, Git, Python standalone archive, pip, prebuilt LLVM archive, Triton/NVIDIA dependency archive, Ubuntu `deb` extraction with `dpkg-deb`, Markdown

## Global Constraints

- Target OS is Ubuntu 22.04 on `x86_64`.
- `nvidia-smi` must already be available and report a usable NVIDIA GPU.
- The installer must not require or use `sudo`.
- Default source directory is `./FlagTree`.
- Default install prefix is `../flagOS-installed/flagTree`.
- Source repository is `https://github.com/KernelLLM/FlagTree.git`.
- Source branch is `common-ir-triton35`.
- Pinned commit is `317f15a426466633c4f37f164b2c58ae9c31bd03`.
- Later runs must reuse existing valid downloads and installed dependencies instead of forcing a full reinstall.
- The installer must generate an environment script and a simple validation path.
- `README.md` must explicitly state that NVIDIA driver installation is out of scope.

---

### Task 1: Replace Placeholder Installer With A Standalone User-Space Build Script

**Files:**
- Modify: `0-install-flagtree.sh`
- Create: `matmul_sm80.py`

**Interfaces:**
- Consumes: existing top-level installer entrypoint `0-install-flagtree.sh`
- Produces: executable installer `0-install-flagtree.sh`, validation example `matmul_sm80.py`, generated runtime file `${PREFIX}/env-flagtree.sh`, copied validation file `${PREFIX}/examples/matmul_sm80.py`

- [ ] **Step 1: Write the failing installer smoke test commands**

Run these commands before editing so the current failure mode is explicit:

```bash
bash 0-install-flagtree.sh --help
```

Expected: output is incorrect because the file currently contains placeholder text instead of a valid installer.

- [ ] **Step 2: Replace `0-install-flagtree.sh` with a complete installer skeleton**

Write the file with these top-level sections and constants:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEFAULT_PREFIX=$(cd -- "$SCRIPT_DIR/../flagOS-installed/flagTree" 2>/dev/null && pwd -P || true)
DEFAULT_SOURCE_DIR="$SCRIPT_DIR/FlagTree"
MAX_JOBS=${MAX_JOBS:-8}
RUN_TEST=1

FLAGTREE_REPOSITORY=https://github.com/KernelLLM/FlagTree.git
FLAGTREE_BRANCH=common-ir-triton35
FLAGTREE_REVISION=317f15a426466633c4f37f164b2c58ae9c31bd03
LLVM_ARCHIVE=llvm-7d5de303-ubuntu-x64.tar.gz
LLVM_URL=https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-7d5de303-ubuntu-x64.tar.gz
TRITON_DEPS_ARCHIVE=build-deps-triton_3.5.x-linux-x64.tar.gz
TRITON_DEPS_URL=https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.5.x-linux-x64.tar.gz
PYTHON_ARCHIVE=cpython-3.10.20+20260718-x86_64-unknown-linux-gnu-install_only.tar.gz
PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260718/cpython-3.10.20%2B20260718-x86_64-unknown-linux-gnu-install_only.tar.gz
```

Include helper functions with exact names:

```bash
usage() { :; }
die() { printf '错误：%s\n' "$*" >&2; exit 1; }
note() { printf '\n==> %s\n' "$*"; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "缺少命令 $1。请先安装：$2"; }
```

- [ ] **Step 3: Add argument parsing and path normalization**

Implement support for `--prefix`, `--source-dir`, `--max-jobs`, `--skip-test`, and `--help` with this structure:

```bash
PREFIX=
SOURCE_DIR=$DEFAULT_SOURCE_DIR

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      [[ $# -ge 2 ]] || die '--prefix 缺少目录参数。'
      PREFIX=$2
      shift 2
      ;;
    --source-dir)
      [[ $# -ge 2 ]] || die '--source-dir 缺少目录参数。'
      SOURCE_DIR=$2
      shift 2
      ;;
    --max-jobs)
      [[ $# -ge 2 && $2 =~ ^[1-9][0-9]*$ ]] || die '--max-jobs 必须是正整数。'
      MAX_JOBS=$2
      shift 2
      ;;
    --skip-test)
      RUN_TEST=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "未知参数：$1"
      ;;
  esac
done

if [[ -z "${PREFIX:-}" ]]; then
  PREFIX="$SCRIPT_DIR/../flagOS-installed/flagTree"
fi
mkdir -p -- "$PREFIX"
PREFIX=$(cd -- "$PREFIX" && pwd -P)
mkdir -p -- "$(dirname -- "$SOURCE_DIR")"
if [[ -e "$SOURCE_DIR" ]]; then
  SOURCE_DIR=$(cd -- "$SOURCE_DIR" && pwd -P)
else
  SOURCE_PARENT=$(cd -- "$(dirname -- "$SOURCE_DIR")" && pwd -P)
  SOURCE_DIR="$SOURCE_PARENT/$(basename -- "$SOURCE_DIR")"
fi
```

- [ ] **Step 4: Implement platform checks and download helpers**

Add functions for platform validation and safe downloads:

```bash
download_tarball() {
  local url=$1
  local destination=$2
  local temporary="${destination}.part.$$"

  if [[ -s "$destination" ]] && tar -tzf "$destination" >/dev/null 2>&1; then
    return
  fi

  rm -f -- "$destination" "$temporary"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "$temporary" "$url"
  else
    wget --output-document="$temporary" "$url"
  fi
  tar -tzf "$temporary" >/dev/null
  mv -- "$temporary" "$destination"
}
```

```bash
check_platform() {
  [[ -r /etc/os-release ]] || die '无法读取 /etc/os-release。'
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 22.04 ]] || die "需要 Ubuntu 22.04，当前为 ${PRETTY_NAME:-未知系统}。"
  [[ $(uname -m) == x86_64 ]] || die "需要 x86_64，当前为 $(uname -m)。"
  require_command nvidia-smi 'NVIDIA 驱动'
  require_command git 'git'
  require_command tar 'tar'
  require_command dpkg-deb 'dpkg-deb'
  require_command apt-get 'apt-get'
  require_command awk 'awk'
  require_command sed 'sed'
  require_command find 'findutils'
  require_command make 'make'
  require_command cc 'gcc'
  require_command c++ 'g++'
  require_command ar 'binutils'
  require_command ld 'binutils'
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    die '缺少 curl 或 wget。'
  fi
  nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi 不可用，请先确认驱动和 GPU。'
}
```

- [ ] **Step 5: Implement reusable dependency installation functions**

Add prefix layout variables and functions mirroring the proven reference flow but without machine-specific hardcoded prefixes:

```bash
DOWNLOADS="$PREFIX/downloads"
DEBS="$DOWNLOADS/debs"
PYTHON="$PREFIX/python-3.10.20"
PYTHON_LINK="$PREFIX/python"
LLVM="$PREFIX/llvm-7d5de303"
TRITON_HOME="$PREFIX/triton-home"
NVIDIA_TOOLCHAIN="$PREFIX/nvidia-toolchain-12.8"
SYSROOT="$PREFIX/sysroot"
BUILD="$PREFIX/build/flagtree-cmake"
WHEELS="$PREFIX/wheels"
ENV_FILE="$PREFIX/env-flagtree.sh"
EXAMPLES_DIR="$PREFIX/examples"
MLIR_DUMP_DIR="$PREFIX/mlir-dumps"
```

Implement functions with these names:

```bash
install_python()
install_llvm()
install_triton_build_dependencies()
download_deb()
install_development_headers()
```

Inside `install_python()`, install these packages:

```bash
"$PYTHON/bin/python" -m pip install --upgrade --no-cache-dir \
  'pip<27' setuptools==83.0.0 wheel==0.47.0 \
  cmake==3.31.10 ninja==1.13.0 pybind11==3.0.4 lit==18.1.8 \
  numpy==1.26.4 pytest==8.3.5
"$PYTHON/bin/python" -m pip install --upgrade --no-cache-dir \
  torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Inside `install_development_headers()`, fetch and unpack:

```bash
for package in zlib1g zlib1g-dev libxml2 libxml2-dev; do
  :
done
```

Retarget any absolute symlinks after extraction:

```bash
ln -sfn 'libz.so.1' "$SYSROOT/usr/lib/x86_64-linux-gnu/libz.so"
ln -sfn 'libxml2.so.2' "$SYSROOT/usr/lib/x86_64-linux-gnu/libxml2.so"
```

- [ ] **Step 6: Implement source checkout, environment generation, build, and validation helpers**

Add these functions:

```bash
write_environment_file()
checkout_flagtree()
build_flagtree()
install_example()
run_validation()
```

`checkout_flagtree()` must:

```bash
if [[ ! -e "$SOURCE_DIR" ]]; then
  git clone --branch "$FLAGTREE_BRANCH" --single-branch "$FLAGTREE_REPOSITORY" "$SOURCE_DIR"
elif [[ ! -d "$SOURCE_DIR/.git" ]]; then
  die "源码目录已存在但不是 Git 仓库：$SOURCE_DIR"
fi

git -C "$SOURCE_DIR" diff --quiet || die "源码目录存在已跟踪的修改：$SOURCE_DIR"
git -C "$SOURCE_DIR" diff --cached --quiet || die "源码目录存在暂存修改：$SOURCE_DIR"
git -C "$SOURCE_DIR" fetch --depth 1 origin "$FLAGTREE_REVISION" >/dev/null 2>&1 || true
git -C "$SOURCE_DIR" checkout --detach "$FLAGTREE_REVISION"
[[ $(git -C "$SOURCE_DIR" rev-parse HEAD) == "$FLAGTREE_REVISION" ]] || die 'FlagTree 源码提交与预期不一致。'
```

`build_flagtree()` must set these build variables before invoking `pip wheel`:

```bash
export TRITON_BUILD_PROTON=OFF
export TRITON_BUILD_UT=OFF
export TRITON_BUILD_WITH_CCACHE=OFF
export TRITON_PARALLEL_LINK_JOBS=${TRITON_PARALLEL_LINK_JOBS:-1}
export MAX_JOBS
export TRITON_APPEND_CMAKE_ARGS="-DLLVM_ENABLE_WERROR=OFF -DTRITON_BUILD_UT=OFF -DZLIB_ROOT=$SYSROOT/usr -DZLIB_LIBRARY=$SYSROOT/usr/lib/x86_64-linux-gnu/libz.so -DZLIB_INCLUDE_DIR=$SYSROOT/usr/include -DLIBXML2_LIBRARY=$SYSROOT/usr/lib/x86_64-linux-gnu/libxml2.so -DLIBXML2_INCLUDE_DIR=$SYSROOT/usr/include/libxml2"
```

The wheel build/install command must be:

```bash
(
  cd "$SOURCE_DIR"
  "$PYTHON/bin/python" -m pip wheel . --no-build-isolation --no-deps --wheel-dir "$WHEELS"
)
```

`write_environment_file()` must export:

```bash
export PATH="$PREFIX/python/bin:$LLVM/bin:$NVIDIA_TOOLCHAIN/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM/lib:$SYSROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$SYSROOT/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LLVM_SYSPATH="$LLVM"
export MLIR_DIR="$LLVM/lib/cmake/mlir"
export LLVM_DIR="$LLVM/lib/cmake/llvm"
export LLD_DIR="$LLVM/lib/cmake/lld"
export TRITON_HOME="$TRITON_HOME"
export TRITON_CACHE_DIR="$PREFIX/cache"
export TRITON_BUILD_DIR="$BUILD"
export TRITON_PTXAS_PATH="$NVIDIA_TOOLCHAIN/bin/ptxas"
export TRITON_CUOBJDUMP_PATH="$NVIDIA_TOOLCHAIN/bin/cuobjdump"
export TRITON_NVDISASM_PATH="$NVIDIA_TOOLCHAIN/bin/nvdisasm"
export TRITON_LIBDEVICE_PATH="$NVIDIA_TOOLCHAIN/nvvm/libdevice/libdevice.10.bc"
export FLAGTREE_IR_DUMP_DIR="$MLIR_DUMP_DIR"
```

`install_example()` must copy `matmul_sm80.py` into `${PREFIX}/examples/matmul_sm80.py`.

`run_validation()` must:

```bash
source "$ENV_FILE"
mkdir -p "$MLIR_DUMP_DIR"
"$PYTHON/bin/python" -c 'import flagtree, torch, triton; print("imports ok")'
"$PYTHON/bin/python" "$EXAMPLES_DIR/matmul_sm80.py"
```

- [ ] **Step 7: Add a simple Triton validation program at `matmul_sm80.py`**

Create `matmul_sm80.py` with the same structure as the NVIDIA reference example:

```python
#!/usr/bin/env python3
import os
import pathlib

import torch
import triton
import triton.language as tl
```

Use a single `@triton.jit` matmul kernel, a `matmul(a, b)` wrapper, and `main()` that:

```python
dump_dir = pathlib.Path(os.environ.get("FLAGTREE_IR_DUMP_DIR", ""))
if dump_dir:
    dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLIR_ENABLE_DUMP", "1")
    os.environ.setdefault("MLIR_DUMP_PATH", str(dump_dir))
```

Then allocate CUDA FP16 tensors, compare against `torch.matmul`, and print:

```python
print("target:", triton.runtime.driver.active.get_current_target())
print("matmul matches torch")
print("mlir dump dir:", dump_dir if dump_dir else "not enabled")
```

- [ ] **Step 8: Run the installer help path and shell syntax checks**

Run:

```bash
bash -n 0-install-flagtree.sh
bash 0-install-flagtree.sh --help
python3 -m py_compile matmul_sm80.py
```

Expected:

- `bash -n` exits successfully
- `--help` prints usage with `--prefix`, `--source-dir`, `--max-jobs`, `--skip-test`
- `py_compile` exits successfully

- [ ] **Step 9: Commit**

If this directory is inside a Git repository, run:

```bash
git add 0-install-flagtree.sh matmul_sm80.py
git commit -m "feat: add standalone FlagTree installer"
```

If this directory is not a Git repository, record that commit was skipped.

### Task 2: Rewrite README For One-Command Installation And Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: installer CLI from `0-install-flagtree.sh`, validation example `matmul_sm80.py`
- Produces: top-level usage documentation for source checkout location, install prefix, prerequisites, rerun behavior, environment setup, and verification steps

- [ ] **Step 1: Write the failing documentation check**

Run:

```bash
sed -n '1,220p' README.md
```

Expected: README is still a draft and does not provide a complete standalone FlagTree install flow.

- [ ] **Step 2: Replace the draft FlagTree section with a practical usage guide**

Rewrite `README.md` so the FlagTree section includes these exact topics:

```md
# flagOS-installers

## 环境前提

- Ubuntu 22.04 x86_64
- 已安装并可使用 NVIDIA 驱动
- `nvidia-smi` 可以正常运行
- 不需要 root，但需要基础命令和联网下载能力

## 1. 安装 FlagTree

- 源码默认下载到 `./FlagTree`
- 默认安装目录是 `../flagOS-installed/flagTree`
- 代码固定到 `common-ir-triton35` 分支上的提交 `317f15a426466633c4f37f164b2c58ae9c31bd03`
```

Document the exact install command:

```bash
bash 0-install-flagtree.sh
```

Document override examples:

```bash
bash 0-install-flagtree.sh --prefix /path/to/flagTree
bash 0-install-flagtree.sh --source-dir /path/to/FlagTree --max-jobs 16
```

Document rerun behavior in plain language: if valid downloads, standalone Python, LLVM, Triton/NVIDIA bundle, sysroot, or source checkout already exist, the installer reuses them and only rebuilds what is necessary.

- [ ] **Step 3: Add environment setup and validation commands**

README must include:

```bash
source ../flagOS-installed/flagTree/env-flagtree.sh
python ../flagOS-installed/flagTree/examples/matmul_sm80.py
```

Also include a short note that intermediate compiler artifacts are written under:

```text
../flagOS-installed/flagTree/mlir-dumps
```

And explicitly state:

```text
本脚本不会安装 NVIDIA 驱动；目标机器必须提前满足 `nvidia-smi` 可用。
```

- [ ] **Step 4: Validate the rewritten README**

Run:

```bash
sed -n '1,260p' README.md
```

Expected: README now describes the standalone FlagTree flow clearly and matches the installer defaults and pinned commit.

- [ ] **Step 5: Commit**

If this directory is inside a Git repository, run:

```bash
git add README.md
git commit -m "docs: document standalone FlagTree install flow"
```

If this directory is not a Git repository, record that commit was skipped.

### Task 3: End-To-End Verification Of The Installer Contract

**Files:**
- Modify: `0-install-flagtree.sh`
- Modify: `README.md`
- Modify: `matmul_sm80.py`

**Interfaces:**
- Consumes: completed installer, README, and validation program from Tasks 1 and 2
- Produces: verified shell syntax, verified Python syntax, and a checked contract between code defaults and documented commands

- [ ] **Step 1: Run a focused contract check on defaults**

Run:

```bash
rg -n "common-ir-triton35|317f15a426466633c4f37f164b2c58ae9c31bd03|../flagOS-installed/flagTree|./FlagTree|mlir-dumps" 0-install-flagtree.sh README.md matmul_sm80.py
```

Expected: the pinned commit, branch, default install prefix, source path, and MLIR dump location are all present and consistent.

- [ ] **Step 2: Run final local verification commands**

Run:

```bash
bash -n 0-install-flagtree.sh
python3 -m py_compile matmul_sm80.py
bash 0-install-flagtree.sh --help
```

Expected:

- no shell syntax errors
- no Python syntax errors
- help text matches README options

- [ ] **Step 3: Check for accidental references to machine-specific paths**

Run:

```bash
rg -n "/media/disk/fengjingge/software/flagos-nvidia|A800|triton_v3.5.x" 0-install-flagtree.sh README.md matmul_sm80.py
```

Expected: no matches, because the new installer must be portable across Ubuntu 22.04 machines with working NVIDIA drivers.

- [ ] **Step 4: Final edit pass if the checks expose mismatches**

If any command from Steps 1 through 3 shows inconsistent defaults, edit the affected files so these exact values are aligned everywhere:

```text
branch: common-ir-triton35
commit: 317f15a426466633c4f37f164b2c58ae9c31bd03
source dir default: ./FlagTree
prefix default: ../flagOS-installed/flagTree
mlir dump dir: <prefix>/mlir-dumps
```

- [ ] **Step 5: Commit**

If this directory is inside a Git repository, run:

```bash
git add 0-install-flagtree.sh README.md matmul_sm80.py
git commit -m "chore: verify standalone FlagTree installer contract"
```

If this directory is not a Git repository, record that commit was skipped.
