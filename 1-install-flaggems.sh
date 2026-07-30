#!/usr/bin/env bash
# Install FlagGems from a pinned source checkout without root.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEFAULT_SOURCE_DIR="$SCRIPT_DIR/FlagTree/FlagGems"
DEFAULT_PREFIX="$SCRIPT_DIR/../flagOS-installed/flagGems"
DEFAULT_FLAGTREE_PREFIX="$SCRIPT_DIR/../flagOS-installed/flagTree"

FLAGGEMS_REPOSITORY=https://github.com/flagos-ai/FlagGems.git
FLAGGEMS_BRANCH=master
FLAGGEMS_REVISION=bfbd21ca85dbfa84061fe90a7ced899c85238b13

RUN_TEST=1
FORCE_RECLONE=0
WITH_CPP=0

usage() {
  cat <<EOF
用法：
  bash 1-install-flaggems.sh [选项]

选项：
  --prefix DIR           FlagGems 安装目录，默认：../flagOS-installed/flagGems
  --source-dir DIR       FlagGems 源码目录，默认：./FlagTree/FlagGems
  --flagtree-prefix DIR  0-install-flagtree.sh 安装目录，默认：../flagOS-installed/flagTree
  --skip-test            跳过安装后的 CUDA smoke test
  --force-reclone        删除已有 FlagGems 源码目录后重新 clone
  --with-cpp             额外尝试构建 FlagGems CUDA C++ wrapped operators
  -h, --help             显示帮助

固定版本：
  仓库：$FLAGGEMS_REPOSITORY
  分支：$FLAGGEMS_BRANCH
  提交：$FLAGGEMS_REVISION

说明：
  本脚本不需要 root。请先成功运行 0-install-flagtree.sh；本脚本会复用
  ../flagOS-installed/flagTree 中的 Python、PyTorch、FlagTree/Triton、LLVM 和
  NVIDIA 用户态编译工具。如果 ../flagOS-installed/flagGems 已安装部分依赖，
  pip 会按需复用或更新，源码目录会固定到上面的 commit。
EOF
}

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

note() {
  printf '\n==> %s\n' "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "缺少命令 $1。"
}

canonicalize_existing_or_parent() {
  local path=$1
  if [[ -e "$path" ]]; then
    (cd -- "$path" && pwd -P)
  else
    mkdir -p -- "$(dirname -- "$path")"
    local parent
    parent=$(cd -- "$(dirname -- "$path")" && pwd -P)
    printf '%s/%s\n' "$parent" "$(basename -- "$path")"
  fi
}

check_platform() {
  [[ -r /etc/os-release ]] || die '无法读取 /etc/os-release。'
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 22.04 ]] || \
    die "需要 Ubuntu 22.04，当前为 ${PRETTY_NAME:-未知系统}。"
  [[ $(uname -m) == x86_64 ]] || die "需要 x86_64，当前为 $(uname -m)。"

  require_command git
  require_command awk
  require_command sed
  require_command find
  require_command nvidia-smi
  nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi 不可用，请先确认 NVIDIA 驱动和 GPU。'
}

activate_flagtree_environment() {
  ENV_FLAGTREE="$FLAGTREE_PREFIX/env-flagtree.sh"
  PYTHON_BIN="$FLAGTREE_PREFIX/python/bin/python"
  local user_triton_dump_dir=${TRITON_DUMP_DIR-}
  local user_mlir_enable_dump=${MLIR_ENABLE_DUMP-}
  local user_mlir_dump_path=${MLIR_DUMP_PATH-}
  local user_triton_kernel_dump=${TRITON_KERNEL_DUMP-}
  local user_triton_always_compile=${TRITON_ALWAYS_COMPILE-}

  [[ -f "$ENV_FLAGTREE" ]] || \
    die "找不到 FlagTree 环境脚本：$ENV_FLAGTREE。请先运行 0-install-flagtree.sh。"
  [[ -x "$PYTHON_BIN" ]] || \
    die "找不到 FlagTree Python：$PYTHON_BIN。请先运行 0-install-flagtree.sh。"

  # shellcheck disable=SC1090
  source "$ENV_FLAGTREE"

  export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PREFIX/pip-cache}
  export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$PREFIX/cache/triton}
  export TRITON_DUMP_DIR=${user_triton_dump_dir:-$PREFIX/triton-stage-dumps}
  export MLIR_ENABLE_DUMP=${user_mlir_enable_dump:-1}
  export MLIR_DUMP_PATH=${user_mlir_dump_path:-$PREFIX/mlir-dumps/flaggems-mlir-dump.mlir}
  export FLAGGEMS_IR_DUMP_DIR=${FLAGGEMS_IR_DUMP_DIR:-$PREFIX/mlir-dumps}
  export TRITON_KERNEL_DUMP=${user_triton_kernel_dump:-1}
  export TRITON_ALWAYS_COMPILE=${user_triton_always_compile:-1}

  if [[ -d "$MLIR_DUMP_PATH" ]]; then
    export MLIR_DUMP_PATH="$MLIR_DUMP_PATH/flaggems-mlir-dump.mlir"
  fi

  mkdir -p "$PREFIX" "$PREFIX/downloads" "$PREFIX/wheels" "$PIP_CACHE_DIR" \
    "$TRITON_CACHE_DIR" "$TRITON_DUMP_DIR" "$(dirname -- "$MLIR_DUMP_PATH")"

  note "Python: $("$PYTHON_BIN" --version 2>&1)"
  note "FlagTree 环境：$ENV_FLAGTREE"
}

checkout_flaggems() {
  if [[ "$FORCE_RECLONE" -eq 1 ]]; then
    [[ "$SOURCE_DIR" != / && -n "$SOURCE_DIR" ]] || die '源码目录不安全，拒绝删除。'
    rm -rf -- "$SOURCE_DIR"
  fi

  if [[ ! -e "$SOURCE_DIR" ]]; then
    mkdir -p -- "$(dirname -- "$SOURCE_DIR")"
    note "clone FlagGems 到：$SOURCE_DIR"
    git clone --branch "$FLAGGEMS_BRANCH" --single-branch "$FLAGGEMS_REPOSITORY" "$SOURCE_DIR"
  elif [[ ! -d "$SOURCE_DIR/.git" ]]; then
    die "源码目录已存在但不是 Git 仓库：$SOURCE_DIR"
  else
    note "复用已有 FlagGems 源码：$SOURCE_DIR"
  fi

  git -C "$SOURCE_DIR" diff --quiet || die "源码目录存在已跟踪的修改：$SOURCE_DIR"
  git -C "$SOURCE_DIR" diff --cached --quiet || die "源码目录存在暂存修改：$SOURCE_DIR"
  git -C "$SOURCE_DIR" fetch --depth 1 origin "$FLAGGEMS_REVISION" >/dev/null 2>&1 || true
  git -C "$SOURCE_DIR" checkout --detach "$FLAGGEMS_REVISION"
  [[ $(git -C "$SOURCE_DIR" rev-parse HEAD) == "$FLAGGEMS_REVISION" ]] || \
    die 'FlagGems 源码提交与预期不一致。'
}

install_build_requirements() {
  note '安装/确认 FlagGems Python 构建依赖'
  "$PYTHON_BIN" -m pip install --upgrade \
    'pip<27' 'setuptools>=64,<77' 'setuptools-scm>=8,<10' \
    wheel

  if [[ "$WITH_CPP" -eq 1 ]]; then
    "$PYTHON_BIN" -m pip install --upgrade \
      'cmake>=3.25,<4' ninja 'pybind11>=3,<4' 'scikit-build-core==0.12.2'
  fi
}

install_flaggems_python() {
  note '安装 FlagGems Python package'
  (
    cd "$SOURCE_DIR"
    "$PYTHON_BIN" -m pip install --no-build-isolation -e .
  )
}

install_flaggems_cpp() {
  [[ "$WITH_CPP" -eq 1 ]] || return 0

  note '尝试构建并安装 FlagGems CUDA C++ wrapped operators'
  local cpp_work_dir="$PREFIX/build/flaggems-cpp-cuda"
  local cpp_repo_dir="$cpp_work_dir/FlagGems"
  rm -rf -- "$cpp_work_dir"
  mkdir -p "$cpp_work_dir"
  git clone --shared "$SOURCE_DIR" "$cpp_repo_dir"
  (
    cd "$cpp_repo_dir"
    git checkout --detach "$FLAGGEMS_REVISION"
    sed -E 's/^name = "flag-gems-cpp-[A-Za-z0-9_]+"/name = "flag-gems-cpp-cuda"/' \
      cpp/pyproject.toml > cpp/pyproject.toml.tmp
    mv cpp/pyproject.toml.tmp cpp/pyproject.toml
    cd cpp
    CMAKE_ARGS='-DFLAGGEMS_BACKEND=CUDA -DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DCMAKE_BUILD_TYPE=Release' \
      "$PYTHON_BIN" -m pip install --no-build-isolation -v .
  )
}

write_environment_file() {
  local prefix_shell source_shell flagtree_shell
  printf -v prefix_shell '%q' "$PREFIX"
  printf -v source_shell '%q' "$SOURCE_DIR"
  printf -v flagtree_shell '%q' "$FLAGTREE_PREFIX"

  cat > "$ENV_FILE" <<EOF
#!/usr/bin/env bash
# Source this file before using this FlagGems installation.

if [[ "\${BASH_SOURCE[0]}" == "\$0" ]]; then
  echo "Source this file instead: source \${BASH_SOURCE[0]}" >&2
  exit 1
fi

FLAGGEMS_PREFIX=$prefix_shell
FLAGGEMS_SOURCE=$source_shell
FLAGTREE_PREFIX=$flagtree_shell

_flaggems_user_triton_dump_dir="\${TRITON_DUMP_DIR-}"
_flaggems_user_mlir_enable_dump="\${MLIR_ENABLE_DUMP-}"
_flaggems_user_mlir_dump_path="\${MLIR_DUMP_PATH-}"
_flaggems_user_triton_kernel_dump="\${TRITON_KERNEL_DUMP-}"
_flaggems_user_triton_always_compile="\${TRITON_ALWAYS_COMPILE-}"

# shellcheck disable=SC1091
source "\$FLAGTREE_PREFIX/env-flagtree.sh"

export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-\$FLAGGEMS_PREFIX/pip-cache}"
export TRITON_CACHE_DIR="\${TRITON_CACHE_DIR:-\$FLAGGEMS_PREFIX/cache/triton}"
export TRITON_DUMP_DIR="\${_flaggems_user_triton_dump_dir:-\$FLAGGEMS_PREFIX/triton-stage-dumps}"
export FLAGGEMS_IR_DUMP_DIR="\${FLAGGEMS_IR_DUMP_DIR:-\$FLAGGEMS_PREFIX/mlir-dumps}"
export MLIR_ENABLE_DUMP="\${_flaggems_user_mlir_enable_dump:-1}"
export MLIR_DUMP_PATH="\${_flaggems_user_mlir_dump_path:-\$FLAGGEMS_IR_DUMP_DIR/flaggems-mlir-dump.mlir}"
export TRITON_KERNEL_DUMP="\${_flaggems_user_triton_kernel_dump:-1}"
export TRITON_ALWAYS_COMPILE="\${_flaggems_user_triton_always_compile:-1}"

if [[ -d "\$MLIR_DUMP_PATH" ]]; then
  export MLIR_DUMP_PATH="\$MLIR_DUMP_PATH/flaggems-mlir-dump.mlir"
fi

mkdir -p -- "\$PIP_CACHE_DIR" "\$TRITON_CACHE_DIR" "\$TRITON_DUMP_DIR" "\$(dirname -- "\$MLIR_DUMP_PATH")"

unset _flaggems_user_triton_dump_dir _flaggems_user_mlir_enable_dump
unset _flaggems_user_mlir_dump_path _flaggems_user_triton_kernel_dump
unset _flaggems_user_triton_always_compile
EOF
  chmod 0644 "$ENV_FILE"
}

check_python_stack() {
  note '检查 PyTorch/Triton/FlagTree/FlagGems'
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata
import importlib.util

import torch
import triton
import flag_gems

print("flag_gems:", getattr(flag_gems, "__version__", "n/a"), flag_gems.__file__)
print("torch:", torch.__version__, "torch cuda:", torch.version.cuda, "cuda available:", torch.cuda.is_available())
print("triton:", triton.__version__, triton.__file__)

spec = importlib.util.find_spec("flag_tree") or importlib.util.find_spec("flagtree")
print("FlagTree import:", spec.origin if spec else "not provided as a top-level module")
try:
    print("FlagTree distribution:", metadata.version("flagtree"))
except metadata.PackageNotFoundError:
    print("FlagTree distribution: not installed")

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
PY
  "$PYTHON_BIN" -m pip check
}

run_smoke_test() {
  [[ "$RUN_TEST" -eq 1 ]] || return 0

  note '运行 FlagGems CUDA smoke test，并打印 IR dump 路径'
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import torch
import flag_gems

torch.manual_seed(0)
x = torch.randn((128, 128), device="cuda", dtype=torch.float32)
y = torch.randn((128, 128), device="cuda", dtype=torch.float32)

with flag_gems.use_gems():
    z = torch.add(x, y)

torch.cuda.synchronize()
max_err = (z - (x + y)).abs().max().item()
print("device:", torch.cuda.get_device_name(0))
print("result shape:", tuple(z.shape))
print("max error:", max_err)
print("MLIR_ENABLE_DUMP:", os.environ.get("MLIR_ENABLE_DUMP"))
print("MLIR_DUMP_PATH:", os.environ.get("MLIR_DUMP_PATH"))
print("TRITON_DUMP_DIR:", os.environ.get("TRITON_DUMP_DIR"))
print("TRITON_CACHE_DIR:", os.environ.get("TRITON_CACHE_DIR"))
dump_path = os.environ.get("MLIR_DUMP_PATH")
if dump_path:
    print("MLIR dump exists:", Path(dump_path).exists())
if max_err != 0:
    raise SystemExit(f"unexpected max error: {max_err}")
PY
}

PREFIX=$DEFAULT_PREFIX
SOURCE_DIR=$DEFAULT_SOURCE_DIR
FLAGTREE_PREFIX=$DEFAULT_FLAGTREE_PREFIX

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
    --flagtree-prefix)
      [[ $# -ge 2 ]] || die '--flagtree-prefix 缺少目录参数。'
      FLAGTREE_PREFIX=$2
      shift 2
      ;;
    --skip-test)
      RUN_TEST=0
      shift
      ;;
    --force-reclone)
      FORCE_RECLONE=1
      shift
      ;;
    --with-cpp)
      WITH_CPP=1
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

[[ -n "$PREFIX" ]] || die '--prefix 不能为空。'
[[ -n "$SOURCE_DIR" ]] || die '--source-dir 不能为空。'
[[ -n "$FLAGTREE_PREFIX" ]] || die '--flagtree-prefix 不能为空。'
[[ "$PREFIX" != *$'\n'* && "$PREFIX" != *$'\r'* ]] || die '--prefix 不能包含换行符。'
[[ "$SOURCE_DIR" != *$'\n'* && "$SOURCE_DIR" != *$'\r'* ]] || die '--source-dir 不能包含换行符。'
[[ "$FLAGTREE_PREFIX" != *$'\n'* && "$FLAGTREE_PREFIX" != *$'\r'* ]] || die '--flagtree-prefix 不能包含换行符。'

PREFIX=$(canonicalize_existing_or_parent "$PREFIX")
SOURCE_DIR=$(canonicalize_existing_or_parent "$SOURCE_DIR")
FLAGTREE_PREFIX=$(canonicalize_existing_or_parent "$FLAGTREE_PREFIX")
[[ "$PREFIX" != / ]] || die '不能把 / 作为 --prefix。'
[[ "$SOURCE_DIR" != / ]] || die '不能把 / 作为 --source-dir。'
[[ "$FLAGTREE_PREFIX" != / ]] || die '不能把 / 作为 --flagtree-prefix。'

ENV_FILE="$PREFIX/env-flaggems.sh"

check_platform
activate_flagtree_environment
checkout_flaggems
install_build_requirements
install_flaggems_python
install_flaggems_cpp
write_environment_file
check_python_stack

note 'FlagGems 已安装。'
printf '环境脚本：source %s\n' "$ENV_FILE"
printf '源码目录：%s\n' "$SOURCE_DIR"
printf '安装目录：%s\n' "$PREFIX"
printf 'MLIR dump 文件：%s\n' "$MLIR_DUMP_PATH"
printf 'Triton dump 目录：%s\n' "$TRITON_DUMP_DIR"

run_smoke_test
