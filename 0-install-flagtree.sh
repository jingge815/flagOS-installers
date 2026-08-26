#!/usr/bin/env bash
# Build and install FlagTree Triton 3.5 into a user-owned prefix.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEFAULT_SOURCE_DIR="$SCRIPT_DIR/FlagTree"
MAX_JOBS=${MAX_JOBS:-8}
RUN_TEST=1

#FLAGTREE_REPOSITORY=https://github.com/KernelLLM/FlagTree.git
#FLAGTREE_BRANCH=common-ir-triton35
#FLAGTREE_REVISION=317f15a426466633c4f37f164b2c58ae9c31bd03

FLAGTREE_REPOSITORY=https://github.com/jingge815/FlagTree.git
FLAGTREE_BRANCH=develop
#FLAGTREE_REVISION=317f15a426466633c4f37f164b2c58ae9c31bd03

FLIR_REPOSITORY=https://github.com/kateyijian/flir.git
FLIR_REVISION=165f387b28e3fdbd03542e7ae9881db902facd16

LLVM_ARCHIVE=llvm-7d5de303-ubuntu-x64.tar.gz
LLVM_URL=https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-7d5de303-ubuntu-x64.tar.gz
TRITON_DEPS_ARCHIVE=build-deps-triton_3.5.x-linux-x64.tar.gz
TRITON_DEPS_URL=https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.5.x-linux-x64.tar.gz
PYTHON_ARCHIVE=cpython-3.10.20+20260718-x86_64-unknown-linux-gnu-install_only.tar.gz
PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260718/cpython-3.10.20%2B20260718-x86_64-unknown-linux-gnu-install_only.tar.gz

usage() {
  cat <<EOF
用法：
  bash 0-install-flagtree.sh [选项]

选项：
  --prefix DIR          安装目录，默认：../flagOS-installed/flagTree
  --source-dir DIR      FlagTree 源码目录，默认：./FlagTree
  --max-jobs N          编译并行度，默认：$MAX_JOBS
  --skip-test           跳过安装后的验证
  --pytorch-prefix DIR  PyTorch 安装目录，默认：../flagOS-installed/pytorch
  --skip-pytorch-sync   跳过"把带 PIM 能力的 triton 同步进 PyTorch 环境"这步
  -h, --help            显示帮助

说明：
  本脚本不需要 root，不安装 NVIDIA 驱动；目标机器必须已经能运行 nvidia-smi。

  每次构建成功后，默认会把这次编译出的 triton（带 PIM mlir -> C 相关 pass）
  同步覆盖到 PyTorch 环境自带的那份 triton——原因和风险见
  sync_triton_to_pytorch 函数的注释；不需要这个行为时用 --skip-pytorch-sync。
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
  command -v "$1" >/dev/null 2>&1 || die "缺少命令 $1。请先安装：$2"
}

download_tarball() {
  local url=$1
  local destination=$2
  local temporary="${destination}.part.$$"

  if [[ -s "$destination" ]] && tar -tzf "$destination" >/dev/null 2>&1; then
    return
  fi

  rm -f -- "$destination" "$temporary"
  note "下载 $(basename -- "$destination")"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "$temporary" "$url"
  else
    wget --output-document="$temporary" "$url"
  fi
  tar -tzf "$temporary" >/dev/null
  mv -- "$temporary" "$destination"
}

check_platform() {
  [[ -r /etc/os-release ]] || die '无法读取 /etc/os-release。'
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 22.04 ]] || \
    die "需要 Ubuntu 22.04，当前为 ${PRETTY_NAME:-未知系统}。"
  [[ $(uname -m) == x86_64 ]] || die "需要 x86_64，当前为 $(uname -m)。"

  require_command nvidia-smi 'NVIDIA 驱动'
  require_command git 'git'
  require_command tar 'tar'
  require_command gzip 'gzip'
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

install_python() {
  if [[ ! -x "$PYTHON/bin/python" ]]; then
    download_tarball "$PYTHON_URL" "$DOWNLOADS/$PYTHON_ARCHIVE"
    rm -rf -- "$PYTHON"
    mkdir -p "$PYTHON"
    tar -xzf "$DOWNLOADS/$PYTHON_ARCHIVE" -C "$PYTHON" --strip-components=1
  fi

  "$PYTHON/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$PYTHON/bin/python" -m pip install --upgrade --no-cache-dir \
    'pip<27' setuptools==83.0.0 wheel==0.47.0 \
    cmake==3.31.10 ninja==1.13.0 pybind11==3.0.4 lit==18.1.8 \
    numpy==1.26.4 pytest==8.3.5
  "$PYTHON/bin/python" -m pip install --upgrade --no-cache-dir \
    torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

  if [[ -e "$PYTHON_LINK" && ! -L "$PYTHON_LINK" ]]; then
    die "Python 链接路径已被普通目录或文件占用：$PYTHON_LINK"
  fi
  ln -sfn "$(basename -- "$PYTHON")" "$PYTHON_LINK"
}

install_llvm() {
  if [[ ! -x "$LLVM/bin/llvm-config" ]]; then
    download_tarball "$LLVM_URL" "$DOWNLOADS/$LLVM_ARCHIVE"
    rm -rf -- "$LLVM"
    mkdir -p "$LLVM"
    tar -xzf "$DOWNLOADS/$LLVM_ARCHIVE" -C "$LLVM" --strip-components=1
  fi

  [[ $("$LLVM/bin/llvm-config" --version) == 22.0.0git ]] || \
    die "LLVM 版本不正确：$("$LLVM/bin/llvm-config" --version)"
}

install_triton_build_dependencies() {
  if [[ ! -x "$TRITON_HOME/nvidia/nvcc/cuda_nvcc-linux-x86_64-12.8.93-archive/bin/ptxas" || \
        ! -x "$TRITON_HOME/nvidia/cuobjdump/cuda_cuobjdump-linux-x86_64-12.8.55-archive/bin/cuobjdump" || \
        ! -x "$TRITON_HOME/nvidia/nvdisasm/cuda_nvdisasm-linux-x86_64-12.8.55-archive/bin/nvdisasm" || \
        ! -f "$TRITON_HOME/nvidia/nvcc/cuda_nvcc-linux-x86_64-12.8.93-archive/nvvm/libdevice/libdevice.10.bc" || \
        ! -f "$TRITON_HOME/nvidia/cudart/cuda_cudart-linux-x86_64-12.8.57-archive/include/cuda_runtime.h" ]]; then
    download_tarball "$TRITON_DEPS_URL" "$DOWNLOADS/$TRITON_DEPS_ARCHIVE"
    rm -rf -- "$TRITON_HOME"
    mkdir -p "$TRITON_HOME"
    tar -xzf "$DOWNLOADS/$TRITON_DEPS_ARCHIVE" -C "$TRITON_HOME" --strip-components=1
  fi

  mkdir -p "$NVIDIA_TOOLCHAIN/bin" "$NVIDIA_TOOLCHAIN/lib" \
    "$NVIDIA_TOOLCHAIN/include" "$NVIDIA_TOOLCHAIN/nvvm/libdevice"
  ln -sfn "$TRITON_HOME/nvidia/nvcc/cuda_nvcc-linux-x86_64-12.8.93-archive/bin/ptxas" \
    "$NVIDIA_TOOLCHAIN/bin/ptxas"
  ln -sfn "$TRITON_HOME/nvidia/cuobjdump/cuda_cuobjdump-linux-x86_64-12.8.55-archive/bin/cuobjdump" \
    "$NVIDIA_TOOLCHAIN/bin/cuobjdump"
  ln -sfn "$TRITON_HOME/nvidia/nvdisasm/cuda_nvdisasm-linux-x86_64-12.8.55-archive/bin/nvdisasm" \
    "$NVIDIA_TOOLCHAIN/bin/nvdisasm"
  ln -sfn "$TRITON_HOME/nvidia/cudart/cuda_cudart-linux-x86_64-12.8.57-archive/include" \
    "$NVIDIA_TOOLCHAIN/include/cudart"
  ln -sfn "$TRITON_HOME/nvidia/cudart/cuda_cudart-linux-x86_64-12.8.57-archive/lib" \
    "$NVIDIA_TOOLCHAIN/lib/cudart"
  ln -sfn "$TRITON_HOME/nvidia/cupti/cuda_cupti-linux-x86_64-12.8.90-archive/include" \
    "$NVIDIA_TOOLCHAIN/include/cupti"
  ln -sfn "$TRITON_HOME/nvidia/cupti/cuda_cupti-linux-x86_64-12.8.90-archive/lib" \
    "$NVIDIA_TOOLCHAIN/lib/cupti"
  ln -sfn "$TRITON_HOME/nvidia/nvcc/cuda_nvcc-linux-x86_64-12.8.93-archive/nvvm/libdevice/libdevice.10.bc" \
    "$NVIDIA_TOOLCHAIN/nvvm/libdevice/libdevice.10.bc"
}

download_deb() {
  local package=$1
  local work_dir
  work_dir=$(mktemp -d "$DOWNLOADS/${package}.XXXXXX")
  (
    cd "$work_dir"
    apt-get download "$package"
  )
  find "$work_dir" -maxdepth 1 -name '*.deb' -type f -exec mv -f {} "$DEBS/" \;
  rmdir "$work_dir"
}

install_development_headers() {
  local package archive
  mkdir -p "$SYSROOT" "$DEBS"
  if [[ ! -f "$SYSROOT/usr/include/zlib.h" || \
        ! -f "$SYSROOT/usr/include/libxml2/libxml/parser.h" || \
        ! -e "$SYSROOT/usr/lib/x86_64-linux-gnu/libz.so.1" || \
        ! -e "$SYSROOT/usr/lib/x86_64-linux-gnu/libxml2.so.2" ]]; then
    for package in zlib1g zlib1g-dev libxml2 libxml2-dev; do
      archive=$(find "$DEBS" -maxdepth 1 -name "${package}_*.deb" -type f -print -quit)
      if [[ -z "$archive" ]]; then
        download_deb "$package"
        archive=$(find "$DEBS" -maxdepth 1 -name "${package}_*.deb" -type f -print -quit)
      fi
      [[ -n "$archive" ]] || die "未能下载 $package 的 deb 包。"
      dpkg-deb -x "$archive" "$SYSROOT"
    done
  fi

  [[ -e "$SYSROOT/usr/lib/x86_64-linux-gnu/libz.so" ]] || \
    die '解压 zlib1g-dev 后未找到 libz.so。请检查 Ubuntu 软件源。'
  [[ -e "$SYSROOT/usr/lib/x86_64-linux-gnu/libxml2.so" ]] || \
    die '解压 libxml2-dev 后未找到 libxml2.so。请检查 Ubuntu 软件源。'

  ln -sfn 'libz.so.1' "$SYSROOT/usr/lib/x86_64-linux-gnu/libz.so"
  ln -sfn 'libxml2.so.2' "$SYSROOT/usr/lib/x86_64-linux-gnu/libxml2.so"
}

write_environment_file() {
  local prefix_shell source_shell
  printf -v prefix_shell '%q' "$PREFIX"
  printf -v source_shell '%q' "$SOURCE_DIR"
  cat > "$ENV_FILE" <<EOF
#!/usr/bin/env bash
# Source this file before building or using this FlagTree installation.

if [[ "\${BASH_SOURCE[0]}" == "\$0" ]]; then
  echo "Source this file instead: source \${BASH_SOURCE[0]}" >&2
  exit 1
fi

FLAGTREE_PREFIX=$prefix_shell
FLAGTREE_SOURCE=$source_shell
FLAGTREE_LLVM="\$FLAGTREE_PREFIX/llvm-7d5de303"
FLAGTREE_SYSROOT="\$FLAGTREE_PREFIX/sysroot/usr"
FLAGTREE_NVIDIA_TOOLBIN="\$FLAGTREE_PREFIX/nvidia-toolchain-12.8/bin"
FLAGTREE_MLIR_DUMP_DIR="\$FLAGTREE_PREFIX/mlir-dumps"
FLAGTREE_TRITON_DUMP_DIR="\$FLAGTREE_PREFIX/triton-stage-dumps"

export PATH="\$FLAGTREE_PREFIX/python/bin:\$FLAGTREE_LLVM/bin:\$FLAGTREE_NVIDIA_TOOLBIN:\$PATH"
export LD_LIBRARY_PATH="\$FLAGTREE_LLVM/lib:\$FLAGTREE_SYSROOT/lib/x86_64-linux-gnu\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export LIBRARY_PATH="\$FLAGTREE_SYSROOT/lib/x86_64-linux-gnu\${LIBRARY_PATH:+:\$LIBRARY_PATH}"
export C_INCLUDE_PATH="\$FLAGTREE_PREFIX/nvidia-toolchain-12.8/include/cudart\${C_INCLUDE_PATH:+:\$C_INCLUDE_PATH}"

export LLVM_SYSPATH="\$FLAGTREE_LLVM"
export LLVM_INCLUDE_DIRS="\$FLAGTREE_LLVM/include"
export LLVM_LIBRARY_DIR="\$FLAGTREE_LLVM/lib"
export MLIR_DIR="\$FLAGTREE_LLVM/lib/cmake/mlir"
export LLVM_DIR="\$FLAGTREE_LLVM/lib/cmake/llvm"
export LLD_DIR="\$FLAGTREE_LLVM/lib/cmake/lld"

export TRITON_HOME="\$FLAGTREE_PREFIX/triton-home"
export TRITON_CACHE_DIR="\$FLAGTREE_PREFIX/cache"
export TRITON_DUMP_DIR="\${TRITON_DUMP_DIR:-\$FLAGTREE_TRITON_DUMP_DIR}"
export TRITON_BUILD_DIR="\$FLAGTREE_PREFIX/build/flagtree-cmake"
export TRITON_PTXAS_PATH="\$FLAGTREE_PREFIX/nvidia-toolchain-12.8/bin/ptxas"
export TRITON_CUOBJDUMP_PATH="\$FLAGTREE_PREFIX/nvidia-toolchain-12.8/bin/cuobjdump"
export TRITON_NVDISASM_PATH="\$FLAGTREE_PREFIX/nvidia-toolchain-12.8/bin/nvdisasm"
export NVDISASM_PATH="\$FLAGTREE_PREFIX/nvidia-toolchain-12.8/bin"
export TRITON_CUDACRT_PATH="\$FLAGTREE_PREFIX/nvidia-toolchain-12.8/include/cudart"
export TRITON_CUDART_PATH="\$FLAGTREE_PREFIX/nvidia-toolchain-12.8/include/cudart"
export TRITON_CUPTI_INCLUDE_PATH="\$FLAGTREE_PREFIX/nvidia-toolchain-12.8/include/cupti"
export TRITON_CUPTI_LIB_PATH="\$FLAGTREE_PREFIX/nvidia-toolchain-12.8/lib/cupti"
export TRITON_LIBDEVICE_PATH="\$FLAGTREE_PREFIX/nvidia-toolchain-12.8/nvvm/libdevice/libdevice.10.bc"

export FLAGTREE_IR_DUMP_DIR="\$FLAGTREE_MLIR_DUMP_DIR"
export MLIR_ENABLE_DUMP="\${MLIR_ENABLE_DUMP:-1}"
export MLIR_DUMP_PATH="\${MLIR_DUMP_PATH:-\$FLAGTREE_MLIR_DUMP_DIR/flagtree-mlir-dump.mlir}"
export TRITON_KERNEL_DUMP="\${TRITON_KERNEL_DUMP:-1}"
export TRITON_ALWAYS_COMPILE="\${TRITON_ALWAYS_COMPILE:-1}"
if [[ -d "\$MLIR_DUMP_PATH" ]]; then
  export MLIR_DUMP_PATH="\$MLIR_DUMP_PATH/flagtree-mlir-dump.mlir"
fi
mkdir -p -- "\$(dirname -- "\$MLIR_DUMP_PATH")"
mkdir -p -- "\$TRITON_DUMP_DIR"

unset FLAGTREE_BACKEND FLAGTREE_PLUGIN TRITON_OFFLINE_BUILD
EOF
  chmod 0644 "$ENV_FILE"
}

checkout_flagtree() {
  if [[ ! -e "$SOURCE_DIR" ]]; then
    git clone --branch "$FLAGTREE_BRANCH" --single-branch "$FLAGTREE_REPOSITORY" "$SOURCE_DIR"
  elif [[ ! -d "$SOURCE_DIR/.git" ]]; then
    die "源码目录已存在但不是 Git 仓库：$SOURCE_DIR"
  fi

  # ALLOW_DIRTY_FLAGTREE_SOURCE=1 时跳过这条检查：用于本地开发时源码目录带
  # 未提交改动（比如正在开发的新 pass）也要能走一遍完整安装流程的场景。
  # 默认还是拒绝，避免脚本后续步骤在带着未提交改动的源码上做 checkout 之类
  # 操作时出现意外覆盖。
  if [[ "${ALLOW_DIRTY_FLAGTREE_SOURCE:-0}" != 1 ]]; then
    git -C "$SOURCE_DIR" diff --quiet || die "源码目录存在已跟踪的修改：$SOURCE_DIR（本地开发可设 ALLOW_DIRTY_FLAGTREE_SOURCE=1 跳过此检查）"
    git -C "$SOURCE_DIR" diff --cached --quiet || die "源码目录存在暂存修改：$SOURCE_DIR（本地开发可设 ALLOW_DIRTY_FLAGTREE_SOURCE=1 跳过此检查）"
  fi
}

checkout_flir() {
  local flir_dir="$SOURCE_DIR/third_party/flir"

  if [[ ! -e "$flir_dir" ]]; then
    git clone "$FLIR_REPOSITORY" "$flir_dir"
  elif [[ ! -d "$flir_dir/.git" ]]; then
    die "FLIR 目录已存在但不是 Git 仓库：$flir_dir"
  fi

  git -C "$flir_dir" diff --quiet || die "FLIR 目录存在已跟踪的修改：$flir_dir"
  git -C "$flir_dir" diff --cached --quiet || die "FLIR 目录存在暂存修改：$flir_dir"
  git -C "$flir_dir" fetch --depth 1 origin "$FLIR_REVISION" >/dev/null 2>&1 || true
  git -C "$flir_dir" checkout --detach "$FLIR_REVISION"
  [[ $(git -C "$flir_dir" rev-parse HEAD) == "$FLIR_REVISION" ]] || \
    die 'FLIR 源码提交与预期不一致。'
}

build_flagtree() {
  local wheel
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  export TRITON_BUILD_PROTON=OFF
  export TRITON_BUILD_UT=OFF
  export TRITON_BUILD_WITH_CCACHE=OFF
  export TRITON_PARALLEL_LINK_JOBS=${TRITON_PARALLEL_LINK_JOBS:-1}
  export MAX_JOBS
  export TRITON_APPEND_CMAKE_ARGS="-DLLVM_ENABLE_WERROR=OFF -DTRITON_BUILD_UT=OFF -DZLIB_ROOT=$SYSROOT/usr -DZLIB_LIBRARY=$SYSROOT/usr/lib/x86_64-linux-gnu/libz.so -DZLIB_INCLUDE_DIR=$SYSROOT/usr/include -DLIBXML2_LIBRARY=$SYSROOT/usr/lib/x86_64-linux-gnu/libxml2.so -DLIBXML2_INCLUDE_DIR=$SYSROOT/usr/include/libxml2"

  mkdir -p "$WHEELS"
  rm -rf -- "$BUILD"
  (
    cd "$SOURCE_DIR"
    "$PYTHON/bin/python" -m pip wheel . --no-build-isolation --no-deps --wheel-dir "$WHEELS"
  )
  wheel=$(find "$WHEELS" -maxdepth 1 -name 'flagtree-*.whl' -type f -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  [[ -n "$wheel" ]] || die '没有生成 FlagTree wheel。'
  "$PYTHON/bin/python" -m pip install --force-reinstall --no-deps "$wheel"
  rm -rf -- "$SOURCE_DIR/python/flagtree.egg-info"
}

install_example() {
  local template="$SCRIPT_DIR/matmul_sm80.py"
  [[ -f "$template" ]] || die "找不到矩阵乘法示例模板：$template"
  mkdir -p "$EXAMPLES_DIR"
  cp -- "$template" "$EXAMPLES_DIR/matmul_sm80.py"
}

run_validation() {
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  mkdir -p "$MLIR_DUMP_DIR"
  "$PYTHON/bin/python" -c 'import torch, triton; print("imports ok")'
  "$PYTHON/bin/python" "$EXAMPLES_DIR/matmul_sm80.py"
}

sync_triton_to_pytorch() {
  # PyTorch 环境（2-install-pytorch.sh 装的那份）自带一份 triton，但那是走
  # pip 装的普通版，没有本仓库 FlagTree 源码里的 PIM 相关能力（pim mlir、
  # LowerPIMSingleTasklet 等 pass）。这两份 triton 同一个 Python 版本
  # （3.10.20）、同一个 LLVM commit（7d5de303）构建，ABI 兼容，可以直接拿
  # 这次刚编译出来的文件去覆盖 PyTorch 那份——不这样做的后果：进程里
  # `import torch`/`transformers` 会先把 PyTorch 自带的普通版 triton 装进
  # `sys.modules`，之后任何代码都无法再切换到带 PIM 能力的版本（Python 的
  # `import` 只认进程内第一次加载的那份），这在 flagos-pim-compiler 的真实
  # 端到端场景里复现过。
  #
  # 用 --skip-pytorch-sync 跳过；PyTorch 安装目录不是默认路径时用
  # --pytorch-prefix 指定。
  local pytorch_triton="$PYTORCH_PREFIX/python-3.10.20/lib/python3.10/site-packages/triton"
  local our_triton="$PYTHON/lib/python3.10/site-packages/triton"

  if [[ ! -d "$pytorch_triton" ]]; then
    note "跳过同步：未找到 PyTorch 环境的 triton（$pytorch_triton 不存在）"
    return
  fi
  if [[ ! -d "$our_triton" ]]; then
    die "本次构建产物里找不到 triton 包：$our_triton"
  fi

  note "把带 PIM 能力的 triton 同步进 PyTorch 环境：$pytorch_triton"
  mkdir -p "$PYTORCH_PREFIX/.triton-backup-pre-pim"
  cp -f "$pytorch_triton/_C/libtriton.so" \
    "$PYTORCH_PREFIX/.triton-backup-pre-pim/libtriton.so.orig.$(date +%s)" 2>/dev/null || true
  cp -f "$pytorch_triton/backends/nvidia/compiler.py" \
    "$PYTORCH_PREFIX/.triton-backup-pre-pim/compiler.py.orig.$(date +%s)" 2>/dev/null || true

  cp -f "$our_triton/_C/libtriton.so" "$pytorch_triton/_C/libtriton.so"
  cp -f "$our_triton/backends/pim_sidecar.py" "$pytorch_triton/backends/pim_sidecar.py"
  cp -f "$our_triton/backends/nvidia/compiler.py" "$pytorch_triton/backends/nvidia/compiler.py"
  rm -rf "$pytorch_triton/backends/nvidia/bin" "$pytorch_triton/backends/nvidia/include" \
    "$pytorch_triton/backends/nvidia/lib/cupti"
  cp -r "$our_triton/backends/nvidia/bin" "$pytorch_triton/backends/nvidia/bin"
  cp -r "$our_triton/backends/nvidia/include" "$pytorch_triton/backends/nvidia/include"
  cp -r "$our_triton/backends/nvidia/lib/cupti" "$pytorch_triton/backends/nvidia/lib/cupti"
  rm -rf "$pytorch_triton/backends/__pycache__" "$pytorch_triton/backends/nvidia/__pycache__"

  note "同步完成。验证：source $PYTORCH_PREFIX/env-pytorch.sh && python3 -c 'from triton._C.libtriton import passes; print(hasattr(passes, \"pim\"))'"
}

PREFIX=
SOURCE_DIR=$DEFAULT_SOURCE_DIR
PYTORCH_PREFIX=
SKIP_PYTORCH_SYNC=0

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
    --pytorch-prefix)
      [[ $# -ge 2 ]] || die '--pytorch-prefix 缺少目录参数。'
      PYTORCH_PREFIX=$2
      shift 2
      ;;
    --skip-pytorch-sync)
      SKIP_PYTORCH_SYNC=1
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
[[ -n "$PREFIX" ]] || die '--prefix 不能为空。'
[[ "$PREFIX" != *$'\n'* && "$PREFIX" != *$'\r'* ]] || die '--prefix 不能包含换行符。'
mkdir -p -- "$PREFIX"
PREFIX=$(cd -- "$PREFIX" && pwd -P)
[[ "$PREFIX" != / ]] || die '不能把 / 作为 --prefix。'

if [[ -z "${PYTORCH_PREFIX:-}" ]]; then
  PYTORCH_PREFIX="$SCRIPT_DIR/../flagOS-installed/pytorch"
fi
if [[ -e "$PYTORCH_PREFIX" ]]; then
  PYTORCH_PREFIX=$(cd -- "$PYTORCH_PREFIX" && pwd -P)
fi

mkdir -p -- "$(dirname -- "$SOURCE_DIR")"
if [[ -e "$SOURCE_DIR" ]]; then
  SOURCE_DIR=$(cd -- "$SOURCE_DIR" && pwd -P)
else
  SOURCE_PARENT=$(cd -- "$(dirname -- "$SOURCE_DIR")" && pwd -P)
  SOURCE_DIR="$SOURCE_PARENT/$(basename -- "$SOURCE_DIR")"
fi

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

mkdir -p "$PREFIX" "$DOWNLOADS" "$DEBS" "$PREFIX/build" "$PREFIX/cache" "$MLIR_DUMP_DIR"

check_platform
install_python
install_llvm
install_triton_build_dependencies
install_development_headers
write_environment_file
checkout_flagtree
checkout_flir
build_flagtree
install_example
if [[ "$SKIP_PYTORCH_SYNC" -eq 0 ]]; then
  sync_triton_to_pytorch
fi

note 'FlagTree 已安装。'
printf '环境脚本：source %s\n' "$ENV_FILE"
printf '源码目录：%s\n' "$SOURCE_DIR"
printf 'wheel 目录：%s\n' "$WHEELS"
printf 'MLIR dump 目录：%s\n' "$MLIR_DUMP_DIR"

if [[ $RUN_TEST -eq 1 ]]; then
  run_validation
fi
