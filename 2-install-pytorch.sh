#!/usr/bin/env bash
# Build and install pinned PyTorch from source without root access.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEFAULT_SOURCE_DIR="$SCRIPT_DIR/pytorch"
DEFAULT_PREFIX="$SCRIPT_DIR/../flagOS-installed/pytorch"

PYTORCH_REPOSITORY=https://github.com/pytorch/pytorch.git
PYTORCH_TAG=v2.9.1
PYTORCH_REVISION=d38164a545b4a4e4e0cf73ce67173f70574890b6
PYTORCH_VERSION=2.9.1

CUDA_VERSION=12.8.1
CUDA_RUNFILE=cuda_12.8.1_570.124.06_linux.run
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/$CUDA_RUNFILE"
PYTHON_ARCHIVE=cpython-3.10.20+20260718-x86_64-unknown-linux-gnu-install_only.tar.gz
PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260718/cpython-3.10.20%2B20260718-x86_64-unknown-linux-gnu-install_only.tar.gz

MAX_JOBS=${MAX_JOBS:-16}
RUN_TEST=1
SKIP_BUILD=0
CLEAN_BUILD=0
FORCE_RECLONE=0

usage() {
  cat <<EOF
用法：
  bash 2-install-pytorch.sh [选项]

选项：
  --prefix DIR        安装目录，默认：../flagOS-installed/pytorch
  --source-dir DIR    PyTorch 源码目录，默认：./pytorch
  --max-jobs N        编译并行度，默认：$MAX_JOBS
  --skip-build        只准备源码、Python、CUDA 和依赖，不编译 wheel
  --skip-test         跳过安装后的 CUDA smoke test
  --clean-build       编译前删除源码目录中的 build/
  --force-reclone     删除已有 PyTorch 源码目录后重新 clone
  -h, --help          显示帮助

固定版本：
  仓库：$PYTORCH_REPOSITORY
  提交：$PYTORCH_REVISION ($PYTORCH_TAG)
  CUDA：$CUDA_VERSION
  Python：3.10.20

说明：
  本脚本不需要 root，不安装 NVIDIA 驱动。目标机器需要 Ubuntu 22.04 x86_64、
  可用的 nvidia-smi、570+ NVIDIA 驱动、基础编译工具和公网访问。
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

download_file() {
  local url=$1
  local destination=$2
  local temporary="${destination}.part.$$"

  [[ -s "$destination" ]] && return
  rm -f -- "$temporary"
  note "下载 $(basename -- "$destination")"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --retry-delay 2 --output "$temporary" "$url"
  else
    wget --tries=3 --output-document="$temporary" "$url"
  fi
  [[ -s "$temporary" ]] || die "下载结果为空：$url"
  mv -- "$temporary" "$destination"
}

check_platform() {
  local driver_major gcc_version

  [[ -r /etc/os-release ]] || die '无法读取 /etc/os-release。'
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 22.04 ]] || \
    die "需要 Ubuntu 22.04，当前为 ${PRETTY_NAME:-未知系统}。"
  [[ $(uname -m) == x86_64 ]] || die "需要 x86_64，当前为 $(uname -m)。"

  require_command git
  require_command tar
  require_command gzip
  require_command make
  require_command cc
  require_command c++
  require_command ar
  require_command ld
  require_command awk
  require_command sed
  require_command find
  require_command nvidia-smi
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    die '缺少 curl 或 wget。'
  fi

  nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi 不可用，请先确认 NVIDIA 驱动和 GPU。'
  driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk -F. 'NR == 1 { print $1 }')
  [[ "$driver_major" =~ ^[0-9]+$ && "$driver_major" -ge 570 ]] || \
    die "CUDA 12.8 需要 570+ NVIDIA 驱动，当前主版本为 ${driver_major:-未知}。"

  gcc_version=$(cc -dumpfullversion -dumpversion)
  [[ "$gcc_version" =~ ^([0-9]+) ]] && (( BASH_REMATCH[1] >= 9 )) || \
    die "PyTorch $PYTORCH_VERSION 需要 GCC 9.4+，当前为 $gcc_version。"
}

install_python() {
  if [[ ! -x "$PYTHON/bin/python" ]]; then
    download_file "$PYTHON_URL" "$DOWNLOADS/$PYTHON_ARCHIVE"
    tar -tzf "$DOWNLOADS/$PYTHON_ARCHIVE" >/dev/null
    rm -rf -- "$PYTHON"
    mkdir -p "$PYTHON"
    tar -xzf "$DOWNLOADS/$PYTHON_ARCHIVE" -C "$PYTHON" --strip-components=1
  fi

  "$PYTHON/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$PYTHON/bin/python" -m pip install --upgrade --cache-dir "$PIP_CACHE_DIR" \
    'pip<27' 'setuptools>=70.1.0,<80.0' wheel \
    'cmake==3.31.10' ninja numpy packaging pyyaml requests six \
    'typing-extensions>=4.13.2' filelock fsspec 'sympy>=1.13.3' \
    'networkx>=2.5.1' jinja2 psutil optree cffi mkl-static mkl-include

  if [[ -e "$PYTHON_LINK" && ! -L "$PYTHON_LINK" ]]; then
    die "Python 链接路径已被普通目录或文件占用：$PYTHON_LINK"
  fi
  ln -sfn "$(basename -- "$PYTHON")" "$PYTHON_LINK"
}

install_cuda_toolkit() {
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    download_file "$CUDA_URL" "$DOWNLOADS/$CUDA_RUNFILE"
    chmod u+x "$DOWNLOADS/$CUDA_RUNFILE"
    note "安装用户态 CUDA Toolkit 到：$CUDA_HOME"
    "$DOWNLOADS/$CUDA_RUNFILE" --silent --toolkit --toolkitpath="$CUDA_HOME" --no-opengl-libs
  fi

  [[ -x "$CUDA_HOME/bin/nvcc" ]] || die "未找到 nvcc：$CUDA_HOME/bin/nvcc"
  "$CUDA_HOME/bin/nvcc" --version | grep -F 'release 12.8' >/dev/null || \
    die "CUDA Toolkit 版本不是 12.8：$("$CUDA_HOME/bin/nvcc" --version | tail -n 1)"
}

install_cuda_python_libraries() {
  "$PYTHON/bin/python" -m pip install --upgrade --cache-dir "$PIP_CACHE_DIR" \
    'nvidia-cuda-nvrtc-cu12==12.8.93' \
    'nvidia-cuda-runtime-cu12==12.8.90' \
    'nvidia-cuda-cupti-cu12==12.8.90' \
    'nvidia-cudnn-cu12==9.10.2.21' \
    'nvidia-cublas-cu12==12.8.4.1' \
    'nvidia-cufft-cu12==11.3.3.83' \
    'nvidia-curand-cu12==10.3.9.90' \
    'nvidia-cusolver-cu12==11.7.3.90' \
    'nvidia-cusparse-cu12==12.5.8.93' \
    'nvidia-cusparselt-cu12==0.7.1' \
    'nvidia-nccl-cu12==2.27.5' \
    'nvidia-nvjitlink-cu12==12.8.93' \
    'nvidia-nvtx-cu12==12.8.90'

  local nvidia_dir="$PYTHON/lib/python3.10/site-packages/nvidia"
  ln -sfn libcudnn.so.9 "$nvidia_dir/cudnn/lib/libcudnn.so"
  ln -sfn libnccl.so.2 "$nvidia_dir/nccl/lib/libnccl.so"
}

write_environment_file() {
  local prefix_shell source_shell
  printf -v prefix_shell '%q' "$PREFIX"
  printf -v source_shell '%q' "$SOURCE_DIR"

  cat > "$ENV_FILE" <<EOF
#!/usr/bin/env bash
# Source this file before using this PyTorch installation.

if [[ "\${BASH_SOURCE[0]}" == "\$0" ]]; then
  echo "Source this file instead: source \${BASH_SOURCE[0]}" >&2
  exit 1
fi

PYTORCH_PREFIX=$prefix_shell
PYTORCH_SOURCE=$source_shell
PYTORCH_CUDA_HOME="\$PYTORCH_PREFIX/cuda-$CUDA_VERSION"
_pytorch_site_packages="\$PYTORCH_PREFIX/python/lib/python3.10/site-packages"
_pytorch_nvidia_dir="\$_pytorch_site_packages/nvidia"

export CUDA_HOME="\$PYTORCH_CUDA_HOME"
export CUDA_PATH="\$PYTORCH_CUDA_HOME"
export PATH="\$PYTORCH_PREFIX/python/bin:\$PYTORCH_CUDA_HOME/bin:\$PATH"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-\$PYTORCH_PREFIX/pip-cache}"
export TORCH_CUDA_ARCH_LIST="\${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="\${MAX_JOBS:-$MAX_JOBS}"

export LD_LIBRARY_PATH="\$PYTORCH_CUDA_HOME/lib64:\$PYTORCH_CUDA_HOME/lib:\$_pytorch_nvidia_dir/cuda_runtime/lib:\$_pytorch_nvidia_dir/cuda_nvrtc/lib:\$_pytorch_nvidia_dir/cuda_cupti/lib:\$_pytorch_nvidia_dir/cublas/lib:\$_pytorch_nvidia_dir/cudnn/lib:\$_pytorch_nvidia_dir/cufft/lib:\$_pytorch_nvidia_dir/curand/lib:\$_pytorch_nvidia_dir/cusolver/lib:\$_pytorch_nvidia_dir/cusparse/lib:\$_pytorch_nvidia_dir/cusparselt/lib:\$_pytorch_nvidia_dir/nccl/lib:\$_pytorch_nvidia_dir/nvjitlink/lib:\$_pytorch_nvidia_dir/nvtx/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export LIBRARY_PATH="\$PYTORCH_CUDA_HOME/lib64:\$PYTORCH_CUDA_HOME/lib:\$_pytorch_nvidia_dir/cublas/lib:\$_pytorch_nvidia_dir/cudnn/lib:\$_pytorch_nvidia_dir/nccl/lib\${LIBRARY_PATH:+:\$LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="\$PYTORCH_PREFIX/python\${CMAKE_PREFIX_PATH:+:\$CMAKE_PREFIX_PATH}"

unset _pytorch_site_packages _pytorch_nvidia_dir
EOF
  chmod 0644 "$ENV_FILE"
}

clone_pytorch() {
  mkdir -p -- "$(dirname -- "$SOURCE_DIR")"
  note "clone PyTorch 到：$SOURCE_DIR"
  git clone "$PYTORCH_REPOSITORY" "$SOURCE_DIR"
}

checkout_pytorch() {
  if [[ "$FORCE_RECLONE" -eq 1 ]]; then
    [[ "$SOURCE_DIR" != / && -n "$SOURCE_DIR" ]] || die '源码目录不安全，拒绝删除。'
    rm -rf -- "$SOURCE_DIR"
  fi

  if [[ ! -e "$SOURCE_DIR" ]]; then
    clone_pytorch
  elif [[ ! -d "$SOURCE_DIR/.git" ]]; then
    if [[ -z "$(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      rmdir -- "$SOURCE_DIR"
      clone_pytorch
    else
      die "源码目录已存在但不是 Git 仓库，且目录非空：$SOURCE_DIR"
    fi
  else
    note "复用已有 PyTorch 源码：$SOURCE_DIR"
  fi

  git -C "$SOURCE_DIR" diff --ignore-submodules=all --quiet || \
    die "源码目录存在已跟踪的非子模块修改：$SOURCE_DIR"
  git -C "$SOURCE_DIR" diff --cached --ignore-submodules=all --quiet || \
    die "源码目录存在暂存的非子模块修改：$SOURCE_DIR"
  git -C "$SOURCE_DIR" fetch --depth 1 origin "$PYTORCH_REVISION" >/dev/null 2>&1 || true
  git -C "$SOURCE_DIR" checkout --detach "$PYTORCH_REVISION"
  [[ $(git -C "$SOURCE_DIR" rev-parse HEAD) == "$PYTORCH_REVISION" ]] || \
    die 'PyTorch 源码提交与预期不一致。'
  git -C "$SOURCE_DIR" submodule update --init --recursive --force --jobs 8
  if git -C "$SOURCE_DIR" submodule status --recursive | grep -Eq '^[+-U]'; then
    die '存在未对齐的 PyTorch Git 子模块。'
  fi
}

prepare_build_environment() {
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  local nvidia_dir="$PYTHON/lib/python3.10/site-packages/nvidia"

  export USE_CUDA=1
  export USE_CUDNN=1
  export USE_NCCL=1
  export USE_SYSTEM_NCCL=1
  export USE_DISTRIBUTED=1
  export BUILD_TEST=0
  export USE_CUSPARSELT=0
  export USE_CUDSS=0
  export USE_CUFILE=0
  export PYTORCH_BUILD_VERSION="$PYTORCH_VERSION"
  export PYTORCH_BUILD_NUMBER=1
  export CMAKE_BUILD_TYPE=Release
  export CMAKE_FRESH=1
  export CUDNN_INCLUDE_DIR="$nvidia_dir/cudnn/include"
  export CUDNN_LIBRARY="$nvidia_dir/cudnn/lib/libcudnn.so.9"
  export CUDNN_ROOT="$nvidia_dir/cudnn"
  export NCCL_INCLUDE_DIR="$nvidia_dir/nccl/include"
  export NCCL_LIB_DIR="$nvidia_dir/nccl/lib"
  export CMAKE_PREFIX_PATH="$PYTHON;$CMAKE_PREFIX_PATH"
  export CC=${CC:-gcc}
  export CXX=${CXX:-g++}
}

build_wheel() {
  local log="$LOGS/build-$(date +%Y%m%d-%H%M%S).log"
  local wheel

  prepare_build_environment
  if [[ "$CLEAN_BUILD" -eq 1 ]]; then
    rm -rf -- "$SOURCE_DIR/build"
  fi

  note "开始编译 PyTorch $PYTORCH_VERSION；完整日志：$log"
  (
    cd "$SOURCE_DIR"
    "$PYTHON/bin/python" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$WHEELS" .
  ) 2>&1 | tee "$log"

  wheel=$(find "$WHEELS" -maxdepth 1 -type f -name "torch-$PYTORCH_VERSION-*.whl" -printf '%T@ %p\n' | \
    sort -nr | head -n 1 | cut -d' ' -f2-)
  [[ -n "$wheel" ]] || die "未找到生成的 torch-$PYTORCH_VERSION wheel。"
  printf '%s\n' "$(basename -- "$wheel")" > "$PREFIX/latest-wheel.txt"
}

install_wheel() {
  local wheel_name wheel
  wheel_name=$(<"$PREFIX/latest-wheel.txt")
  wheel="$WHEELS/$wheel_name"
  [[ -f "$wheel" ]] || die "wheel 不存在：$wheel"
  prepare_build_environment
  "$PYTHON/bin/python" -m pip install --force-reinstall --no-deps "$wheel"
}

reuse_existing_wheel() {
  local wheel
  wheel=$(find "$WHEELS" -maxdepth 1 -type f -name "torch-$PYTORCH_VERSION-*.whl" -printf '%T@ %p\n' | \
    sort -nr | head -n 1 | cut -d' ' -f2-)
  [[ -n "$wheel" ]] || return 1
  printf '%s\n' "$(basename -- "$wheel")" > "$PREFIX/latest-wheel.txt"
  install_wheel
  return 0
}

run_validation() {
  [[ "$RUN_TEST" -eq 1 ]] || return 0
  prepare_build_environment
  note '运行 PyTorch CUDA smoke test'
  "$PYTHON/bin/python" - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")

torch.manual_seed(0)
x = torch.randn((128, 128), device="cuda")
y = torch.randn((128, 128), device="cuda")
z = x @ y
torch.cuda.synchronize()
expected = torch.matmul(x, y)
max_error = (z - expected).abs().max().item()
print("device:", torch.cuda.get_device_name(0))
print("max error:", max_error)
if max_error != 0:
    raise SystemExit(f"unexpected max error: {max_error}")
PY
  "$PYTHON/bin/python" -m pip check
}

PREFIX=$DEFAULT_PREFIX
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
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-test)
      RUN_TEST=0
      shift
      ;;
    --clean-build)
      CLEAN_BUILD=1
      shift
      ;;
    --force-reclone)
      FORCE_RECLONE=1
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
[[ "$PREFIX" != *$'\n'* && "$PREFIX" != *$'\r'* ]] || die '--prefix 不能包含换行符。'
[[ "$SOURCE_DIR" != *$'\n'* && "$SOURCE_DIR" != *$'\r'* ]] || die '--source-dir 不能包含换行符。'

PREFIX=$(canonicalize_existing_or_parent "$PREFIX")
SOURCE_DIR=$(canonicalize_existing_or_parent "$SOURCE_DIR")
[[ "$PREFIX" != / ]] || die '不能把 / 作为 --prefix。'
[[ "$SOURCE_DIR" != / ]] || die '不能把 / 作为 --source-dir。'

DOWNLOADS="$PREFIX/downloads"
PYTHON="$PREFIX/python-3.10.20"
PYTHON_LINK="$PREFIX/python"
CUDA_HOME="$PREFIX/cuda-$CUDA_VERSION"
WHEELS="$PREFIX/wheels"
LOGS="$PREFIX/logs"
PIP_CACHE_DIR="$PREFIX/pip-cache"
ENV_FILE="$PREFIX/env-pytorch.sh"

mkdir -p "$PREFIX" "$DOWNLOADS" "$WHEELS" "$LOGS" "$PIP_CACHE_DIR"

check_platform
install_python
install_cuda_toolkit
install_cuda_python_libraries
write_environment_file
checkout_pytorch

if [[ "$SKIP_BUILD" -eq 1 ]]; then
  note '源码、依赖和环境已准备完毕，按请求跳过源码编译。'
else
  if [[ -f "$PREFIX/latest-wheel.txt" && -f "$WHEELS/$(<"$PREFIX/latest-wheel.txt")" ]]; then
    note "复用已有 wheel：$(<"$PREFIX/latest-wheel.txt")"
    install_wheel
  elif reuse_existing_wheel; then
    note "复用已有 wheel：$(<"$PREFIX/latest-wheel.txt")"
  else
    build_wheel
    install_wheel
  fi

  note 'PyTorch 已安装。'
  printf '环境脚本：source %s\n' "$ENV_FILE"
  printf '源码目录：%s\n' "$SOURCE_DIR"
  printf '安装目录：%s\n' "$PREFIX"
  printf 'wheel 目录：%s\n' "$WHEELS"

  run_validation
fi
