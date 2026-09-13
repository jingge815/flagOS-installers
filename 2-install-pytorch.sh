#!/usr/bin/env bash
# Install a pinned PyTorch CUDA wheel and FlagTree's PIM Triton without root.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEFAULT_PREFIX="$SCRIPT_DIR/../flagOS-installed/pytorch"
DEFAULT_FLAGTREE_PREFIX="$SCRIPT_DIR/../flagOS-installed/flagTree"

# 两套固定版本：有 GPU 装 CUDA 版，没有就装 CPU 版。torch 版本号一致，只差后缀。
# CPU 版少掉十几个 nvidia-*/cuda-* 依赖（cuda-toolkit、cudnn、nccl 等，共数 GB），
# 纯 CPU 机器上一个都用不到——算子编译只需要 TTIR → pim mlir，不执行 GPU kernel。
PYTORCH_WHEEL_VERSION_CUDA=2.9.1+cu128
PYTORCH_WHEEL_INDEX_CUDA=https://download.pytorch.org/whl/cu128
PYTORCH_WHEEL_VERSION_CPU=2.9.1+cpu
PYTORCH_WHEEL_INDEX_CPU=https://download.pytorch.org/whl/cpu
# 空表示按机器实际情况自动判断；--torch-cpu / --torch-cuda 可强制。
TORCH_CPU_ONLY=""
# 非空时从本地目录装 torch wheel，不联网（离线或网络受限时用）。
PYTORCH_WHEEL_DIR=""
PYTHON_ARCHIVE=cpython-3.10.20+20260718-x86_64-unknown-linux-gnu-install_only.tar.gz
PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260718/cpython-3.10.20%2B20260718-x86_64-unknown-linux-gnu-install_only.tar.gz

RUN_TEST=1

usage() {
  cat <<EOF
用法：
  bash 2-install-pytorch.sh [选项]

选项：
  --prefix DIR           安装目录，默认：../flagOS-installed/pytorch
  --flagtree-prefix DIR  0-install-flagtree.sh 安装目录，默认：../flagOS-installed/flagTree
  --skip-test            跳过安装后的 smoke test
  --torch-cpu            强制装 CPU 版 torch（跳过十几个 nvidia-*/cuda-* 包）
  --wheel-dir DIR        从本地目录装 torch wheel，不联网（离线/网络受限时用）
  --torch-cuda           强制装 CUDA 版 torch
  -h, --help             显示帮助

固定版本：
  有 GPU：torch==$PYTORCH_WHEEL_VERSION_CUDA（索引 $PYTORCH_WHEEL_INDEX_CUDA）
  无 GPU：torch==$PYTORCH_WHEEL_VERSION_CPU（索引 $PYTORCH_WHEEL_INDEX_CPU）
  Python：3.10.20

说明：
  本脚本不需要 root，不编译 PyTorch，也不安装 CUDA Toolkit——只下载官方 wheel 并
  配置环境，然后把 FlagTree 的 PIM Triton 同步进这个 Python 环境。请先成功运行
  0-install-flagtree.sh。

  GPU 是可选的：探测不到 nvidia-smi 就装 CPU 版 torch，其余步骤完全相同。注意
  Triton 同步里的 backends/nvidia/{bin,include} 在纯 CPU 上**也要同步**——
  图编译器需要其中的 cuda.h 和 ptxas，它们是随 pip 包分发的文件，不需要驱动。

  目标机器需 Ubuntu 22.04 x86_64；有 GPU 时还需 570+ NVIDIA 驱动（CUDA 12.8 要求）。
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
  local driver_major

  [[ -r /etc/os-release ]] || die '无法读取 /etc/os-release。'
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 22.04 ]] || \
    die "需要 Ubuntu 22.04，当前为 ${PRETTY_NAME:-未知系统}。"
  [[ $(uname -m) == x86_64 ]] || die "需要 x86_64，当前为 $(uname -m)。"

  require_command tar
  require_command gzip
  require_command awk
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    die '缺少 curl 或 wget。'
  fi

  # GPU 可选。没显式指定就按机器实际情况判断装哪个 wheel。
  if [[ -z "$TORCH_CPU_ONLY" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
      TORCH_CPU_ONLY=false
    else
      TORCH_CPU_ONLY=true
    fi
  fi

  if [[ "$TORCH_CPU_ONLY" == true ]]; then
    PYTORCH_WHEEL_VERSION=$PYTORCH_WHEEL_VERSION_CPU
    PYTORCH_WHEEL_INDEX=$PYTORCH_WHEEL_INDEX_CPU
    note '未检测到 NVIDIA GPU（或指定了 --torch-cpu）：装 CPU 版 torch。'
    note '算子编译不需要 GPU 硬件——只做 TTIR → pim mlir，不执行 GPU kernel。'
  else
    PYTORCH_WHEEL_VERSION=$PYTORCH_WHEEL_VERSION_CUDA
    PYTORCH_WHEEL_INDEX=$PYTORCH_WHEEL_INDEX_CUDA
    # CUDA 12.8 对驱动有硬要求，这一条只在真要装 CUDA 版时才检查。
    nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi 不可用；纯 CPU 机器请加 --torch-cpu。'
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk -F. 'NR == 1 { print $1 }')
    [[ "$driver_major" =~ ^[0-9]+$ && "$driver_major" -ge 570 ]] || \
      die "CUDA 12.8 需要 570+ NVIDIA 驱动，当前主版本为 ${driver_major:-未知}。"
    note "检测到 NVIDIA GPU（驱动 $driver_major）：装 CUDA 版 torch。"
  fi
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
  PYTHONNOUSERSITE=1 "$PYTHON/bin/python" -m pip install --cache-dir "$PIP_CACHE_DIR" \
    'pip<27' setuptools==83.0.0 wheel==0.47.0 numpy

  if [[ -e "$PYTHON_LINK" && ! -L "$PYTHON_LINK" ]]; then
    die "Python 链接路径已被普通目录或文件占用：$PYTHON_LINK"
  fi
  ln -sfn "$(basename -- "$PYTHON")" "$PYTHON_LINK"
}

write_environment_file() {
  local prefix_shell
  printf -v prefix_shell '%q' "$PREFIX"

  cat > "$ENV_FILE" <<EOF
#!/usr/bin/env bash
# Source this file before using this PyTorch installation.

if [[ "\${BASH_SOURCE[0]}" == "\$0" ]]; then
  echo "Source this file instead: source \${BASH_SOURCE[0]}" >&2
  exit 1
fi

PYTORCH_PREFIX=$prefix_shell
_pytorch_site_packages="\$PYTORCH_PREFIX/python/lib/python3.10/site-packages"
_pytorch_nvidia_dir="\$_pytorch_site_packages/nvidia"

export PATH="\$PYTORCH_PREFIX/python/bin:\$PATH"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-\$PYTORCH_PREFIX/pip-cache}"
export LD_LIBRARY_PATH="\$_pytorch_nvidia_dir/cuda_runtime/lib:\$_pytorch_nvidia_dir/cuda_nvrtc/lib:\$_pytorch_nvidia_dir/cuda_cupti/lib:\$_pytorch_nvidia_dir/cublas/lib:\$_pytorch_nvidia_dir/cudnn/lib:\$_pytorch_nvidia_dir/cufft/lib:\$_pytorch_nvidia_dir/curand/lib:\$_pytorch_nvidia_dir/cusolver/lib:\$_pytorch_nvidia_dir/cusparse/lib:\$_pytorch_nvidia_dir/cusparselt/lib:\$_pytorch_nvidia_dir/nccl/lib:\$_pytorch_nvidia_dir/nvjitlink/lib:\$_pytorch_nvidia_dir/nvtx/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="\$PYTORCH_PREFIX/python\${CMAKE_PREFIX_PATH:+:\$CMAKE_PREFIX_PATH}"

unset _pytorch_site_packages _pytorch_nvidia_dir
EOF
  chmod 0644 "$ENV_FILE"
}

install_pytorch_wheel() {
  # 已经装好同一版本就不重复下载：torch wheel 是 184 MB（CPU）/ 约 2.5 GB（CUDA），
  # 网络不稳时重跑脚本会卡在这一步，而这一步本来就是幂等的。
  if PYTHONNOUSERSITE=1 "$PYTHON/bin/python" - <<PY >/dev/null 2>&1
import sys, torch
sys.exit(0 if torch.__version__ == "$PYTORCH_WHEEL_VERSION" else 1)
PY
  then
    note "PyTorch $PYTORCH_WHEEL_VERSION 已安装，跳过下载"
    return
  fi

  # 网络受限时，把 torch wheel 预先放到一个目录，用 --wheel-dir 指过来。
  # 只有 torch 自己从本地取（它是最大的一个，184 MB / 2.5 GB）；filelock、sympy 等
  # 小依赖仍走索引，所以这里不能加 --no-index。
  if [[ -n "$PYTORCH_WHEEL_DIR" ]]; then
    [[ -d "$PYTORCH_WHEEL_DIR" ]] || die "--wheel-dir 不是目录：$PYTORCH_WHEEL_DIR"
    note "从本地目录安装 PyTorch $PYTORCH_WHEEL_VERSION（$PYTORCH_WHEEL_DIR）"
    PYTHONNOUSERSITE=1 "$PYTHON/bin/python" -m pip install --upgrade \
      --cache-dir "$PIP_CACHE_DIR" --find-links "$PYTORCH_WHEEL_DIR" \
      --index-url "$PYTORCH_WHEEL_INDEX" \
      "torch==$PYTORCH_WHEEL_VERSION"
    return
  fi

  note "通过官方 wheel 安装 PyTorch $PYTORCH_WHEEL_VERSION（索引 $PYTORCH_WHEEL_INDEX）"
  PYTHONNOUSERSITE=1 "$PYTHON/bin/python" -m pip install --upgrade \
    --cache-dir "$PIP_CACHE_DIR" --index-url "$PYTORCH_WHEEL_INDEX" \
    "torch==$PYTORCH_WHEEL_VERSION"
}

sync_triton_to_pytorch() {
  local flagtree_triton="$FLAGTREE_PREFIX/python-3.10.20/lib/python3.10/site-packages/triton"
  local pytorch_triton="$PYTHON/lib/python3.10/site-packages/triton"
  local backup_dir="$PREFIX/.triton-backup-pre-pim"
  local source_path destination_path
  local required_files=(
    '_C/libtriton.so'
    'backends/pim_sidecar.py'
    'backends/nvidia/compiler.py'
  )
  local required_directories=(
    'backends/nvidia/bin'
    'backends/nvidia/include'
    'backends/nvidia/lib/cupti'
  )

  [[ -d "$flagtree_triton" ]] || \
    die "找不到 FlagTree Triton：$flagtree_triton。请先运行 0-install-flagtree.sh。"
  [[ -d "$pytorch_triton" ]] || die "PyTorch wheel 未安装 Triton：$pytorch_triton"

  for source_path in "${required_files[@]}"; do
    [[ -f "$flagtree_triton/$source_path" ]] || \
      die "FlagTree Triton 缺少同步文件：$flagtree_triton/$source_path"
  done
  [[ -f "$pytorch_triton/_C/libtriton.so" ]] || \
    die "PyTorch Triton 缺少目标文件：$pytorch_triton/_C/libtriton.so"
  [[ -f "$pytorch_triton/backends/nvidia/compiler.py" ]] || \
    die "PyTorch Triton 缺少目标文件：$pytorch_triton/backends/nvidia/compiler.py"
  for source_path in "${required_directories[@]}"; do
    [[ -d "$flagtree_triton/$source_path" ]] || \
      die "FlagTree Triton 缺少同步目录：$flagtree_triton/$source_path"
  done

  note "把 FlagTree 的 PIM Triton 同步进 PyTorch 环境：$pytorch_triton"
  mkdir -p "$backup_dir"
  cp -f "$pytorch_triton/_C/libtriton.so" \
    "$backup_dir/libtriton.so.orig.$(date +%s)"
  cp -f "$pytorch_triton/backends/nvidia/compiler.py" \
    "$backup_dir/compiler.py.orig.$(date +%s)"

  for source_path in "${required_files[@]}"; do
    cp -f "$flagtree_triton/$source_path" "$pytorch_triton/$source_path"
  done
  for source_path in "${required_directories[@]}"; do
    destination_path="$pytorch_triton/$source_path"
    rm -rf -- "$destination_path"
    cp -r "$flagtree_triton/$source_path" "$destination_path"
  done
  rm -rf -- "$pytorch_triton/backends/__pycache__" "$pytorch_triton/backends/nvidia/__pycache__"
}

run_validation() {
  [[ "$RUN_TEST" -eq 1 ]] || return 0
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  note '运行 PyTorch 和 PIM Triton smoke test'
  "$PYTHON/bin/python" - <<'PY'
import torch
from triton._C.libtriton import passes

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

# PIM pass 是这一步真正的判据：图编译器全靠它把 TTIR 降到 pim mlir，缺了就没有
# 任何后续可言。这一条与有没有 GPU 无关——pass 是 CPU 上跑的 MLIR 变换。
if not hasattr(passes, "pim"):
    raise SystemExit("PIM Triton passes are unavailable")
print("PIM Triton passes: OK")

# 矩阵乘只是验证这份 torch 能算数，在哪个设备上算不影响本脚本的目的。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
x = torch.randn((128, 128), device=DEVICE)
y = torch.randn((128, 128), device=DEVICE)
z = x @ y
if DEVICE == "cuda":
    torch.cuda.synchronize()
    print("device:", torch.cuda.get_device_name(0))
else:
    print("device: cpu（未检测到 GPU；算子编译不需要 GPU 硬件）")
max_error = (z - torch.matmul(x, y)).abs().max().item()
print("max error:", max_error)
if max_error != 0:
    raise SystemExit(f"unexpected max error: {max_error}")
PY
  "$PYTHON/bin/python" -m pip check
}

PREFIX=$DEFAULT_PREFIX
FLAGTREE_PREFIX=$DEFAULT_FLAGTREE_PREFIX

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      [[ $# -ge 2 ]] || die '--prefix 缺少目录参数。'
      PREFIX=$2
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
    --torch-cpu)
      TORCH_CPU_ONLY=true
      shift
      ;;
    --wheel-dir)
      [[ $# -ge 2 ]] || die '--wheel-dir 缺少目录参数。'
      PYTORCH_WHEEL_DIR=$2
      shift 2
      ;;
    --torch-cuda)
      TORCH_CPU_ONLY=false
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
[[ -n "$FLAGTREE_PREFIX" ]] || die '--flagtree-prefix 不能为空。'
[[ "$PREFIX" != *$'\n'* && "$PREFIX" != *$'\r'* ]] || die '--prefix 不能包含换行符。'
[[ "$FLAGTREE_PREFIX" != *$'\n'* && "$FLAGTREE_PREFIX" != *$'\r'* ]] || die '--flagtree-prefix 不能包含换行符。'

PREFIX=$(canonicalize_existing_or_parent "$PREFIX")
FLAGTREE_PREFIX=$(canonicalize_existing_or_parent "$FLAGTREE_PREFIX")
[[ "$PREFIX" != / ]] || die '不能把 / 作为 --prefix。'
[[ "$FLAGTREE_PREFIX" != / ]] || die '不能把 / 作为 --flagtree-prefix。'

DOWNLOADS="$PREFIX/downloads"
PYTHON="$PREFIX/python-3.10.20"
PYTHON_LINK="$PREFIX/python"
PIP_CACHE_DIR="$PREFIX/pip-cache"
ENV_FILE="$PREFIX/env-pytorch.sh"

mkdir -p "$PREFIX" "$DOWNLOADS" "$PIP_CACHE_DIR"

check_platform
install_python
write_environment_file
install_pytorch_wheel
sync_triton_to_pytorch

note 'PyTorch 已安装，并已同步 FlagTree 的 PIM Triton。'
printf '环境脚本：source %s\n' "$ENV_FILE"
printf '安装目录：%s\n' "$PREFIX"
printf 'FlagTree 目录：%s\n' "$FLAGTREE_PREFIX"

run_validation
