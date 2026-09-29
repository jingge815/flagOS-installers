#!/usr/bin/env bash
# Build and install FlagTree Triton 3.5 into a user-owned prefix.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEFAULT_SOURCE_DIR="$SCRIPT_DIR/FlagTree"
DEFAULT_PYTORCH_PREFIX="$SCRIPT_DIR/../flagOS-installed/pytorch"
MAX_JOBS=${MAX_JOBS:-8}
RUN_TEST=1
# 构建完成后把新编译的 PIM Triton 同步进 PyTorch 环境；--skip-pytorch-sync 关掉。
SYNC_PYTORCH=1

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
  --prefix DIR         安装目录，默认：../flagOS-installed/flagTree
  --source-dir DIR     FlagTree 源码目录，默认：./FlagTree
  --max-jobs N         编译并行度，默认：$MAX_JOBS
  --pytorch-prefix DIR 要同步 PIM Triton 的 PyTorch 安装目录，默认：../flagOS-installed/pytorch
  --skip-pytorch-sync  不同步 PIM Triton 到 PyTorch 环境
  --skip-test          跳过安装后的验证
  -h, --help           显示帮助

说明：
  本脚本不需要 root，不安装 NVIDIA 驱动。

  构建并验证通过后，如果 --pytorch-prefix 里已经有一个装好的 PyTorch 环境（即
  2-install-pytorch.sh 跑过一次），本脚本会把刚构建出来的 PIM Triton 同步进去。因此
  重建 FlagTree 之后不需要再跑一次 2-install-pytorch.sh：source
  <pytorch-prefix>/env-pytorch.sh 用到的就是新编译的 triton。PyTorch 前缀还没装好时
  这一步跳过并提示，等 2-install-pytorch.sh 自己完成首次同步。
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

# 只规范化、不创建任何东西。第一次安装的顺序是 0 → 1 → 2，跑 0 的时候 PyTorch 前缀
# 根本还不存在：这里不能像 2-install-pytorch.sh 的同名函数那样顺手 mkdir（那是它接下来
# 就要写入的目录），也不能因为路径不存在就报错。存在与否交给 pytorch_env_is_installed
# 判断，不存在就当这次不需要同步。目录不存在时原样返回，相对路径由调用时的 cwd 解释。
canonicalize_path() {
  local path=$1
  if [[ -d "$path" ]]; then
    (cd -- "$path" && pwd -P)
  else
    printf '%s\n' "$path"
  fi
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
  [[ "${ID:-}" == ubuntu && ( "${VERSION_ID:-}" == 22.04 || "${VERSION_ID:-}" == 24.04 ) ]] || \
    die "需要 Ubuntu 22.04 或 24.04，当前为 ${PRETTY_NAME:-未知系统}。"
  [[ $(uname -m) == x86_64 ]] || die "需要 x86_64，当前为 $(uname -m)。"

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
  # GPU 是可选的：本脚本编译的是 triton（含 PIM pass）和 LLVM，全程在 CPU 上跑，
  # ptxas / cuda.h / libdevice 都来自下载的 tarball，不需要驱动。有卡就报一下型号，
  # 没卡也照常装——纯 CPU 机器上算子编译走 opcompiler_bridge/cpu_host.py 的前端路径，
  # 产出的 pim mlir 与有卡路径逐字节相同。
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    note "检测到 NVIDIA GPU：$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  else
    note '未检测到 NVIDIA GPU，按纯 CPU 模式安装（算子编译不需要 GPU 硬件）。'
  fi
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
  local package archive extract_dir
  mkdir -p "$SYSROOT/usr/lib" "$DEBS"
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

      # 有些 deb（比如 zlib1g）的文件路径还是 usrmerge 之前的 /lib/...，
      # 而不是 /usr/lib/...；真实的 Ubuntu 系统里 /lib 是指向 /usr/lib 的
      # 符号链接，两者其实是同一个目录。这里的 SYSROOT 是从空目录搭建的，
      # 没有这个链接，且 dpkg-deb -x 解包时会把 deb 里的 ./lib/ 目录项
      # 实体化成一个真实目录（覆盖掉预先建好的符号链接），所以没法通过
      # 提前建好 lib -> usr/lib 链接来处理。改成解到临时目录，再把
      # lib/ 下的内容搬进 usr/lib/，让效果等价于真实系统上的 usrmerge。
      extract_dir=$(mktemp -d "$DOWNLOADS/sysroot-extract.XXXXXX")
      dpkg-deb -x "$archive" "$extract_dir"
      if [[ -d "$extract_dir/lib" ]]; then
        mkdir -p "$SYSROOT/usr/lib"
        cp -an "$extract_dir/lib/." "$SYSROOT/usr/lib/"
        rm -rf -- "$extract_dir/lib"
      fi
      cp -an "$extract_dir/." "$SYSROOT/"
      rm -rf -- "$extract_dir"
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

# ---- 把刚构建出来的 PIM Triton 同步进 PyTorch 环境 ------------------------------
# 这份逻辑与 2-install-pytorch.sh 里的 sync_triton_to_pytorch 是同一套（同步清单、
# CPU/GPU 两条分支、必需文件检查都保持一致）：那边是安装 PyTorch 时做首次同步，这边
# 是重建 FlagTree 之后再同步一次，省掉「重建完还要再跑一遍 2-install-pytorch.sh」。
# 改同步清单时两处都要改。刻意不同的只有两处，见下面各自的注释。

find_unique_site_packages() {
  local python_root=$1
  local matches=("$python_root"/lib/python3.*/site-packages)

  [[ ${#matches[@]} -eq 1 && -d "${matches[0]}" ]] || \
    die "在 $python_root/lib 下未找到唯一的 python3.*/site-packages 目录（找到 ${#matches[@]} 个）。"
  printf '%s\n' "${matches[0]}"
}

# PyTorch 前缀里是否已有装好的 Python，也就是 2-install-pytorch.sh 是否跑过。
pytorch_env_is_installed() {
  local matches=("$PYTORCH_PREFIX"/python/lib/python3.*/site-packages)
  [[ ${#matches[@]} -eq 1 && -d "${matches[0]}" ]]
}

# 同目录临时副本 + rename 替换。与 2-install-pytorch.sh 的 cp -f 不同：本脚本会在实验
# 过程中被反复执行，这时 PyTorch 环境很可能正跑着训练或图编译器——cp -f 是就地截断，
# 已经被 mmap 的 libtriton.so 会把那个进程直接带崩；先删后拷的目录同理，会留下一段
# 「文件不全」的窗口。rename 之后旧 inode 仍然完整，正在跑的进程不受影响。
replace_atomically() {
  local source_path=$1
  local destination_path=$2
  local temporary="${destination_path}.flagtree-new.$$"

  rm -rf -- "$temporary"
  if [[ -d "$source_path" ]]; then
    cp -r -- "$source_path" "$temporary" || { rm -rf -- "$temporary"; die "复制目录失败：$source_path"; }
  else
    cp -f -- "$source_path" "$temporary" || { rm -f -- "$temporary"; die "复制文件失败：$source_path"; }
  fi
  rm -rf -- "$destination_path"
  mv -- "$temporary" "$destination_path"
}

sync_triton_to_pytorch() {
  local pytorch_python_link="$PYTORCH_PREFIX/python"
  local flagtree_site_packages pytorch_site_packages
  local flagtree_triton pytorch_triton

  flagtree_site_packages=$(find_unique_site_packages "$PYTHON_LINK")
  pytorch_site_packages=$(find_unique_site_packages "$pytorch_python_link")
  flagtree_triton="$flagtree_site_packages/triton"
  pytorch_triton="$pytorch_site_packages/triton"
  local backup_dir="$PYTORCH_PREFIX/.triton-backup-pre-pim"
  local existing_backups
  local source_path
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
    die "找不到刚构建出来的 FlagTree Triton：$flagtree_triton"

  # CPU 版 torch wheel 不依赖 triton（只有 CUDA 版才带），PyTorch 侧没有可覆盖的
  # 上游 triton，把 FlagTree 那份连同 dist-info 整体装进去；带上 dist-info，pip 才
  # 知道 triton 已经装好，后续步骤不会再去拉一份上游 triton 把它盖掉。
  if [[ ! -d "$pytorch_triton" ]]; then
    note "PyTorch 侧没有 triton（CPU 版 wheel 不带），整体安装 FlagTree 的 PIM Triton"
    cp -r -- "$flagtree_triton" "$pytorch_triton"
    local dist_info
    for dist_info in "$flagtree_site_packages"/triton-*.dist-info; do
      [[ -d "$dist_info" ]] || \
        die "找不到 FlagTree Triton 的 dist-info：$flagtree_site_packages/triton-*.dist-info"
      cp -r -- "$dist_info" "$pytorch_site_packages/"
    done
    return
  fi

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

  # 上一次运行若在复制中途退出，会留下 .flagtree-new.<pid> 临时文件/目录。
  find "$pytorch_triton" -maxdepth 4 -name '*.flagtree-new.*' -exec rm -rf -- {} + 2>/dev/null || true

  # 与 2-install-pytorch.sh 的第二处不同：备份只做一次。那边每跑一次备份一份带时间戳的
  # 原文件，一年也跑不了几次；这边每次重建都会走到，留一份 798 MB 的旧 libtriton.so 就够，
  # 逐次累积会把磁盘吃光。
  mkdir -p "$backup_dir"
  existing_backups=("$backup_dir"/libtriton.so.orig.*)
  if [[ ! -e "${existing_backups[0]}" ]]; then
    cp -f "$pytorch_triton/_C/libtriton.so" "$backup_dir/libtriton.so.orig.$(date +%s)"
  fi
  existing_backups=("$backup_dir"/compiler.py.orig.*)
  if [[ ! -e "${existing_backups[0]}" ]]; then
    cp -f "$pytorch_triton/backends/nvidia/compiler.py" "$backup_dir/compiler.py.orig.$(date +%s)"
  fi

  for source_path in "${required_files[@]}"; do
    replace_atomically "$flagtree_triton/$source_path" "$pytorch_triton/$source_path"
  done
  for source_path in "${required_directories[@]}"; do
    replace_atomically "$flagtree_triton/$source_path" "$pytorch_triton/$source_path"
  done
  rm -rf -- "$pytorch_triton/backends/__pycache__" "$pytorch_triton/backends/nvidia/__pycache__"
}

# 同步完顺手验一次，判据与 2-install-pytorch.sh 的 smoke test 一致：PIM pass 在不在。
# 只提示不致命——PyTorch 环境自身的问题不该否掉一次已经成功的 FlagTree 构建。
verify_pytorch_triton() {
  local pytorch_env="$PYTORCH_PREFIX/env-pytorch.sh"
  [[ -f "$pytorch_env" ]] || return 0

  if ! (
    # shellcheck disable=SC1090
    source "$pytorch_env"
    "$PYTORCH_PREFIX/python/bin/python" - <<'PY'
import triton
from triton._C.libtriton import passes

print("triton:", triton.__version__, triton.__file__)
if not hasattr(passes, "pim"):
    raise SystemExit("PyTorch 环境里的 PIM passes 不可用")
print("PyTorch 环境 PIM Triton: OK")
PY
  ); then
    printf '警告：同步完成，但 PyTorch 环境检查未通过；请手动确认：source %s\n' "$pytorch_env" >&2
  fi
}

sync_pytorch_triton_if_installed() {
  if ! pytorch_env_is_installed; then
    note "PyTorch 前缀里还没有装好的 Python（$PYTORCH_PREFIX），跳过 PIM Triton 同步"
    note '2-install-pytorch.sh 安装时会完成首次同步；之后重建 FlagTree 由本脚本同步'
    return 0
  fi

  sync_triton_to_pytorch
  verify_pytorch_triton
  printf 'PyTorch 环境现在用的是这次构建的 triton：source %s\n' "$PYTORCH_PREFIX/env-pytorch.sh"
}

run_validation() {
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  mkdir -p "$MLIR_DUMP_DIR"
  "$PYTHON/bin/python" -c 'import torch, triton; print("imports ok")'
  "$PYTHON/bin/python" "$EXAMPLES_DIR/matmul_sm80.py"
}

PREFIX=
SOURCE_DIR=$DEFAULT_SOURCE_DIR
PYTORCH_PREFIX=$DEFAULT_PYTORCH_PREFIX

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
    --pytorch-prefix)
      [[ $# -ge 2 ]] || die '--pytorch-prefix 缺少目录参数。'
      PYTORCH_PREFIX=$2
      shift 2
      ;;
    --skip-pytorch-sync)
      SYNC_PYTORCH=0
      shift
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
[[ -n "$PREFIX" ]] || die '--prefix 不能为空。'
[[ "$PREFIX" != *$'\n'* && "$PREFIX" != *$'\r'* ]] || die '--prefix 不能包含换行符。'
mkdir -p -- "$PREFIX"
PREFIX=$(cd -- "$PREFIX" && pwd -P)
[[ "$PREFIX" != / ]] || die '不能把 / 作为 --prefix。'

[[ -n "$PYTORCH_PREFIX" ]] || die '--pytorch-prefix 不能为空。'
[[ "$PYTORCH_PREFIX" != *$'\n'* && "$PYTORCH_PREFIX" != *$'\r'* ]] || die '--pytorch-prefix 不能包含换行符。'
PYTORCH_PREFIX=$(canonicalize_path "$PYTORCH_PREFIX")
[[ "$PYTORCH_PREFIX" != / ]] || die '不能把 / 作为 --pytorch-prefix。'

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

note 'FlagTree 已安装。'
printf '环境脚本：source %s\n' "$ENV_FILE"
printf '源码目录：%s\n' "$SOURCE_DIR"
printf 'wheel 目录：%s\n' "$WHEELS"
printf 'MLIR dump 目录：%s\n' "$MLIR_DUMP_DIR"

if [[ $RUN_TEST -eq 1 ]]; then
  run_validation
fi

# 放在验证之后：只有这次构建自己先跑通了，才把结果同步进 PyTorch 环境。
if [[ $SYNC_PYTORCH -eq 1 ]]; then
  sync_pytorch_triton_if_installed
fi
