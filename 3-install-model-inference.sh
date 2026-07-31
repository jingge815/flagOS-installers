#!/usr/bin/env bash
# Validate the local model inference installer interface and prerequisites.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEFAULT_PREFIX="$SCRIPT_DIR/../flagOS-installed/model-inference"
DEFAULT_SOURCE_DIR="$SCRIPT_DIR/model-inference"
DEFAULT_FLAGTREE_PREFIX="$SCRIPT_DIR/../flagOS-installed/flagTree"
DEFAULT_FLAGGEMS_PREFIX="$SCRIPT_DIR/../flagOS-installed/flagGems"
DEFAULT_PYTORCH_PREFIX="$SCRIPT_DIR/../flagOS-installed/pytorch"
DEFAULT_MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT="Explain in one sentence what FlagGems does for PyTorch."
DEFAULT_MAX_NEW_TOKENS=32

PYTORCH_MODE=auto
RUN_TEST=1
SKIP_DOWNLOAD=0
SKIP_INFERENCE=0
COMPARE_BASELINE=0

usage() {
  cat <<EOF
用法：
  bash 3-install-model-inference.sh [选项]

选项：
  --prefix DIR            安装目录，默认：../flagOS-installed/model-inference
  --source-dir DIR        本地 model-inference 源码目录，默认：./model-inference
  --flagtree-prefix DIR   0-install-flagtree.sh 安装目录，默认：../flagOS-installed/flagTree
  --flaggems-prefix DIR   1-install-flaggems.sh 安装目录，默认：../flagOS-installed/flagGems
  --pytorch-prefix DIR    2-install-pytorch.sh 安装目录，默认：../flagOS-installed/pytorch
  --pytorch-mode MODE     PyTorch 来源：auto、compiled 或 wheel，默认：auto
  --model-id ID           Hugging Face 模型 ID，默认：$DEFAULT_MODEL_ID
  --revision REV          Hugging Face 模型 revision
  --model-path DIR        已下载的本地模型目录
  --local-dir DIR         模型下载目录
  --prompt TEXT           推理提示词，默认：$DEFAULT_PROMPT
  --max-new-tokens N      生成 token 数，默认：$DEFAULT_MAX_NEW_TOKENS
  --skip-test             跳过 NVIDIA GPU 可用性检查
  --skip-download         跳过模型下载
  --skip-inference        跳过推理运行
  --compare-baseline      同时运行非 FlagGems baseline
  -h, --help              显示帮助

说明：
  当前任务只实现参数、路径、平台和前置条件校验；不会安装依赖、下载模型或运行推理。
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

reject_newline_path() {
  local name=$1
  local path=$2
  [[ "$path" != *$'\n'* && "$path" != *$'\r'* ]] || die "$name 不能包含换行符。"
}

canonicalize_path() {
  local path=$1
  local target parent previous_parent base

  [[ -n "$path" ]] || die '路径不能为空。'
  reject_newline_path '路径' "$path"

  if [[ "$path" == /* ]]; then
    target=$path
  else
    target=$PWD/$path
  fi

  if [[ -d "$target" ]]; then
    (cd -- "$target" && pwd -P)
  else
    parent=$(dirname -- "$target")
    base=$(basename -- "$target")
    while [[ ! -d "$parent" ]]; do
      previous_parent=$parent
      base="$(basename -- "$parent")/$base"
      parent=$(dirname -- "$parent")
      [[ "$parent" != "$previous_parent" ]] || die "无法规范化路径：$path"
    done
    parent=$(cd -- "$parent" && pwd -P)
    printf '%s/%s\n' "$parent" "$base"
  fi
}

require_user_owned_path() {
  local name=$1
  local path=$2
  local probe=$path

  while [[ ! -e "$probe" ]]; do
    probe=$(dirname -- "$probe")
  done

  [[ -O "$probe" ]] || die "$name 不属于当前用户：$probe"
}

reject_root_prefix() {
  local name=$1
  local path=$2
  [[ "$path" != / ]] || die "不能把 / 作为 $name。"
}

check_platform() {
  [[ -r /etc/os-release ]] || die '无法读取 /etc/os-release。'
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 22.04 ]] || \
    die "需要 Ubuntu 22.04，当前为 ${PRETTY_NAME:-未知系统}。"
  [[ $(uname -m) == x86_64 ]] || die "需要 x86_64，当前为 $(uname -m)。"

  require_command git
  require_command tar
  require_command awk
  require_command sed
  require_command find
  require_command python3
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    die '缺少 curl 或 wget。'
  fi

  if [[ "$RUN_TEST" -eq 1 ]]; then
    require_command nvidia-smi
    nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi 不可用，请先确认 NVIDIA 驱动和 GPU，或使用 --skip-test。'
  fi
}

validate_prerequisites() {
  local env_flagtree="$FLAGTREE_PREFIX/env-flagtree.sh"
  local env_flaggems="$FLAGGEMS_PREFIX/env-flaggems.sh"
  local flagtree_python="$FLAGTREE_PREFIX/python/bin/python"
  local inference_entrypoint="$SOURCE_DIR/examples/run_llm_with_flaggems.py"

  [[ -f "$env_flagtree" ]] || die "找不到 FlagTree 环境脚本：$env_flagtree。请先运行 0-install-flagtree.sh。"
  [[ -f "$env_flaggems" ]] || die "找不到 FlagGems 环境脚本：$env_flaggems。请先运行 1-install-flaggems.sh。"
  [[ -x "$flagtree_python" ]] || die "找不到可执行的 FlagTree Python：$flagtree_python。"
  [[ -f "$inference_entrypoint" ]] || die "找不到推理入口：$inference_entrypoint。"

  if [[ "$PYTORCH_MODE" == compiled ]]; then
    [[ -d "$PYTORCH_PREFIX" ]] || die "找不到 PyTorch 安装目录：$PYTORCH_PREFIX。请先运行 2-install-pytorch.sh。"
  fi
}

source_runtime_envs() {
  # shellcheck disable=SC1090
  source "$FLAGGEMS_PREFIX/env-flaggems.sh"
  if [[ "${RUNTIME_MODE:-}" == compiled ]]; then
    # shellcheck disable=SC1090
    source "$PYTORCH_PREFIX/env-pytorch.sh"
  fi
}

python_has_cuda_torch() {
  local candidate_python=$1

  [[ -x "$candidate_python" ]] || return 1
  "$candidate_python" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.version.cuda else 1)
PY
}

compiled_python_has_cuda_torch() {
  local compiled_env="$PYTORCH_PREFIX/env-pytorch.sh"
  local compiled_python="$PYTORCH_PREFIX/python/bin/python"

  [[ -f "$compiled_env" && -x "$compiled_python" ]] || return 1
  (
    # shellcheck disable=SC1090
    source "$compiled_env"
    python_has_cuda_torch "$compiled_python"
  )
}

newest_flagtree_wheel() {
  find "$FLAGTREE_PREFIX/wheels" -maxdepth 1 -type f -name 'flagtree-*.whl' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

select_runtime() {
  local flagtree_python="$FLAGTREE_PREFIX/python/bin/python"
  local compiled_env="$PYTORCH_PREFIX/env-pytorch.sh"
  local compiled_python="$PYTORCH_PREFIX/python/bin/python"

  case "$PYTORCH_MODE" in
    wheel)
      RUNTIME_MODE=wheel
      RUNTIME_PYTHON=$flagtree_python
      ;;
    compiled)
      [[ -f "$compiled_env" ]] || \
        die "compiled 模式需要 PyTorch 环境脚本：$compiled_env。请先运行 2-install-pytorch.sh，或改用 --pytorch-mode wheel。"
      [[ -x "$compiled_python" ]] || \
        die "compiled 模式需要可执行 Python：$compiled_python。请先运行 2-install-pytorch.sh，或改用 --pytorch-mode wheel。"
      if ! compiled_python_has_cuda_torch; then
        die "compiled 模式需要 $compiled_python 能导入 CUDA PyTorch。请确认编译安装成功，或改用 --pytorch-mode wheel。"
      fi
      RUNTIME_MODE=compiled
      RUNTIME_PYTHON=$compiled_python
      ;;
    auto)
      if compiled_python_has_cuda_torch; then
        RUNTIME_MODE=compiled
        RUNTIME_PYTHON=$compiled_python
      else
        RUNTIME_MODE=wheel
        RUNTIME_PYTHON=$flagtree_python
      fi
      ;;
  esac
}

install_compiled_runtime_packages() {
  local flagtree_wheel flaggems_source

  [[ "$RUNTIME_MODE" == compiled ]] || return 0
  source_runtime_envs

  flagtree_wheel=$(newest_flagtree_wheel)
  [[ -n "$flagtree_wheel" ]] || die "找不到 FlagTree wheel：$FLAGTREE_PREFIX/wheels/flagtree-*.whl。"
  "$RUNTIME_PYTHON" -m pip install --force-reinstall --no-deps "$flagtree_wheel"

  flaggems_source=${FLAGGEMS_SOURCE:-}
  [[ -n "$flaggems_source" && -d "$flaggems_source" ]] || \
    die "env-flaggems.sh 未导出有效的 FLAGGEMS_SOURCE：${flaggems_source:-空}。"
  PIP_CACHE_DIR="$PREFIX/pip-cache" "$RUNTIME_PYTHON" -m pip install --upgrade \
    'setuptools>=64,<77' 'setuptools-scm>=8,<10' 'wheel==0.45.0'
  "$RUNTIME_PYTHON" -m pip install --no-build-isolation --no-deps "$flaggems_source"
}

ensure_wheel_torch() {
  [[ "$RUNTIME_MODE" == wheel ]] || return 0
  source_runtime_envs

  if ! python_has_cuda_torch "$RUNTIME_PYTHON"; then
    PIP_CACHE_DIR="$PREFIX/pip-cache" "$RUNTIME_PYTHON" -m pip install \
      --index-url https://download.pytorch.org/whl/cu128 \
      'torch==2.7.1+cu128'
  fi
}

missing_model_requirements() {
  "$RUNTIME_PYTHON" - <<'PY'
from importlib import metadata

from packaging.requirements import Requirement

requirements = [
    "transformers>=4.43,<5",
    "huggingface_hub>=0.23,<1",
    "accelerate>=0.33,<2",
    "safetensors>=0.4,<1",
    "sentencepiece>=0.2,<1",
]

missing = []
for raw in requirements:
    requirement = Requirement(raw)
    try:
        version = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        missing.append(raw)
        continue
    if version not in requirement.specifier:
        missing.append(raw)

print("\n".join(missing))
raise SystemExit(1 if missing else 0)
PY
}

ensure_model_dependencies() {
  local requirements status

  set +e
  requirements=$(missing_model_requirements)
  status=$?
  set -e

  [[ $status -eq 0 ]] && return 0
  [[ -n "$requirements" ]] || die '模型推理依赖检查失败，但未返回缺失依赖。'

  mapfile -t MODEL_REQUIREMENTS_TO_INSTALL <<<"$requirements"
  PIP_CACHE_DIR="$PREFIX/pip-cache" "$RUNTIME_PYTHON" -m pip install "${MODEL_REQUIREMENTS_TO_INSTALL[@]}"

  set +e
  requirements=$(missing_model_requirements)
  status=$?
  set -e

  [[ $status -eq 0 ]] || die "模型推理依赖安装后仍不满足：$requirements"
}

write_model_inference_env() {
  local env_file="$PREFIX/env-model-inference.sh"
  local flaggems_env_shell pytorch_env_shell prefix_shell source_shell runtime_python_shell

  mkdir -p \
    "$PREFIX" \
    "$PREFIX/cache/huggingface" \
    "$PREFIX/cache/huggingface/hub" \
    "$PREFIX/cache/transformers" \
    "$PREFIX/pip-cache" \
    "$PREFIX/artifacts/triton-dumps"

  printf -v flaggems_env_shell '%q' "$FLAGGEMS_PREFIX/env-flaggems.sh"
  printf -v pytorch_env_shell '%q' "$PYTORCH_PREFIX/env-pytorch.sh"
  printf -v prefix_shell '%q' "$PREFIX"
  printf -v source_shell '%q' "$SOURCE_DIR"
  printf -v runtime_python_shell '%q' "$RUNTIME_PYTHON"

  cat >"$env_file" <<EOF
# Source this file to use the FlagOS model inference runtime.
if [[ "\${BASH_SOURCE[0]}" == "\$0" ]]; then
  echo "env-model-inference.sh must be sourced, not executed." >&2
  exit 1
fi

# shellcheck disable=SC1091
source $flaggems_env_shell
EOF

  if [[ "$RUNTIME_MODE" == compiled ]]; then
    cat >>"$env_file" <<EOF
# shellcheck disable=SC1091
source $pytorch_env_shell
EOF
  fi

  cat >>"$env_file" <<EOF

export MODEL_INFERENCE_PREFIX=$prefix_shell
export MODEL_INFERENCE_ROOT=$source_shell
export MODEL_INFERENCE_PYTHON=$runtime_python_shell
export HF_HOME="\$MODEL_INFERENCE_PREFIX/cache/huggingface"
export HF_HUB_CACHE="\$MODEL_INFERENCE_PREFIX/cache/huggingface/hub"
export TRANSFORMERS_CACHE="\$MODEL_INFERENCE_PREFIX/cache/transformers"
export PIP_CACHE_DIR="\$MODEL_INFERENCE_PREFIX/pip-cache"
export MODEL_INFERENCE_ARTIFACTS="\$MODEL_INFERENCE_PREFIX/artifacts"
export TRITON_DUMP_DIR="\$MODEL_INFERENCE_PREFIX/artifacts/triton-dumps"
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1

mkdir -p \\
  "\$HF_HOME" \\
  "\$HF_HUB_CACHE" \\
  "\$TRANSFORMERS_CACHE" \\
  "\$PIP_CACHE_DIR" \\
  "\$MODEL_INFERENCE_ARTIFACTS" \\
  "\$TRITON_DUMP_DIR"
EOF
}

PREFIX=$DEFAULT_PREFIX
SOURCE_DIR=$DEFAULT_SOURCE_DIR
FLAGTREE_PREFIX=$DEFAULT_FLAGTREE_PREFIX
FLAGGEMS_PREFIX=$DEFAULT_FLAGGEMS_PREFIX
PYTORCH_PREFIX=$DEFAULT_PYTORCH_PREFIX
MODEL_ID=$DEFAULT_MODEL_ID
REVISION=
MODEL_PATH=
LOCAL_DIR=
PROMPT=$DEFAULT_PROMPT
MAX_NEW_TOKENS=$DEFAULT_MAX_NEW_TOKENS

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
    --flaggems-prefix)
      [[ $# -ge 2 ]] || die '--flaggems-prefix 缺少目录参数。'
      FLAGGEMS_PREFIX=$2
      shift 2
      ;;
    --pytorch-prefix)
      [[ $# -ge 2 ]] || die '--pytorch-prefix 缺少目录参数。'
      PYTORCH_PREFIX=$2
      shift 2
      ;;
    --pytorch-mode)
      [[ $# -ge 2 ]] || die '--pytorch-mode 缺少模式参数。'
      PYTORCH_MODE=$2
      shift 2
      ;;
    --model-id)
      [[ $# -ge 2 ]] || die '--model-id 缺少参数。'
      MODEL_ID=$2
      shift 2
      ;;
    --revision)
      [[ $# -ge 2 ]] || die '--revision 缺少参数。'
      REVISION=$2
      shift 2
      ;;
    --model-path)
      [[ $# -ge 2 ]] || die '--model-path 缺少目录参数。'
      MODEL_PATH=$2
      shift 2
      ;;
    --local-dir)
      [[ $# -ge 2 ]] || die '--local-dir 缺少目录参数。'
      LOCAL_DIR=$2
      shift 2
      ;;
    --prompt)
      [[ $# -ge 2 ]] || die '--prompt 缺少文本参数。'
      PROMPT=$2
      shift 2
      ;;
    --max-new-tokens)
      [[ $# -ge 2 && $2 =~ ^[1-9][0-9]*$ ]] || die '--max-new-tokens 必须是正整数。'
      MAX_NEW_TOKENS=$2
      shift 2
      ;;
    --skip-test)
      RUN_TEST=0
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --skip-inference)
      SKIP_INFERENCE=1
      shift
      ;;
    --compare-baseline)
      COMPARE_BASELINE=1
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
[[ -n "$FLAGGEMS_PREFIX" ]] || die '--flaggems-prefix 不能为空。'
[[ -n "$PYTORCH_PREFIX" ]] || die '--pytorch-prefix 不能为空。'
[[ -n "$MODEL_ID" ]] || die '--model-id 不能为空。'
[[ -n "$PROMPT" ]] || die '--prompt 不能为空。'
[[ "$PYTORCH_MODE" == auto || "$PYTORCH_MODE" == compiled || "$PYTORCH_MODE" == wheel ]] || \
  die '--pytorch-mode 必须是 auto、compiled 或 wheel。'

reject_newline_path '--prefix' "$PREFIX"
reject_newline_path '--source-dir' "$SOURCE_DIR"
reject_newline_path '--flagtree-prefix' "$FLAGTREE_PREFIX"
reject_newline_path '--flaggems-prefix' "$FLAGGEMS_PREFIX"
reject_newline_path '--pytorch-prefix' "$PYTORCH_PREFIX"
[[ -z "$MODEL_PATH" ]] || reject_newline_path '--model-path' "$MODEL_PATH"
[[ -z "$LOCAL_DIR" ]] || reject_newline_path '--local-dir' "$LOCAL_DIR"

PREFIX=$(canonicalize_path "$PREFIX")
SOURCE_DIR=$(canonicalize_path "$SOURCE_DIR")
FLAGTREE_PREFIX=$(canonicalize_path "$FLAGTREE_PREFIX")
FLAGGEMS_PREFIX=$(canonicalize_path "$FLAGGEMS_PREFIX")
PYTORCH_PREFIX=$(canonicalize_path "$PYTORCH_PREFIX")
[[ -z "$MODEL_PATH" ]] || MODEL_PATH=$(canonicalize_path "$MODEL_PATH")
[[ -z "$LOCAL_DIR" ]] || LOCAL_DIR=$(canonicalize_path "$LOCAL_DIR")

reject_root_prefix '--prefix' "$PREFIX"
reject_root_prefix '--flagtree-prefix' "$FLAGTREE_PREFIX"
reject_root_prefix '--flaggems-prefix' "$FLAGGEMS_PREFIX"
reject_root_prefix '--pytorch-prefix' "$PYTORCH_PREFIX"
[[ -z "$LOCAL_DIR" ]] || reject_root_prefix '--local-dir' "$LOCAL_DIR"

require_user_owned_path '--prefix' "$PREFIX"
require_user_owned_path '--flagtree-prefix' "$FLAGTREE_PREFIX"
require_user_owned_path '--flaggems-prefix' "$FLAGGEMS_PREFIX"
require_user_owned_path '--pytorch-prefix' "$PYTORCH_PREFIX"
[[ -z "$LOCAL_DIR" ]] || require_user_owned_path '--local-dir' "$LOCAL_DIR"

check_platform
validate_prerequisites
mkdir -p "$PREFIX/pip-cache"
select_runtime
install_compiled_runtime_packages
ensure_wheel_torch
ensure_model_dependencies
write_model_inference_env

note '模型推理安装器前置条件已通过。'
printf '安装目录：%s\n' "$PREFIX"
printf '源码目录：%s\n' "$SOURCE_DIR"
printf 'FlagTree：%s\n' "$FLAGTREE_PREFIX"
printf 'FlagGems：%s\n' "$FLAGGEMS_PREFIX"
printf 'PyTorch 模式：%s\n' "$PYTORCH_MODE"
printf '运行时模式：%s\n' "$RUNTIME_MODE"
printf '运行时 Python：%s\n' "$RUNTIME_PYTHON"
printf 'PyTorch 目录：%s\n' "$PYTORCH_PREFIX"
printf '模型 ID：%s\n' "$MODEL_ID"
[[ -z "$REVISION" ]] || printf '模型 revision：%s\n' "$REVISION"
[[ -z "$MODEL_PATH" ]] || printf '本地模型目录：%s\n' "$MODEL_PATH"
[[ -z "$LOCAL_DIR" ]] || printf '模型下载目录：%s\n' "$LOCAL_DIR"
printf '最大生成 token：%s\n' "$MAX_NEW_TOKENS"
printf '跳过下载：%s\n' "$SKIP_DOWNLOAD"
printf '跳过推理：%s\n' "$SKIP_INFERENCE"
printf '对比 baseline：%s\n' "$COMPARE_BASELINE"
printf '环境文件：%s\n' "$PREFIX/env-model-inference.sh"
note '当前任务未实现模型下载、stack preflight 或推理运行。'
