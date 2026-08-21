#!/usr/bin/env bash
# Validate the local model inference installer interface and prerequisites.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEFAULT_PREFIX="$SCRIPT_DIR/../flagOS-installed/model-inference"
DEFAULT_SOURCE_DIR="$SCRIPT_DIR/model-inference"
DEFAULT_FLAGTREE_PREFIX="$SCRIPT_DIR/../flagOS-installed/flagTree"
DEFAULT_FLAGGEMS_PREFIX="$SCRIPT_DIR/../flagOS-installed/flagGems"
DEFAULT_PYTORCH_PREFIX="$SCRIPT_DIR/../flagOS-installed/pytorch"
DEFAULT_MODEL_BACKEND="huggingface"
DEFAULT_MODEL_ID="meta-llama/Llama-2-7b-hf"
DEFAULT_MODEL_DIRNAME="Llama-2-7b-hf"
BUILTIN_MODEL_NAME="builtin-gpt2-random"
DEFAULT_PROMPT="Explain in one sentence what FlagGems does for PyTorch."
DEFAULT_MAX_NEW_TOKENS=32
DEFAULT_MAX_SEQ=128

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
  --model-id ID           使用 Hugging Face 模型 ID（会下载或复用模型）
  --revision REV          Hugging Face 模型 revision
  --model-path DIR        使用已下载的本地 Hugging Face 模型目录
  --local-dir DIR         模型下载目录
  --builtin-model NAME    显式使用 legacy/debug 内置 GPT-2 smoke 模型
  --prompt TEXT           推理提示词，默认：$DEFAULT_PROMPT
  --max-new-tokens N      生成 token 数，默认：$DEFAULT_MAX_NEW_TOKENS
  --max-seq N             内置 GPT-2 最大序列长度，默认：$DEFAULT_MAX_SEQ
  --skip-test             跳过 NVIDIA GPU 可用性检查
  --skip-download         跳过模型下载
  --skip-inference        跳过推理运行
  --compare-baseline      同时运行非 FlagGems baseline
  -h, --help              显示帮助

说明：
  默认使用 Hugging Face 模型 meta-llama/Llama-2-7b-hf。
  本地默认目录为：
    ../flagOS-installed/model-inference/models/Llama-2-7b-hf
  若目录完整则复用；若不存在或不完整则下载。官方 LLaMA2 仓库需要
  HuggingFace 授权，请先设置 HF_TOKEN 或运行 huggingface-cli login。
  --builtin-model builtin-gpt2-random 仅用于显式 legacy/debug smoke，不作为默认回退。
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
  local flaggems_runtime_requirements=(
    'packaging>=26'
    'PyYAML==6.0.1'
    'sqlalchemy==2.0.48'
    'numpy'
  )

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
  PIP_CACHE_DIR="$PREFIX/pip-cache" "$RUNTIME_PYTHON" -m pip install \
    "${flaggems_runtime_requirements[@]}"
  "$RUNTIME_PYTHON" -m pip install --no-build-isolation --no-deps "$flaggems_source"
  "$RUNTIME_PYTHON" - <<'PY'
from importlib import metadata

import flag_gems
import torch
import triton

if not torch.cuda.is_available() or not torch.version.cuda:
    raise SystemExit("compiled runtime does not provide CUDA PyTorch")
flagtree_version = metadata.version("flagtree")
flaggems_version = metadata.version("flag_gems")
print(f"compiled_runtime_imports: torch={torch.__version__} triton={triton.__version__}")
print(f"compiled_runtime_imports: flagtree={flagtree_version} flag_gems={flaggems_version}")
PY
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

model_id_to_dirname() {
  local model_id=$1

  if [[ "$model_id" == "$DEFAULT_MODEL_ID" ]]; then
    printf '%s\n' "$DEFAULT_MODEL_DIRNAME"
    return 0
  fi

  printf '%s\n' "${model_id//\//-}"
}

configure_huggingface_cache() {
  export HF_HOME="${HF_HOME:-$PREFIX/cache/huggingface}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$PREFIX/cache/huggingface/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PREFIX/cache/transformers}"
  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PREFIX/models"
}

require_usable_model_path() {
  local model_path=$1

  [[ -d "$model_path" ]] || die "模型目录不存在：$model_path"
  [[ -f "$model_path/config.json" ]] || die "模型目录缺少 config.json：$model_path"
  if [[ ! -f "$model_path/tokenizer.json" && ! -f "$model_path/tokenizer.model" &&
        ! ( -f "$model_path/vocab.json" && -f "$model_path/merges.txt" ) ]]; then
    die "模型目录缺少 tokenizer 文件：$model_path"
  fi
  if find "$model_path" -type f -name '*.incomplete' -print -quit | grep -q .; then
    die "模型目录存在未完成的 Hugging Face 下载标记：$model_path"
  fi
  "$RUNTIME_PYTHON" - "$model_path" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
weight_patterns = (
    "*.safetensors",
    "pytorch_model*.bin",
    "model*.bin",
)

weights = [
    path
    for pattern in weight_patterns
    for path in model_path.glob(pattern)
    if path.is_file() and path.stat().st_size > 0
]

index_files = [
    model_path / "model.safetensors.index.json",
    model_path / "pytorch_model.bin.index.json",
]
for index_file in index_files:
    if not index_file.exists():
        continue
    data = json.loads(index_file.read_text())
    shard_names = sorted(set(data.get("weight_map", {}).values()))
    missing = [
        name
        for name in shard_names
        if not (model_path / name).is_file() or (model_path / name).stat().st_size == 0
    ]
    if missing:
        raise SystemExit(f"model index references missing or empty shard files: {missing[:5]}")
    if shard_names:
        weights.extend(model_path / name for name in shard_names)

if not weights:
    raise SystemExit(
        "model directory has config.json but no non-empty model weight files "
        "(*.safetensors, pytorch_model*.bin, model*.bin)"
    )
PY
}

model_path_is_usable() {
  local model_path=$1

  (require_usable_model_path "$model_path") >/dev/null 2>&1
}

download_or_reuse_model() {
  local revision_arg=()

  configure_huggingface_cache

  if [[ -n "$MODEL_PATH" ]]; then
    require_usable_model_path "$MODEL_PATH"
    note '使用已有本地模型。'
    return 0
  fi

  if [[ -z "$LOCAL_DIR" ]]; then
    LOCAL_DIR="$PREFIX/models/$(model_id_to_dirname "$MODEL_ID")"
  fi
  MODEL_PATH=$LOCAL_DIR

  if model_path_is_usable "$MODEL_PATH"; then
    note '复用完整的本地模型目录。'
    return 0
  fi

  if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
    require_usable_model_path "$MODEL_PATH"
    note '跳过下载并复用本地模型。'
    return 0
  fi

  mkdir -p "$MODEL_PATH"
  [[ -z "$REVISION" ]] || revision_arg=(--revision "$REVISION")

  note "下载或复用 Hugging Face 模型：$MODEL_ID"
  "$RUNTIME_PYTHON" - "$MODEL_ID" "$MODEL_PATH" "${revision_arg[@]}" <<'PY'
import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("repo_id")
parser.add_argument("local_dir")
parser.add_argument("--revision")
args = parser.parse_args()

snapshot_download(
    repo_id=args.repo_id,
    revision=args.revision,
    local_dir=args.local_dir,
)
PY

  require_usable_model_path "$MODEL_PATH"
}

run_stack_preflight() {
  [[ "$RUN_TEST" -eq 1 ]] || return 0

  note '运行 CUDA / Triton / FlagGems preflight。'
  "$RUNTIME_PYTHON" - <<'PY'
import flag_gems
import torch
import triton

print(f"torch: {torch.__version__} ({torch.__file__})")
print(f"triton_import: {triton.__version__} ({triton.__file__})")
print(f"flag_gems: {getattr(flag_gems, '__version__', 'unknown')} ({flag_gems.__file__})")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
print(f"gpu: {torch.cuda.get_device_name(0)}")
with flag_gems.use_gems():
    x = torch.arange(4, device="cuda", dtype=torch.float32)
    y = x + 1
torch.cuda.synchronize()
if y.detach().cpu().tolist() != [1.0, 2.0, 3.0, 4.0]:
    raise SystemExit("FlagGems CUDA tensor operation returned an unexpected result")
print("flag_gems_preflight: ok")
PY
}

validate_generated_text() {
  local log_file=$1

  awk '/^inference_status: ok /{found=1; exit} END{exit !found}' "$log_file" || \
    die "推理日志未包含 inference_status: ok：$log_file"
  awk '/^flaggems_generated_tokens:[[:space:]]+[1-9][0-9]*$/{found=1; exit} END{exit !found}' "$log_file" || \
    die "推理日志未包含有效的 flaggems_generated_tokens：$log_file"
  awk '/flaggems_text:/{seen=1; next} seen && NF {found=1; exit} END{exit !found}' "$log_file" || \
    die "推理日志未包含 flaggems_text: 后的非空生成文本：$log_file"
}

run_inference() {
  local timestamp log_dir log_file triton_dump_dir
  local inference_args=()

  [[ "$SKIP_INFERENCE" -eq 0 ]] || return 0

  if [[ "$MODEL_BACKEND" == huggingface ]]; then
    require_usable_model_path "$MODEL_PATH"
  fi
  timestamp=$(date +%Y%m%d_%H%M%S)
  log_dir="$PREFIX/logs"
  log_file="$log_dir/inference-$timestamp.log"
  triton_dump_dir="$PREFIX/artifacts/triton-dumps/$timestamp"
  mkdir -p "$log_dir" "$triton_dump_dir"

  export TRITON_ALWAYS_COMPILE=1
  export TRITON_KERNEL_DUMP=1
  export TRITON_DUMP_DIR=$triton_dump_dir

  inference_args=(
    "$RUNTIME_PYTHON" "$SOURCE_DIR/examples/run_llm_with_flaggems.py"
    --prompt "$PROMPT"
    --max-new-tokens "$MAX_NEW_TOKENS"
  )
  if [[ "$MODEL_BACKEND" == builtin-gpt2 ]]; then
    inference_args+=(
      --builtin-model "$BUILTIN_MODEL_NAME"
      --max-seq "$MAX_SEQ"
    )
  else
    inference_args+=(--model-path "$MODEL_PATH")
  fi
  if [[ "$COMPARE_BASELINE" -eq 1 ]]; then
    inference_args+=(--compare-baseline)
  fi

  note '运行 FlagGems 模型推理。'
  if ! "${inference_args[@]}" 2>&1 | tee "$log_file"; then
    die "推理运行失败，日志：$log_file"
  fi

  validate_generated_text "$log_file"
  printf 'Triton artifact directory: %s\n' "$triton_dump_dir"
  printf 'Inference log: %s\n' "$log_file"
}

print_runtime_versions() {
  "$RUNTIME_PYTHON" - <<'PY'
import sys
from importlib import metadata

import flag_gems
import torch

print("运行时版本确认：")
print("python:", sys.executable)
print("torch:", torch.__version__, torch.__file__)
print("flag_gems:", metadata.version("flag_gems"), flag_gems.__file__)
print("flagtree:", metadata.version("flagtree"))
PY
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
MODEL_BACKEND=$DEFAULT_MODEL_BACKEND
MODEL_ID=$DEFAULT_MODEL_ID
REVISION=
MODEL_PATH=
LOCAL_DIR=
PROMPT=$DEFAULT_PROMPT
MAX_NEW_TOKENS=$DEFAULT_MAX_NEW_TOKENS
MAX_SEQ=$DEFAULT_MAX_SEQ

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
      MODEL_BACKEND=huggingface
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
      MODEL_BACKEND=huggingface
      shift 2
      ;;
    --local-dir)
      [[ $# -ge 2 ]] || die '--local-dir 缺少目录参数。'
      LOCAL_DIR=$2
      shift 2
      ;;
    --builtin-model)
      [[ $# -ge 2 ]] || die '--builtin-model 缺少模型名。'
      [[ "$2" == "$BUILTIN_MODEL_NAME" ]] || die "--builtin-model 仅支持 $BUILTIN_MODEL_NAME。"
      MODEL_BACKEND=builtin-gpt2
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
    --max-seq)
      [[ $# -ge 2 && $2 =~ ^[1-9][0-9]*$ ]] || die '--max-seq 必须是正整数。'
      MAX_SEQ=$2
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
if [[ "$MODEL_BACKEND" == huggingface && -z "$MODEL_PATH" ]]; then
  [[ -n "$MODEL_ID" ]] || die 'Hugging Face 模式需要 --model-id 或 --model-path。'
fi
if [[ "$MODEL_BACKEND" == builtin-gpt2 ]]; then
  [[ -z "$REVISION" ]] || die '--revision 仅适用于 Hugging Face 模型，请同时指定 --model-id。'
  [[ -z "$LOCAL_DIR" ]] || die '--local-dir 仅适用于 Hugging Face 模型，请同时指定 --model-id。'
fi
[[ -n "$PROMPT" ]] || die '--prompt 不能为空。'
[[ "$PYTORCH_MODE" == auto || "$PYTORCH_MODE" == compiled || "$PYTORCH_MODE" == wheel ]] || \
  die '--pytorch-mode 必须是 auto、compiled 或 wheel。'
[[ "$MODEL_BACKEND" == builtin-gpt2 || "$MODEL_BACKEND" == huggingface ]] || \
  die '内部错误：未知模型后端。'

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
if [[ "$MODEL_BACKEND" == huggingface ]]; then
  download_or_reuse_model
else
  configure_huggingface_cache
fi
run_stack_preflight
run_inference

print_runtime_versions

note '模型推理安装器前置条件已通过。'
printf '安装目录：%s\n' "$PREFIX"
printf '源码目录：%s\n' "$SOURCE_DIR"
printf 'FlagTree：%s\n' "$FLAGTREE_PREFIX"
printf 'FlagGems：%s\n' "$FLAGGEMS_PREFIX"
printf 'PyTorch 模式：%s\n' "$PYTORCH_MODE"
printf '运行时模式：%s\n' "$RUNTIME_MODE"
printf '运行时 Python：%s\n' "$RUNTIME_PYTHON"
printf 'PyTorch 目录：%s\n' "$PYTORCH_PREFIX"
printf '模型后端：%s\n' "$MODEL_BACKEND"
if [[ "$MODEL_BACKEND" == builtin-gpt2 ]]; then
  printf '内置模型：%s\n' "$BUILTIN_MODEL_NAME"
  printf '内置 GPT-2 配置：n_layer=4 n_head=8 n_embd=512 max_seq=%s\n' "$MAX_SEQ"
else
  printf '模型 ID：%s\n' "$MODEL_ID"
fi
[[ -z "$REVISION" ]] || printf '模型 revision：%s\n' "$REVISION"
[[ -z "$MODEL_PATH" ]] || printf '本地模型目录：%s\n' "$MODEL_PATH"
[[ -z "$LOCAL_DIR" ]] || printf '模型下载目录：%s\n' "$LOCAL_DIR"
printf '最大生成 token：%s\n' "$MAX_NEW_TOKENS"
printf '最大序列长度：%s\n' "$MAX_SEQ"
printf '跳过下载：%s\n' "$SKIP_DOWNLOAD"
printf '跳过推理：%s\n' "$SKIP_INFERENCE"
printf '对比 baseline：%s\n' "$COMPARE_BASELINE"
printf '环境文件：%s\n' "$PREFIX/env-model-inference.sh"
