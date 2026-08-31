# flagOS-installers

本文档用于安装 flagOS 相关组件。当前已经实现：

- `0-install-flagtree.sh`：安装 FlagTree/Triton 及其用户态依赖
- `1-install-flaggems.sh`：在 FlagTree 环境之上安装并验证 FlagGems
- `2-install-pytorch.sh`：安装官方 PyTorch wheel，并同步 FlagTree 的 PIM Triton
- `3-install-model-inference.sh`：下载或复用 Hugging Face 模型，并运行
  FlagGems 大模型推理

## 环境前提

- Ubuntu 22.04 x86_64
- 已安装并可使用 NVIDIA 驱动
- `nvidia-smi` 可以正常运行
- 可以访问公网下载源码、Python、LLVM、Triton/NVIDIA 编译依赖、PyTorch wheel 和 Python package wheel
- 不需要 root 权限。FlagTree 安装仍需要 `git`、`tar`、`gzip`、`dpkg-deb`、`apt-get`、`make`、`cc`、`c++`、`ar`、`ld`、`curl` 或 `wget`；PyTorch wheel 安装仅需要 `tar`、`gzip`、`awk` 和 `curl` 或 `wget`

本脚本不会安装 NVIDIA 驱动；目标机器必须提前满足 `nvidia-smi` 可用。

## 1. 安装 FlagTree

安装脚本：`0-install-flagtree.sh`

- 源码默认下载到 `./FlagTree`
- 默认安装目录是 `../flagOS-installed/flagTree`
- 代码固定到已验证可编译的提交 `317f15a426466633c4f37f164b2c58ae9c31bd03`
- FLIR 源码会下载到 `./FlagTree/third_party/flir`，并固定到提交 `165f387b28e3fdbd03542e7ae9881db902facd16`
- 安装目录中会放置独立 Python、LLVM、Triton/NVIDIA 编译工具、Ubuntu deb sysroot、wheel、缓存、环境脚本和验证示例

一键安装：

```bash
bash 0-install-flagtree.sh
```

指定安装目录：

```bash
bash 0-install-flagtree.sh --prefix /path/to/flagTree
```

指定源码目录和编译并行度：

```bash
bash 0-install-flagtree.sh --source-dir /path/to/FlagTree --max-jobs 16
```

跳过安装后验证：

```bash
bash 0-install-flagtree.sh --skip-test
```

### 重复执行行为

第一次执行时，默认假设 `../flagOS-installed/flagTree` 为空。

后续重复执行时，脚本会复用已经存在且校验通过的内容：

- 已下载且可解压的压缩包
- 已安装的独立 Python
- LLVM 目录
- Triton/NVIDIA 编译工具包
- 已解包的 `zlib`、`libxml2` 头文件和库
- 干净的 `./FlagTree` Git 源码目录

构建目录会重新生成，以减少旧构建缓存导致的不确定性。

### 使用环境

安装完成后，先加载环境脚本：

```bash
source ../flagOS-installed/flagTree/env-flagtree.sh
```

基础导入验证：

```bash
python -c 'import torch, triton; print("imports ok")'
```

运行 GPU matmul 验证示例：

```bash
python ../flagOS-installed/flagTree/examples/matmul_sm80.py
```

验证示例会打印当前 Triton target，并检查 Triton matmul 结果是否接近
`torch.matmul`。

### MLIR / IR dump

安装环境默认设置：

```text
FLAGTREE_IR_DUMP_DIR=../flagOS-installed/flagTree/mlir-dumps
MLIR_ENABLE_DUMP=1
MLIR_DUMP_PATH=../flagOS-installed/flagTree/mlir-dumps/flagtree-mlir-dump.mlir
TRITON_KERNEL_DUMP=1
TRITON_ALWAYS_COMPILE=1
TRITON_DUMP_DIR=../flagOS-installed/flagTree/triton-stage-dumps
```

注意：`MLIR_DUMP_PATH` 必须是具体文件路径，不能是目录。安装环境脚本会把
误设成目录的旧值自动改成目录下的 `flagtree-mlir-dump.mlir` 文件。

运行验证示例后，可以查看两类 dump：

```text
../flagOS-installed/flagTree/mlir-dumps/flagtree-mlir-dump.mlir
../flagOS-installed/flagTree/triton-stage-dumps
```

其中 `mlir-dumps/flagtree-mlir-dump.mlir` 是 MLIR PassManager 的完整串行
pass dump；`triton-stage-dumps` 会按 Triton 编译阶段输出多个文件，例如
`.ttir`、`.ttgir`、`.llir`、`.ptx`、`.cubin`、`.sass` 等。脚本默认设置
`TRITON_ALWAYS_COMPILE=1`，避免命中 cache 时不重新生成阶段 dump。

如果使用了自定义安装目录，请把路径替换为：

```text
<prefix>/mlir-dumps
<prefix>/triton-stage-dumps
```

## 2. 安装 FlagGems

安装脚本：`1-install-flaggems.sh`

该脚本需要在 `0-install-flagtree.sh` 成功执行后运行。它不需要 root，
会复用 `../flagOS-installed/flagTree/env-flagtree.sh` 中的 Python、
PyTorch、FlagTree/Triton、LLVM 和 NVIDIA 用户态编译工具。

- 源码默认下载到 `./FlagTree/FlagGems`
- 默认安装目录是 `../flagOS-installed/flagGems`
- 默认复用的 FlagTree 安装目录是 `../flagOS-installed/flagTree`
- 代码固定到 `https://github.com/flagos-ai/FlagGems` 的提交 `bfbd21ca85dbfa84061fe90a7ced899c85238b13`
- 默认以 editable Python package 方式安装 FlagGems，并运行 CUDA smoke test
- 默认生成 `../flagOS-installed/flagGems/env-flaggems.sh`

一键安装并验证：

```bash
bash 1-install-flaggems.sh
```

只安装不运行 CUDA smoke test：

```bash
bash 1-install-flaggems.sh --skip-test
```

指定安装目录、源码目录或 FlagTree 前缀：

```bash
bash 1-install-flaggems.sh \
  --prefix /path/to/flagGems \
  --source-dir /path/to/FlagTree/FlagGems \
  --flagtree-prefix /path/to/flagTree
```

删除已有 FlagGems 源码后重新 clone：

```bash
bash 1-install-flaggems.sh --force-reclone
```

额外尝试构建 CUDA C++ wrapped operators：

```bash
bash 1-install-flaggems.sh --with-cpp
```

默认安装不构建 C++ wrapped operators；普通 FlagGems 算子会在 smoke test
中通过 Triton JIT 编译 CUDA kernel。

### 重复执行行为

后续重复执行时，脚本会复用已经存在且干净的 `./FlagTree/FlagGems`
Git 源码目录，并重新 checkout 到固定 commit。Python 依赖和 FlagGems
package 会由 pip 按需复用或重新安装。

如果源码目录存在已跟踪修改或暂存修改，脚本会停止，避免覆盖本地改动。

### 使用环境

安装完成后，先加载环境脚本：

```bash
source ../flagOS-installed/flagGems/env-flaggems.sh
```

基础验证：

```bash
python - <<'PY'
import torch
import flag_gems

x = torch.randn((128, 128), device="cuda")
y = torch.randn((128, 128), device="cuda")
with flag_gems.use_gems():
    z = torch.add(x, y)
torch.cuda.synchronize()
print("max error:", (z - (x + y)).abs().max().item())
PY
```

`max error` 应为 `0.0`。

### MLIR / IR dump

FlagGems 环境脚本默认设置：

```text
FLAGGEMS_IR_DUMP_DIR=../flagOS-installed/flagGems/mlir-dumps
MLIR_ENABLE_DUMP=1
MLIR_DUMP_PATH=../flagOS-installed/flagGems/mlir-dumps/flaggems-mlir-dump.mlir
TRITON_KERNEL_DUMP=1
TRITON_ALWAYS_COMPILE=1
TRITON_DUMP_DIR=../flagOS-installed/flagGems/triton-stage-dumps
```

运行安装脚本的 smoke test 时，会打印 Triton/MLIR pass dump，例如
`LowerLoops`、`ExpandLoops`，并打印 dump 文件路径。运行后可以查看：

```text
../flagOS-installed/flagGems/mlir-dumps/flaggems-mlir-dump.mlir
../flagOS-installed/flagGems/triton-stage-dumps
```

如果使用了自定义安装目录，请把路径替换为：

```text
<prefix>/mlir-dumps
<prefix>/triton-stage-dumps
```

## 3. 安装 PyTorch

安装脚本：`2-install-pytorch.sh`

该脚本需要在 `0-install-flagtree.sh` 成功执行后运行。它会在无 root 权限下准备
独立 Python，从 PyTorch 官方 CUDA 12.8 wheel 索引安装
`torch==2.9.1+cu128`，再把 FlagTree 中带 PIM pass 的 Triton 同步进该环境。
它不会下载 PyTorch 源码、不编译 PyTorch，也不安装 CUDA Toolkit。

- 默认安装目录是 `../flagOS-installed/pytorch`
- 默认 FlagTree 目录是 `../flagOS-installed/flagTree`
- 安装目录中会放置独立 Python、pip cache、PyTorch 与 CUDA Python 运行库、原 Triton 备份和环境脚本
- 目标机器仍需 Ubuntu 22.04 x86_64、可用的 `nvidia-smi` 和 570+ NVIDIA 驱动

一键安装并验证：

```bash
bash 2-install-pytorch.sh
```

指定安装目录和 FlagTree 安装目录：

```bash
bash 2-install-pytorch.sh \
  --prefix /path/to/pytorch \
  --flagtree-prefix /path/to/flagTree
```

跳过安装后 CUDA smoke test：

```bash
bash 2-install-pytorch.sh --skip-test
```

### 重复执行行为

后续重复执行时，脚本会复用已经存在的独立 Python 和 pip cache，重新确保
`torch==2.9.1+cu128` 已安装，并再次从 FlagTree 同步 PIM Triton。覆盖前的
PyTorch `libtriton.so` 和 NVIDIA compiler 文件会备份到
`<prefix>/.triton-backup-pre-pim/`。

### 使用环境

安装完成后，先加载环境脚本：

```bash
source ../flagOS-installed/pytorch/env-pytorch.sh
```

基础验证：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
x = torch.randn((128, 128), device="cuda")
y = torch.randn((128, 128), device="cuda")
z = x @ y
torch.cuda.synchronize()
print("device:", torch.cuda.get_device_name(0))
print("shape:", tuple(z.shape))
PY
```

确认 PIM Triton pass 已同步：

```bash
python -c 'from triton._C.libtriton import passes; print(hasattr(passes, "pim"))'
```

## 4. 大模型推理

安装脚本：`3-install-model-inference.sh`

该脚本需要在 `0-install-flagtree.sh` 和 `1-install-flaggems.sh` 成功执行后运行。
如果已经运行 `2-install-pytorch.sh` 且
`../flagOS-installed/pytorch/env-pytorch.sh` 中的 Python 能导入 CUDA PyTorch，
推理脚本会在 `auto` 模式下优先使用该 PyTorch 环境；否则使用 FlagTree
Python 并按需下载 CUDA PyTorch wheel。

### 手动下载默认 Llama 2 7B HF 模型（必需）

默认模型是 Hugging Face 的 **HF 格式**检查点
[`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf)，
不是 GGUF 或其他 Llama 2 变体。该模型是 gated model；**授权和下载必须手动
完成**，安装脚本不会代为注册、申请授权或绕过访问限制。

1. 打开上面的模型页面，用邮箱注册或登录 Hugging Face，按页面提示申请并接受
   Llama 2 的访问条款。只有页面显示 `You have been granted access to this model`
   后，才可以下载。
2. 在 Hugging Face 的 [Access Tokens 页面](https://huggingface.co/settings/tokens)
   创建可读取模型的 token。
3. 安装 Hugging Face 的 **`hf`** 命令（后续命令使用 `hf`，不是旧的
   `huggingface-cli`）：

   ```bash
   python3 -m pip install -U "huggingface_hub[cli]"
   hf --help
   ```

4. 通过代理手动下载 HF 格式的 `Llama-2-7b-hf` 检查点；把 `xxx` 替换为上一步
   创建的 token：

   ```bash
   http_proxy=http://127.0.0.1:7500 \
   https_proxy=http://127.0.0.1:7500 \
   all_proxy=http://127.0.0.1:7500 \
   hf download meta-llama/Llama-2-7b-hf \
     --local-dir ../flagOS-installed/model-inference/models/Llama-2-7b-hf \
     --token xxx
   ```

下载完成后，使用
`--model-path ~/.llama/checkpoints/Llama-2-7b-hf` 让安装脚本复用
这个本地 HF 格式模型目录。

推荐执行顺序：

```bash
bash 0-install-flagtree.sh
bash 1-install-flaggems.sh
# 可选：需要验证源码编译 PyTorch 时再执行
bash 2-install-pytorch.sh
# 完成上面的 Hugging Face 授权和手动下载后，复用本地 HF 格式模型
bash 3-install-model-inference.sh \
  --model-path /home/fengjingge/.llama/checkpoints/Llama-2-7b-hf
```

全流程不需要 root 权限。脚本会复用已有安装目录和 pip/Hugging Face cache，
缺少模型推理依赖时才用所选 Python 自动安装。默认模型为 Hugging Face 官方
`Llama-2-7b-hf`。请先按上节手动完成 Hugging Face 授权和下载；模型目录完整时会
复用，否则脚本会尝试下载。下载或访问失败时不会回退到 GPT-2、TinyLlama 或随机
权重。

- 本地源码快照：`./model-inference`
- 默认安装目录：`../flagOS-installed/model-inference`
- 默认模型：`Llama-2-7b-hf`
- 默认模型目录：`../flagOS-installed/model-inference/models/Llama-2-7b-hf`
- 环境脚本：`../flagOS-installed/model-inference/env-model-inference.sh`
- 推理日志：`../flagOS-installed/model-inference/logs/inference-YYYYMMDD_HHMMSS.log`
- Triton dump：`../flagOS-installed/model-inference/artifacts/triton-dumps/<timestamp>/`

下载或复用默认模型并运行推理：

```bash
bash 3-install-model-inference.sh
```

脚本会验证推理日志包含 `inference_status: ok`、正整数
`flaggems_generated_tokens` 和非空 `flaggems_text`，通过时会打印日志路径和
Triton dump 目录。

只下载或复用默认模型，不运行推理：

```bash
bash 3-install-model-inference.sh --skip-inference
bash 3-install-model-inference.sh --skip-download --skip-inference
```

强制使用 FlagTree Python 和 PyTorch wheel，不使用 `2-install-pytorch.sh` 的独立 PyTorch 环境：

```bash
bash 3-install-model-inference.sh --pytorch-mode wheel
```

复用已经下载好的本地模型目录：

```bash
bash 3-install-model-inference.sh \
  --model-path ../flagOS-installed/model-inference/models/Llama-2-7b-hf \
  --prompt "Explain FlagGems in one sentence." \
  --max-new-tokens 32
```

指定自定义安装前缀：

```bash
bash 3-install-model-inference.sh \
  --prefix /path/to/model-inference-prefix \
  --flagtree-prefix /path/to/flagTree \
  --flaggems-prefix /path/to/flagGems \
  --pytorch-prefix /path/to/pytorch
```

安装完成后加载推理环境：

```bash
source ../flagOS-installed/model-inference/env-model-inference.sh
```

`model-inference/` 目录只提交必要的推理入口和说明文件。大模型权重通常较大，
默认下载到安装前缀下的 `models/`，不要提交到 Git；只有很小的测试模型资产才
可以放入 `model-inference/models/` 并随仓库提交。
