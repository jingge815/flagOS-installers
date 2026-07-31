# flagOS-installers

本文档用于安装 flagOS 相关组件。当前已经实现：

- `0-install-flagtree.sh`：安装 FlagTree/Triton 及其用户态依赖
- `1-install-flaggems.sh`：在 FlagTree 环境之上安装并验证 FlagGems
- `2-install-pytorch.sh`：从固定提交源码编译并安装 PyTorch
- `3-install-model-inference.sh`：下载或复用 Hugging Face 模型，并运行
  FlagGems 大模型推理

## 环境前提

- Ubuntu 22.04 x86_64
- 已安装并可使用 NVIDIA 驱动
- `nvidia-smi` 可以正常运行
- 可以访问公网下载源码、Python、LLVM、Triton/NVIDIA 编译依赖、PyTorch wheel 和 Python package wheel
- 不需要 root 权限，但系统需要预装基础命令：`git`、`tar`、`gzip`、`dpkg-deb`、`apt-get`、`make`、`cc`、`c++`、`ar`、`ld`、`curl` 或 `wget`

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

该脚本可在 `0-install-flagtree.sh` 和 `1-install-flaggems.sh` 执行完成后运行，
但不依赖它们的 Python 或环境脚本。它会在无 root 权限下自动准备独立 Python、
CUDA 12.8 Toolkit、CUDA Python libraries、PyTorch Python 构建依赖，下载
PyTorch 源码并编译安装 wheel。

- 源码默认下载到 `./pytorch`
- 默认安装目录是 `../flagOS-installed/pytorch`
- 代码从 `https://github.com/pytorch/pytorch.git` 下载，并固定到提交 `d38164a545b4a4e4e0cf73ce67173f70574890b6`，对应 `v2.9.1`
- 安装目录中会放置独立 Python、CUDA Toolkit、Python CUDA libraries、wheel、缓存、日志和环境脚本

一键安装并验证：

```bash
bash 2-install-pytorch.sh
```

指定安装目录、源码目录和编译并行度：

```bash
bash 2-install-pytorch.sh \
  --prefix /path/to/pytorch \
  --source-dir /path/to/pytorch-source \
  --max-jobs 16
```

只准备源码和依赖，不执行源码编译：

```bash
bash 2-install-pytorch.sh --skip-build
```

跳过安装后 CUDA smoke test：

```bash
bash 2-install-pytorch.sh --skip-test
```

删除已有 PyTorch 源码后重新 clone：

```bash
bash 2-install-pytorch.sh --force-reclone
```

### 重复执行行为

后续重复执行时，脚本会复用已经存在且校验通过的下载包、独立 Python、
CUDA Toolkit、pip cache、已生成 wheel 和干净的 `./pytorch` Git 源码目录。
如果源码目录存在已跟踪修改或暂存修改，脚本会停止，避免覆盖本地改动。

默认不会删除 PyTorch 源码目录中的 `build/`；如果怀疑旧构建缓存影响结果，
可以加 `--clean-build`。

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

## 4. 大模型推理

安装脚本：`3-install-model-inference.sh`

该脚本需要在 `0-install-flagtree.sh` 和 `1-install-flaggems.sh` 成功执行后
运行。`2-install-pytorch.sh` 是可选步骤：如果已经运行并且
`../flagOS-installed/pytorch/env-pytorch.sh` 中的 Python 能导入 CUDA PyTorch，
推理脚本会在 `auto` 模式下优先使用该编译版 PyTorch；否则使用 FlagTree
Python 并按需下载 CUDA PyTorch wheel。

推荐执行顺序：

```bash
bash 0-install-flagtree.sh
bash 1-install-flaggems.sh
# 可选：需要验证源码编译 PyTorch 时再执行
bash 2-install-pytorch.sh
bash 3-install-model-inference.sh
```

全流程不需要 root 权限。脚本会复用已有安装目录和 pip/Hugging Face cache，
缺少模型推理依赖时才用所选 Python 自动安装。

- 本地源码快照：`./model-inference`
- 默认安装目录：`../flagOS-installed/model-inference`
- 默认模型：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- 默认模型目录：`../flagOS-installed/model-inference/models/TinyLlama-TinyLlama-1.1B-Chat-v1.0`
- 环境脚本：`../flagOS-installed/model-inference/env-model-inference.sh`
- 推理日志：`../flagOS-installed/model-inference/logs/inference-YYYYMMDD_HHMMSS.log`
- Triton dump：`../flagOS-installed/model-inference/artifacts/triton-dumps/<timestamp>/`

一键下载或复用默认模型并运行推理：

```bash
bash 3-install-model-inference.sh
```

显式指定 Hugging Face 模型：

```bash
bash 3-install-model-inference.sh --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

只安装依赖和下载模型，不运行推理：

```bash
bash 3-install-model-inference.sh --skip-inference
```

强制使用 FlagTree Python 和 PyTorch wheel，不使用源码编译版 PyTorch：

```bash
bash 3-install-model-inference.sh --pytorch-mode wheel
```

复用已经下载好的本地模型目录：

```bash
bash 3-install-model-inference.sh \
  --model-path /path/to/TinyLlama-TinyLlama-1.1B-Chat-v1.0 \
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
