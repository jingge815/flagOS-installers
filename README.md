# flagOS-installers

本文档用于安装 flagOS 相关组件。当前已经实现：

- `0-install-flagtree.sh`：安装 FlagTree/Triton 及其用户态依赖
- `1-install-flaggems.sh`：在 FlagTree 环境之上安装并验证 FlagGems
- `2-install-pytorch.sh`：安装官方 PyTorch wheel，并同步 FlagTree 的 PIM Triton
- `3-install-model-inference.sh`：下载或复用 Hugging Face 模型，并运行
  FlagGems 大模型推理

## 环境前提

- Ubuntu 22.04 或 24.04，x86_64
- 可以访问公网下载源码、Python、LLVM、Triton/NVIDIA 编译依赖、PyTorch wheel 和 Python package wheel

两个系统版本都在纯 CPU 容器里跑通过完整流程（Ubuntu 24.04.5 实测）。脚本按
`/etc/os-release` 的 `ID`/`VERSION_ID` 判断，非 Ubuntu 或非 x86_64 会直接报错退出。

### GPU 是可选的

四个脚本都会自行探测 `nvidia-smi`，两种环境都支持：

| 环境 | 行为 |
| --- | --- |
| 有 NVIDIA GPU（驱动 570+） | 装 CUDA 版 torch，行为与之前完全一致 |
| 纯 CPU，无 GPU 硬件 | 装 CPU 版 torch，跳过十几个 `nvidia-*`/`cuda-*` 包（数 GB） |

想强制某一种，**装 torch 的那两个脚本**（`2-install-pytorch.sh`、
`3-install-model-inference.sh`）接受这两个开关：

```bash
bash 2-install-pytorch.sh --torch-cpu    # 强制 CPU 版
bash 2-install-pytorch.sh --torch-cuda   # 强制 CUDA 版
```

脚本 0 和 1 没有这两个开关——它们只是把"没有 nvidia-smi 就退出"改成了可选检测。

需要说明的是脚本 0 确实会装一个 torch：它给自己那份独立 Python 装
`torch==2.7.1+cu128`，供安装后验证（`matmul_sm80.py`）和后续 FlagGems 的 smoke test
使用，是 FlagTree 环境自己的运行时 torch。**这一份目前不区分有没有 GPU**，纯 CPU 机器上
同样会拉十几个 `nvidia-*`/`cuda-*` 包（约 2.5 GB）。它们只是占下载流量和磁盘，不影响
纯 CPU 流程正确性——验证在无卡时会自动退化为"确认 PIM pass 存在"，不执行 kernel。
要让它在无卡时改装 `torch==2.7.1+cpu`（省掉这 2.5 GB），改
`0-install-flagtree.sh` 里 `install_python()` 那一行即可；经确认 FlagTree 的构建过程
不依赖 torch，只有安装后验证和 FlagGems 用到它。

脚本 2 装的 `torch==2.9.1+cpu`/`+cu128` 是另一份，装在独立的 PyTorch 前缀下，与这份无关。

纯 CPU 环境下能做什么、不能做什么，见
`flagos-pim-compiler/docs/pim-compiler-v0.0.4.md`。简单说：**算子编译（TTIR → pim
mlir）、7B 推理、NumPy 对拍、GeneSim 仿真全都可用**，产出的 pim mlir 与有 GPU 时逐字节
相同；不可用的只有"需要真实执行 GPU kernel"的那类步骤。

### 先装这些系统包（需要 root，只此一步）

纯净的 Ubuntu 缺 `git`、`make`、`cc`、`c++`、`ar`、`ld`、`curl`、`python3`，
必须先补上：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl tar gzip python3
```

装完之后四个脚本本身**不需要 root**。这条命令在纯净 `ubuntu:22.04` 和 `ubuntu:24.04`
容器里都逐个验证过，覆盖四个脚本 `require_command` 的完整并集：`git`、`tar`、`gzip`、
`dpkg-deb`、`apt-get`、`awk`、`sed`、`find`、`make`、`cc`、`c++`、`ar`、`ld`、
`curl` 或 `wget`、`python3`。

注意 `python3` 只有脚本 3 需要（脚本 0/1/2 用的是自带的独立 Python），漏装会在第三步
报 `缺少命令 python3`。脚本 3 只做存在性检查、不调用它，但后面下载 HF 模型要用。

### 24.04 上的 `python3` 和 pip

两个版本的 `python3` 不是同一个：22.04 是 3.10，24.04 是 3.12。四个安装脚本都用自带的
独立 Python（3.10.20），只有上面这条系统包检查会碰到系统 `python3`，所以版本差异不影响
安装。

但 **24.04 的系统 Python 默认没有 pip，而且带 PEP 668 的 `EXTERNALLY-MANAGED` 标记**，
直接往系统 Python 装包会被拒绝：

```text
error: externally-managed-environment
```

所以下面「大模型推理」一节里安装 `hf` 命令行时，24.04 上不能用
`python3 -m pip install`，要改用独立 venv（22.04 用同样写法也没问题）：

```bash
sudo apt-get install -y python3-venv
python3 -m venv ~/.hf-cli
~/.hf-cli/bin/pip install -U "huggingface_hub[cli]"
~/.hf-cli/bin/hf --help
```

后续用到 `hf` 的地方，把命令换成 `~/.hf-cli/bin/hf` 即可。

### 网络不稳时建议设置

安装过程要从 GitHub 和 PyPI 拉几个 GB，跨境网络下默认超时太短：

```bash
export UV_HTTP_TIMEOUT=600      # GeneSim 的 install.sh 用 uv，默认仅 30 秒
```

实测遇到过两类网络失败，重跑同一条命令即可继续（脚本每一步都是幂等的）：

- `GnuTLS recv error (-110)`：clone FLIR 子模块时 TLS 中断
- `Failed to download rich==15.0.0 ... network timeout`：PyPI 下载超时

### 不要直接拷贝已装好的安装目录

`flagOS-installed/` **不能整体打包拷到另一台机器或另一个路径**。pip 生成的 32 个
命令包装脚本（`cmake`、`ninja`、`lit` …）把解释器绝对路径写进了 shebang：

```
$ head -1 flagTree/python-3.10.20/bin/cmake
#!/media/disk/.../flagOS-installed/flagTree/python-3.10.20/bin/python
```

换路径后这些命令全部失效，FlagTree 编译会报
`RuntimeError: CMake must be installed to build the following extensions: triton`。

正确做法是在目标机器上跑一遍安装脚本——脚本各步骤都有幂等判断，已存在的 LLVM、
Python、下载缓存都会自动跳过，不会重复下载。

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
独立 Python，从 PyTorch 官方 wheel 索引安装 torch，再把 FlagTree 中带 PIM pass 的
Triton 装进该环境。它不会下载 PyTorch 源码、不编译 PyTorch，也不安装 CUDA Toolkit。

- 默认安装目录是 `../flagOS-installed/pytorch`
- 默认 FlagTree 目录是 `../flagOS-installed/flagTree`
- 安装目录中会放置独立 Python、pip cache、PyTorch 与 CUDA Python 运行库、原 Triton 备份和环境脚本
- 有 GPU 时：`torch==2.9.1+cu128`，需要 570+ NVIDIA 驱动
- 纯 CPU 时：`torch==2.9.1+cpu`，不需要驱动，也不需要 `nvidia-smi`

### 两种 torch 的 Triton 来源不一样

| wheel | triton 从哪来 |
| --- | --- |
| `2.9.1+cu128`（CUDA 版） | **自带**上游 `triton==3.3.1`。脚本把 FlagTree 的 PIM 文件覆盖进这份 triton，并把被覆盖的 `libtriton.so`、`compiler.py` 备份到 `<prefix>/.triton-backup-pre-pim/` |
| `2.9.1+cpu`（CPU 版） | **不带** triton（它的依赖只有 filelock、fsspec、jinja2、networkx、sympy、typing-extensions）。脚本把 FlagTree 那份 triton 连同 `triton-*.dist-info` 整体装进 PyTorch 的 site-packages |

CPU 版这条分支不是优化，是必需：CPU wheel 里根本没有可覆盖的 triton 目录，老写法会直接
报 `PyTorch wheel 未安装 Triton`。两条分支都保留 `backends/nvidia/{bin,include,lib/cupti}`
——图编译器要用其中的 `cuda.h` 和 `ptxas`，它们是 pip 包里的**文件**，不需要驱动。

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

后续重复执行时，脚本会复用已经存在的独立 Python 和 pip cache，用
`torch.__version__` 与目标版本比对，一致就跳过下载（CPU wheel 有 184 MB，CUDA wheel
约 2.5 GB，网络不稳时这一步值得跳过），并再次把 FlagTree 的 PIM Triton 装进该环境。

CUDA 版覆盖前会把 PyTorch 的 `libtriton.so` 和 NVIDIA compiler 文件备份到
`<prefix>/.triton-backup-pre-pim/`（每跑一次产生一份带时间戳的备份，脚本不会自动清理）。
CPU 版是整体安装，没有可备份的原文件，不会产生备份目录。

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
   `huggingface-cli`）。**24.04 必须用独立 venv**——系统 Python 没有 pip 且带
   PEP 668 保护，直接 `python3 -m pip install` 会报
   `error: externally-managed-environment`；22.04 用同样写法也可以：

   ```bash
   sudo apt-get install -y python3-venv
   python3 -m venv ~/.hf-cli
   ~/.hf-cli/bin/pip install -U "huggingface_hub[cli]"
   ~/.hf-cli/bin/hf --help
   ```

4. 通过代理手动下载 HF 格式的 `Llama-2-7b-hf` 检查点；把 `xxx` 替换为上一步
   创建的 token：

   ```bash
   http_proxy=http://127.0.0.1:7500 \
   https_proxy=http://127.0.0.1:7500 \
   all_proxy=http://127.0.0.1:7500 \
   ~/.hf-cli/bin/hf download meta-llama/Llama-2-7b-hf \
     --local-dir ../flagOS-installed/model-inference/models/Llama-2-7b-hf \
     --token xxx
   ```

下载完成后，用
`--model-path ../flagOS-installed/model-inference/models/Llama-2-7b-hf`
让安装脚本复用这个目录（就是上面 `--local-dir` 指定的位置）。换成别的目录也行，
只要和 `--local-dir` 保持一致。

推荐执行顺序：

```bash
bash 0-install-flagtree.sh
bash 1-install-flaggems.sh
# 可选：需要验证源码编译 PyTorch 时再执行
bash 2-install-pytorch.sh
# 完成上面的 Hugging Face 授权和手动下载后，复用本地 HF 格式模型
bash 3-install-model-inference.sh \
  --model-path ../flagOS-installed/model-inference/models/Llama-2-7b-hf
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

## 5. 图编译器与 GeneSim（在四个脚本之后）

四个脚本只负责 FlagOS 软件栈。图编译器和模拟器是两个独立仓库，**顺序不能颠倒**：
GeneSim 的 `install.sh` 会顺带补上图编译器测试要用的 `scipy`、`datasets`，所以先装它。

```bash
cd /path                       # 与 flagOS-installers 同级即可

cd /path/genesim
./install.sh --skip-attacc

# 后：图编译器
cd /path
git clone https://github.com/jingge815/flagos-pim-compiler.git
```

`install.sh` 会自行探测 GPU：无 GPU 时装 CPU 版 torch，跳过十几个 `nvidia-*`/`cuda-*`
包（数 GB）。也可以用 `--torch-cpu` / `--torch-cuda` 强制。

### 配置图编译器的站点路径

图编译器用仓库根目录的 `paths.json` 记录站点路径。**六个键都要填对**。

其中后两个是 **GML 参考产物**——芯方舟底层编译器的标准输出样例，由甲方提供，
**四个安装脚本都不会产出它们**，需要单独放到某个目录再把路径指过去。需要的文件：

| 键 | 目录内容 | 体积 |
| --- | --- | --- |
| `gml_reference_dir` | `relay2gml_graph.gml` + `runtime_files/` | 约 68 MB |
| `gml_llama2_reference_dir` | `llama2_w4a8_decode_block_0/parser_output/` | 约 246 MB |

实测这约 314 MB 是跑通图编译器全部快速回归所需的最小集合。不配的后果有两层，都要避免：

- 一组结构校验测试直接报 `未配置站点路径 gml_llama2_reference_dir`
- `export_gml.py` **静默跳过** dtype 覆盖检查与 `--orchestrate` 的文件族检查——
  输出照样显示"验证全部通过"，但实际少做了两项

```json
{
  "pytorch_env_script": "/path/flagOS-installed/pytorch/env-pytorch.sh",
  "llama2_7b_model_dir": "/path/flagOS-installed/model-inference/models/Llama-2-7b-hf",
  "flagtree_prefix": "/path/flagOS-installed/flagTree",
  "genesim_root": "/path/genesim",
  "gml_reference_dir": "/path/gml-reference",
  "gml_llama2_reference_dir": "/path/gml-reference/llama2_w4a8_decode_block_0/parser_output"
}
```

```json
{
  "pytorch_env_script": "/path/flagOS-installed/pytorch/env-pytorch.sh",
  "llama2_7b_model_dir": "/path/flagOS-installed/model-inference/models/Llama-2-7b-hf",
  "flagtree_prefix": "/path/flagOS-installed/flagTree",
  "genesim_root": "/path/genesim",
  "gml_reference_dir": "/path/gml-reference",
  "gml_llama2_reference_dir": "/path/gml-reference/llama2_w4a8_decode_block_0/parser_output"
}
```

也可以不改文件，用同名环境变量覆盖（`PYTORCH_ENV_SCRIPT`、`LLAMA2_7B_MODEL_DIR`、
`FLAGTREE_PREFIX`、`GENESIM_ROOT`、`GML_REFERENCE_DIR`、`GML_LLAMA2_REFERENCE_DIR`），
环境变量优先级更高。确认解析结果：

```bash
cd /path/flagos-pim-compiler
source /path/flagOS-installed/pytorch/env-pytorch.sh
python -c 'from genesim_bridge.paths import describe; print(describe())'
```

### 验证清单

上面每一步都在纯 CPU 的 Ubuntu 24.04 容器里实测过，预期结果如下。**任何一条对不上都
说明前面的安装有问题**，先回头查对应脚本的输出，不要直接往下走。

| # | 命令 | 预期 |
| --- | --- | --- |
| 1 | `python -m pytest tests/ -q -k "not llama2_7b"` | `714 passed, 42 deselected`（约 2 分钟） |
| 2 | `python -m pytest tests/ -q -k "llama2_7b"` | `42 passed`（纯 CPU 约 34 分钟，含真实编译 224 个 GEMM） |
| 3 | `python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_out` | 尾部 `验证全部通过（3 项）`，其中「GML 引用集 == 落盘集」会给出文件数 |
| 4 | `cd /path/genesim && ./run.sh --test sim` | `All simulator tests passed (37/37 test files)` |
| 5 | `cd /path/genesim && ./run.sh --test config_loader` | `Ran 38 tests ... OK` + `[SUCCESS]` |
| 6 | `cd /path/genesim && ./run.sh --test model_ir` | `Ran 49 tests ... OK` + `[SUCCESS]` |

第 1、2、3 条都在图编译器目录下执行，并且要先 `source` PyTorch 的 `env-pytorch.sh`。
第 2 条最慢但最关键——它加载真实 Llama-2-7B 权重，把整条编译链路跑一遍。

第 1 条如果出现成片的 `ERROR ... RuntimeError: 未配置站点路径 gml_*`，就是上面的
`paths.json` 没配全，不是代码问题。

### 可选：GNN 性能预测器

`tests/predictor/` 那一组测试需要额外的可选依赖，默认不装。不装的话跑全量测试会看到 9 个
失败，报错会指明原因：

```
ModuleNotFoundError: No module named 'torch_geometric'
ImportError: the GAT predictor backbone requires torch-geometric;
  install predictor dependencies with: ./install.sh --predictor
```

需要这一组测试时执行：

```bash
./install.sh --predictor        # 装 torch-geometric==2.8.0.post1
./run.sh --test predictor
```

性能预测器是独立特性，**不参与 PIM 编译链路**——不装它，图编译、算子编译、仿真闭环
全部照常工作。

### 纯 CPU 环境的能力边界

| 能力 | 纯 CPU |
| --- | --- |
| 算子编译（TTIR → pim mlir） | **可用**，产物与有 GPU 时逐字节相同 |
| 7B 推理、NumPy 对拍 | **可用** |
| GeneSim 仿真（含 TP/PP 切分评估） | **可用** |
| 执行 FlagGems 的 Triton kernel | 不可用（需要 GPU 硬件） |
| `genesim/scripts/refine_ir_with_flagtree.py` | 不可用，改走 `export_pp_placement.py` |

详见 `flagos-pim-compiler/docs/pim-compiler-v0.0.4.md`。
