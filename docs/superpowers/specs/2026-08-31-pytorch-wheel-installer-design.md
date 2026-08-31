# PyTorch 预编译 wheel 安装与 Triton 同步设计

## 目标

将 `2-install-pytorch.sh` 从源码编译安装改为无 root 权限的预编译 wheel
安装器，并在 PyTorch 安装完成后把 FlagTree 中包含 PIM pass 的 Triton 同步到
该环境。这样执行顺序保持为 FlagTree、FlagGems、PyTorch，且不会因为 PyTorch
目录尚不存在而跳过同步。

## 安装架构

`2-install-pytorch.sh` 默认在 `../flagOS-installed/pytorch` 创建独立的
Python 3.10.20 环境。它从 PyTorch 官方 CUDA 12.8 wheel 索引安装固定版本
`torch==2.9.1+cu128`，以及由该 wheel 声明的 Python CUDA 运行时依赖。

脚本不再下载 PyTorch 源码，不编译 wheel，也不下载或安装 CUDA Toolkit。因此
不需要 root、编译器、`nvcc` 或 Git；运行机器仍必须是 Ubuntu 22.04 x86_64，且
已有可用的 NVIDIA 驱动和 `nvidia-smi`。CUDA 12.8 要求驱动主版本至少为 570。

生成的 `env-pytorch.sh` 继续公开 `PYTORCH_PREFIX`，并将其独立 Python 和
wheel 的 `nvidia` 运行库目录加入运行环境。由于不提供 Toolkit，脚本不导出
伪造的 `CUDA_HOME` 或 `nvcc` 路径。

## Triton 同步

PyTorch wheel 安装完成后，脚本从 `--flagtree-prefix` 指定的 FlagTree 环境读取
Triton，并覆盖 PyTorch 环境中对应的 Triton 文件：`libtriton.so`、PIM sidecar、
NVIDIA compiler 模块以及其随附的 `bin`、`include`、`lib/cupti`。

默认 FlagTree 前缀为 `../flagOS-installed/flagTree`。同步前必须确认两侧 Triton
目录存在，以及 FlagTree 侧所需文件均存在；任一条件不满足时安装以可操作的
报错退出，避免得到看似成功但没有 PIM 能力的 PyTorch 环境。覆盖前将现有的
PyTorch `libtriton.so` 和 `compiler.py` 备份到 PyTorch 前缀内
`.triton-backup-pre-pim/`，随后删除无效的 Python 字节码缓存。

`0-install-flagtree.sh` 不再了解或修改 PyTorch 前缀：移除 `--pytorch-prefix`、
`--skip-pytorch-sync`、同步函数和相关变量。该职责完全由安装顺序靠后的
`2-install-pytorch.sh` 承担。

## 命令行与重复执行

`2-install-pytorch.sh` 保留：

- `--prefix DIR`
- `--flagtree-prefix DIR`
- `--skip-test`
- `--help`

移除仅适用于源码构建的 `--source-dir`、`--max-jobs`、`--skip-build`、
`--clean-build` 与 `--force-reclone`。

重复执行会复用可执行的独立 Python 和 pip cache，并用固定的预编译 wheel
重新确保 PyTorch 版本正确、重新完成 Triton 同步。安装目标均在用户指定且可写
的前缀内；脚本不使用 `sudo` 或系统包管理器。

## 验证和文档

完成安装后，CUDA smoke test 将检查 `torch.__version__`、CUDA 可用性和 GPU
矩阵乘；同步验证将从 `triton._C.libtriton` 导入 `passes` 并确认 `pim` 属性。
`pip check` 用于检查 Python 依赖一致性。

README 将更新脚本职责、依赖条件、选项、重复执行行为和环境说明，移除所有有关
PyTorch 源码、CUDA Toolkit 和本地编译的表述。
