# PyTorch Wheel Installer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在无需 root、源码或本地 CUDA Toolkit 的前提下安装 `torch==2.9.1+cu128`，并在 PyTorch wheel 安装后同步 FlagTree 的 PIM Triton。

**Architecture:** `2-install-pytorch.sh` 保留独立 Python 前缀和 CUDA smoke test，但以官方 cu128 wheel 替代所有源码、编译和 Toolkit 步骤。Triton 同步函数迁移到该脚本，并将 FlagTree 作为明确、必需的输入。`0-install-flagtree.sh` 只负责构建 FlagTree；README 与 shell 回归测试共同描述和保护新的调用契约。

**Tech Stack:** Bash、Python 3.10 standalone、pip、PyTorch 官方 cu128 wheel 索引、Bash 回归测试。

## Global Constraints

- 固定安装 `torch==2.9.1+cu128`，索引为 `https://download.pytorch.org/whl/cu128`。
- 所有安装文件必须位于用户传入的 `--prefix` 下；不使用 `sudo`、apt 或系统写入。
- 目标平台为 Ubuntu 22.04 x86_64，要求可用的 `nvidia-smi` 与 570+ 驱动。
- Triton 必须从 `--flagtree-prefix` 中同步；FlagTree 或所需文件缺失必须失败，不能静默降级。
- `0-install-flagtree.sh` 不得再包含 PyTorch 前缀、Triton 同步或对应 CLI 选项。

---

### Task 1: 为新 CLI 和职责边界建立回归测试

**Files:**
- Create: `tests/test-installers.sh`
- Test: `tests/test-installers.sh`

**Interfaces:**
- Consumes: `0-install-flagtree.sh`、`2-install-pytorch.sh` 的 `--help` 输出和源文本。
- Produces: 可由 `bash tests/test-installers.sh` 执行的零网络回归测试。

- [ ] **Step 1: 写出失败的契约测试**

创建测试函数，要求新选项已出现、已删除选项不再出现，并检查 wheel 与同步职责：

```bash
assert_contains "$pytorch_help" '--flagtree-prefix DIR'
assert_not_contains "$pytorch_help" '--source-dir DIR'
assert_contains "$pytorch_script" 'torch==2.9.1+cu128'
assert_contains "$pytorch_script" 'sync_triton_to_pytorch()'
assert_not_contains "$flagtree_help" '--pytorch-prefix DIR'
assert_not_contains "$flagtree_script" 'sync_triton_to_pytorch()'
```

- [ ] **Step 2: 运行测试，确认其因新功能缺失而失败**

Run: `bash tests/test-installers.sh`

Expected: 在现有 `2-install-pytorch.sh --help` 缺少 `--flagtree-prefix DIR` 时失败。

- [ ] **Step 3: 实现最小测试运行器**

测试使用 `set -euo pipefail`、仓库根目录定位、`assert_contains` / `assert_not_contains` 和 `bash -n`。它不得联网、不得执行安装器正文，也不得依赖 GPU。

- [ ] **Step 4: 待脚本实现后运行测试**

Run: `bash tests/test-installers.sh`

Expected: 输出 `installer contract tests passed` 并以 0 退出。

- [ ] **Step 5: 提交测试**

```bash
git add tests/test-installers.sh
git commit -m "test: cover wheel installer contract"
```

### Task 2: 将 FlagTree 安装器恢复为独立职责

**Files:**
- Modify: `0-install-flagtree.sh:29-49,353-395,397-456,490-494`
- Test: `tests/test-installers.sh`

**Interfaces:**
- Consumes: 无。
- Produces: 仅构建和验证 FlagTree 的 `0-install-flagtree.sh`；其 CLI 只包含 `--prefix`、`--source-dir`、`--max-jobs`、`--skip-test`。

- [ ] **Step 1: 先运行现有失败测试**

Run: `bash tests/test-installers.sh`

Expected: Task 1 的测试仍失败，表明新 PyTorch 接口尚未实现。

- [ ] **Step 2: 删除同步接口与实现**

从帮助文本、函数区、参数解析、PyTorch 前缀规范化和主流程中删除下列项目：

```bash
--pytorch-prefix DIR
--skip-pytorch-sync
sync_triton_to_pytorch
PYTORCH_PREFIX=
SKIP_PYTORCH_SYNC=0
```

保留 `build_flagtree`、`install_example` 与 `run_validation` 原有行为。

- [ ] **Step 3: 运行语法和职责测试**

Run: `bash -n 0-install-flagtree.sh && bash tests/test-installers.sh`

Expected: 语法检查通过；若 Task 3 未完成，契约测试仍只因新 PyTorch 接口失败。

- [ ] **Step 4: 提交 FlagTree 职责拆分**

```bash
git add 0-install-flagtree.sh
git commit -m "refactor: move Triton sync out of FlagTree installer"
```

### Task 3: 以官方 wheel 重写 PyTorch 安装器并同步 PIM Triton

**Files:**
- Modify: `2-install-pytorch.sh`
- Test: `tests/test-installers.sh`

**Interfaces:**
- Consumes: `--prefix DIR`、`--flagtree-prefix DIR`、`--skip-test`；`$FLAGTREE_PREFIX/python-3.10.20/lib/python3.10/site-packages/triton`。
- Produces: `$PREFIX/python`、`$PREFIX/env-pytorch.sh`，以及同步过 PIM Triton 的 `$PREFIX/python-3.10.20/lib/python3.10/site-packages/triton`。

- [ ] **Step 1: 运行失败测试**

Run: `bash tests/test-installers.sh`

Expected: 因 `--flagtree-prefix`、wheel 固定版本和同步函数尚不存在而失败。

- [ ] **Step 2: 用 wheel 安装路径取代源码构建路径**

删除源码目录、git clone、submodule、构建、wheel 复用、CUDA Toolkit 与 CUDA Python 库安装函数，以及相应的配置和选项。保留 Python 下载校验与 `canonicalize_existing_or_parent`，新增：

```bash
DEFAULT_FLAGTREE_PREFIX="$SCRIPT_DIR/../flagOS-installed/flagTree"
PYTORCH_VERSION=2.9.1
PYTORCH_WHEEL_VERSION=2.9.1+cu128
PYTORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu128

install_pytorch_wheel() {
  note "通过官方 cu128 wheel 安装 PyTorch $PYTORCH_WHEEL_VERSION"
  "$PYTHON/bin/python" -m pip install --upgrade --force-reinstall \
    --cache-dir "$PIP_CACHE_DIR" --index-url "$PYTORCH_WHEEL_INDEX" \
    "torch==$PYTORCH_WHEEL_VERSION"
}
```

`check_platform` 只检查 tar、gzip、awk、nvidia-smi、curl/wget、平台与驱动版本；不再要求 Git 或编译工具。环境脚本只导出独立 Python、`PYTHONNOUSERSITE`、`PIP_CACHE_DIR`、wheel 的 nvidia 运行库 `LD_LIBRARY_PATH` 与 `CMAKE_PREFIX_PATH`，不导出 `CUDA_HOME`、`CUDA_PATH` 或 `nvcc` 路径。

- [ ] **Step 3: 添加严格的 Triton 同步函数**

定义 `sync_triton_to_pytorch`：计算两个 site-packages Triton 目录；逐一检查 FlagTree 的 `libtriton.so`、`pim_sidecar.py`、`compiler.py`、`bin`、`include` 和 `lib/cupti`，缺失则 `die`；在 `$PREFIX/.triton-backup-pre-pim` 备份 PyTorch 原始 `libtriton.so` 与 `compiler.py`；覆盖文件、替换三个目录并删除 `__pycache__`。主流程必须是：

```bash
check_platform
install_python
write_environment_file
install_pytorch_wheel
sync_triton_to_pytorch
run_validation
```

在 smoke test 尾部加入：

```python
from triton._C.libtriton import passes
if not hasattr(passes, "pim"):
    raise SystemExit("PIM Triton passes are unavailable")
```

- [ ] **Step 4: 运行语法、帮助、契约测试**

Run: `bash -n 2-install-pytorch.sh && bash 2-install-pytorch.sh --help && bash tests/test-installers.sh`

Expected: 帮助显示 `--flagtree-prefix` 和 `--skip-test`，不显示源码构建选项；契约测试通过。

- [ ] **Step 5: 提交 PyTorch 安装器**

```bash
git add 2-install-pytorch.sh tests/test-installers.sh
git commit -m "feat: install PyTorch wheel and sync PIM Triton"
```

### Task 4: 更新用户文档并执行完整静态验证

**Files:**
- Modify: `README.md:5-17,241-291`
- Test: `tests/test-installers.sh`

**Interfaces:**
- Consumes: Task 3 的命令行与环境行为。
- Produces: 准确说明预编译 wheel、FlagTree 同步依赖和用户级安装约束的 README。

- [ ] **Step 1: 更新 README 的组件概览和 PyTorch 章节**

将组件描述改为“下载官方预编译 wheel 并同步 FlagTree 的 PIM Triton”。PyTorch 章节保留仅有的调用示例：

```bash
bash 2-install-pytorch.sh
bash 2-install-pytorch.sh --prefix /path/to/pytorch --flagtree-prefix /path/to/flagTree
bash 2-install-pytorch.sh --skip-test
```

明确固定版本为 `torch==2.9.1+cu128`、要求先成功运行 FlagTree、没有源码目录/编译/CUDA Toolkit、所有文件安装到用户前缀，且重复执行会重新确保版本并重做同步。

- [ ] **Step 2: 运行完整静态验证**

Run: `bash -n 0-install-flagtree.sh 1-install-flaggems.sh 2-install-pytorch.sh 3-install-model-inference.sh && bash tests/test-installers.sh && git diff --check`

Expected: 所有命令以 0 退出，测试打印 `installer contract tests passed`，且没有空白错误。

- [ ] **Step 3: 审阅实际改动**

Run: `git diff --check HEAD~3..HEAD && git status --short`

Expected: 仅包含本任务的脚本、README、测试和计划/规格文档；没有生成的安装产物。

- [ ] **Step 4: 提交文档**

```bash
git add README.md docs/superpowers/plans/2026-08-31-pytorch-wheel-installer.md
git commit -m "docs: document PyTorch wheel installation"
```
