"""让 `import flag_gems` 在没有 GPU 硬件的机器上也能成功。

安装脚本的验证段用 `exec(open(...).read())` 载入本文件，必须在 `import flag_gems`
**之前**执行。

## 为什么需要

FlagGems 和 Triton 在 **import 期**就会去问"当前是什么设备"，无 GPU 时两处都会抛：

1. `flag_gems/utils/triton_driver_helper.py` 读 `triton.runtime.driver.active`，
   而 nvidia backend 的 `is_active()` 返回 `torch.cuda.is_available()`，无卡时
   Triton 抛 `RuntimeError: 0 active drivers`。
2. `flag_gems/runtime/backend/device_finder.py` 的 `DeviceDetector` 探测不到任何
   厂商设备，抛 `RuntimeError: No device were detected on your machine !`。

两处都只是"问一下设备是什么"，没有一处真要跑 kernel——算子编译只做
TTIR → pim mlir，全程在 CPU 上。所以注入一个只回答设备探测的 driver 就够了。

## 为什么不继承 CudaDriver

`CudaDriver.__init__` 第一行是 `CudaUtils()`，它会编译一个 C 扩展并链接
`libcuda.so.1`。纯 CPU 机器上那个库不存在，构造直接 assert 失败：

    AssertionError: libcuda.so cannot found!

这一点用 `CUDA_VISIBLE_DEVICES=""` 测不出来（那只让 `cuda.is_available()` 返回
False，驱动和 libcuda 仍在），必须在真正没有 GPU 的环境里才会暴露。

## 为什么 GEMS_VENDOR 选 nvidia

`arm` vendor 的 `device_name` 恰好是 `"cpu"`，看起来更贴切，但它的数学函数 shim 缺
`asin` 等符号，`import flag_gems` 会在 `ops/arcsin.py` 挂掉。`nvidia` vendor 走
libdevice（纯符号表，编译期用），不会真的调用到 GPU。
"""

import os

import torch

if not torch.cuda.is_available():
    from triton.backends.compiler import GPUTarget
    from triton.backends.driver import DriverBase
    from triton.runtime import driver as _driver_config

    # 编译期会被问到的设备属性。取值只影响 Triton 前端的共享内存上限校验，不进入
    # pim mlir——PIM 的分块由 `-pim-tile-to-budget` 按 WRAM 预算独立决定。按 sm80
    # 的真实参数填，避免前端因为"共享内存为 0"拒绝合法分块。
    _SM80_PROPERTIES = {
        "max_shared_mem": 166912,
        "multiprocessor_count": 108,
        "sm_clock_rate": 1410000,
        "mem_clock_rate": 1215000,
        "mem_bus_width": 5120,
    }

    class _CpuHostUtils:
        """回答编译期的设备属性查询，不加载 CUDA 运行时。"""

        @staticmethod
        def get_device_properties(device=None):
            return dict(_SM80_PROPERTIES)

        @staticmethod
        def load_binary(*args, **kwargs):
            raise RuntimeError(
                "纯 CPU 环境不能加载 GPU kernel 二进制。算子编译只需要 "
                "TTIR → pim mlir，不需要这一步。"
            )

    class _CpuHostDriver(DriverBase):
        """无 GPU 机器上的编译期 driver：只回答设备探测。

        这份成员清单是从 Triton 和 FlagGems 源码里逐个 grep 出来的，缺一个就会在
        import 期或编译期抛 AttributeError。
        """

        def __init__(self):
            self.utils = _CpuHostUtils()
            self._target = GPUTarget("cuda", 80, 32)

        @staticmethod
        def is_active():
            # 只通过 set_active 显式注入，不参与自动探测——否则有卡机器上会变成
            # "两个 active driver"。
            return False

        def get_current_target(self):
            return self._target

        def get_active_torch_device(self):
            return torch.device("cpu")

        def get_current_device(self):
            return 0

        def set_current_device(self, device):
            return None

        def get_current_stream(self, device=None):
            return 0

        def get_device_capability(self, device=None):
            return (8, 0)

        def get_device_interface(self):
            return torch.cpu

        def map_python_to_cpp_type(self, ty):
            # DriverBase 的抽象方法，生成 launcher 的 C 签名时才用。纯编译路径不
            # 生成 launcher，沿用 nvidia backend 的映射即可（纯字符串表）。
            from triton.backends.nvidia.driver import ty_to_cpp

            return ty_to_cpp(ty)

        @property
        def current_arch_id(self):
            # FlagGems 用它索引 autotune 配置，纯编译路径下取值无实质影响。
            return 80

        @property
        def launcher_cls(self):
            raise RuntimeError(
                "纯 CPU 环境没有 kernel launcher。走到这里说明有人试图在无 GPU 的"
                "机器上启动 Triton kernel。"
            )

        def get_benchmarker(self):
            raise RuntimeError(
                "纯 CPU 环境不能 benchmark GPU kernel：autotune 需要真实执行。"
            )

    _driver_config.set_active(_CpuHostDriver())
    os.environ.setdefault("GEMS_VENDOR", "nvidia")
    print("cpu-host-driver: 已注入编译期 driver（无 GPU 硬件，算子编译不需要）")
