#!/usr/bin/env python3
"""Minimal FP16 Triton matmul validation for NVIDIA SM80-class GPUs."""

import os
import pathlib


def configure_ir_dump() -> pathlib.Path | None:
    dump_dir_text = os.environ.get("FLAGTREE_IR_DUMP_DIR", "")
    if not dump_dir_text:
        return None

    dump_dir = pathlib.Path(dump_dir_text)
    dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLIR_ENABLE_DUMP", "1")
    dump_path = pathlib.Path(
        os.environ.get("MLIR_DUMP_PATH", str(dump_dir / "flagtree-mlir-dump.mlir"))
    )
    if dump_path.is_dir():
        dump_path = dump_path / "flagtree-mlir-dump.mlir"
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MLIR_DUMP_PATH"] = str(dump_path)

    stage_dump_dir = pathlib.Path(
        os.environ.get("TRITON_DUMP_DIR", str(dump_dir.parent / "triton-stage-dumps"))
    )
    stage_dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_KERNEL_DUMP", "1")
    os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
    os.environ["TRITON_DUMP_DIR"] = str(stage_dump_dir)
    return dump_dir


_DUMP_DIR = configure_ir_dump()

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(n, block_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    a_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn

    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for tile_k in range(0, tl.cdiv(k, block_k)):
        remaining_k = k - tile_k * block_k
        a = tl.load(
            a_ptrs,
            mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < remaining_k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offsets_k[:, None] < remaining_k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        accumulator.to(tl.float16),
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply contiguous CUDA FP16 matrices with a Triton kernel."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected compatible two-dimensional matrices")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("expected matrices on the same CUDA device")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("expected FP16 matrices")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("expected contiguous matrices")

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(m, 64) * triton.cdiv(n, 128),)
    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        block_m=64,
        block_n=128,
        block_k=32,
        num_warps=4,
        num_stages=4,
    )
    return c


def main() -> None:
    torch.manual_seed(0)
    m, n, k = 513, 769, 1025
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    result = matmul(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.matmul(a, b), atol=5e-2, rtol=1e-2)
    print("target:", triton.runtime.driver.active.get_current_target())
    print("matmul matches torch")
    if _DUMP_DIR:
        print("mlir pass dump:", os.environ["MLIR_DUMP_PATH"])
        print("triton stage dumps:", os.environ["TRITON_DUMP_DIR"])
    else:
        print("mlir dump dir: not enabled")


if __name__ == "__main__":
    main()
