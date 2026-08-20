import torch
import triton
import triton.language as tl
from torch import Tensor

from model.registry import Component, register


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, stride_row, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # -inf on masked lanes so they lose the max and contribute exp(-inf)=0.
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask,
                other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    numerator = tl.where(mask, numerator, 0.0)
    tl.store(out_ptr + row * stride_row + cols,
             numerator / tl.sum(numerator, axis=0), mask=mask)


@register(Component.SOFTMAX, "triton")
def softmax(x: Tensor) -> Tensor:
    x_flat = x.contiguous().reshape(-1, x.shape[-1])
    rows, n_cols = x_flat.shape
    out = torch.empty_like(x_flat)
    block = triton.next_power_of_2(n_cols)
    _softmax_kernel[(rows,)](x_flat, out, x_flat.stride(0), n_cols,
                             BLOCK=block, num_warps=4)
    return out.reshape(x.shape)


@triton.autotune(
    configs=[triton.Config({"ROWS": r}, num_warps=4) for r in (1, 2, 4, 8, 16)],
    key=["n_cols"],
)
@triton.jit
def _softmax_tuned_kernel(x_ptr, out_ptr, stride_row, n_rows, n_cols,
                          BLOCK: tl.constexpr, ROWS: tl.constexpr):
    # Committed kernel gives each program one 64-wide row: at BLOCK=64,
    # num_warps=4 that's 128 threads / 64 elems = 0.5 elem/thread, well
    # below the >=2 elem/thread a memory-bound kernel needs to hide
    # latency. Batching ROWS rows per program raises elem/thread to
    # ROWS * 64 / 128 without changing num_warps, at the cost of 2D
    # masking on both axes (row count may not divide evenly, e.g. at
    # batch=1 with ROWS=8 the block still covers 8 row-slots but only 3
    # are real: rows*heads=3).
    row_start = tl.program_id(0) * ROWS
    rows = row_start + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    row_mask = rows < n_rows
    col_mask = cols < n_cols
    mask = row_mask[:, None] & col_mask[None, :]

    ptrs = x_ptr + rows[:, None] * stride_row + cols[None, :]
    x = tl.load(ptrs, mask=mask, other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, axis=1)[:, None]
    numerator = tl.exp(x)
    numerator = tl.where(mask, numerator, 0.0)
    denom = tl.sum(numerator, axis=1)[:, None]
    tl.store(out_ptr + rows[:, None] * stride_row + cols[None, :],
             numerator / denom, mask=mask)


@register(Component.SOFTMAX, "triton_tuned")
def softmax_tuned(x: Tensor) -> Tensor:
    x_flat = x.contiguous().reshape(-1, x.shape[-1])
    rows, n_cols = x_flat.shape
    out = torch.empty_like(x_flat)
    block = triton.next_power_of_2(n_cols)
    grid = lambda meta: (triton.cdiv(rows, meta["ROWS"]),)
    _softmax_tuned_kernel[grid](x_flat, out, x_flat.stride(0), rows, n_cols,
                                BLOCK=block)
    return out.reshape(x.shape)
