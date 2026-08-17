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
