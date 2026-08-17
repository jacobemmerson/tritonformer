import torch
import triton
import triton.language as tl
from torch import Tensor

from model.registry import Component, register


@triton.jit
def _layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                      stride_row, n_cols, eps,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    out_row = out_ptr + row * stride_row

    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # `other=0.0` matters twice: masked lanes must not pollute the sum, and
    # they must not fault. D=192 with BLOCK=256 leaves 64 masked lanes.
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_row + cols, centered * rstd * w + b, mask=mask)


@register(Component.LAYERNORM, "triton")
def layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    x_flat = x.reshape(-1, x.shape[-1]).contiguous()
    rows, n_cols = x_flat.shape
    out = torch.empty_like(x_flat)
    block = triton.next_power_of_2(n_cols)
    _layernorm_kernel[(rows,)](
        x_flat, weight, bias, out,
        x_flat.stride(0), n_cols, eps,
        BLOCK=block,
        num_warps=4 if block <= 512 else 8)
    return out.reshape(x.shape)


@triton.jit
def _layernorm_residual_kernel(x_ptr, res_ptr, w_ptr, b_ptr,
                               out_ptr, res_out_ptr,
                               stride_row, n_cols, eps,
                               BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offset = row * stride_row
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    combined = x + res
    # Written out because the next block's residual path needs it; the
    # saving over the unfused pair is one read and one write of the
    # intermediate, not of this tensor.
    tl.store(res_out_ptr + offset + cols, combined, mask=mask)

    mean = tl.sum(combined, axis=0) / n_cols
    centered = tl.where(mask, combined - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offset + cols, centered * rstd * w + b, mask=mask)


@register(Component.LAYERNORM, "triton_residual")
def layernorm_residual(x: Tensor, residual: Tensor, weight: Tensor,
                       bias: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    shape = x.shape
    x_flat = x.contiguous().reshape(-1, shape[-1])
    res_flat = residual.contiguous().reshape(-1, shape[-1])
    rows, n_cols = x_flat.shape
    out = torch.empty_like(x_flat)
    res_out = torch.empty_like(x_flat)
    block = triton.next_power_of_2(n_cols)
    _layernorm_residual_kernel[(rows,)](
        x_flat, res_flat, weight, bias, out, res_out,
        x_flat.stride(0), n_cols, eps,
        BLOCK=block, num_warps=4 if block <= 512 else 8)
    return out.reshape(shape), res_out.reshape(shape)
