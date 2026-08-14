import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from model.registry import Component, register

SQRT_2_OVER_PI = 0.7978845608028654


@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tl.store(out_ptr + offsets, 0.5 * x * (1.0 + libdevice.tanh(inner)),
             mask=mask)


@register(Component.GELU, "triton")
def gelu(x: Tensor) -> Tensor:
    x_flat = x.contiguous().reshape(-1)
    out = torch.empty_like(x_flat)
    n_elements = x_flat.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    _gelu_kernel[grid](x_flat, out, n_elements, BLOCK=1024)
    return out.reshape(x.shape)
