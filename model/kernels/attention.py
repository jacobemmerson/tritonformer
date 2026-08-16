import torch
from torch import Tensor

from model.kernels.linear import linear
from model.kernels.softmax import softmax
from model.registry import Component, register


@register(Component.ATTENTION, "triton_composed")
def attention_composed(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Rung 8: unfused. The [B, H, S, S] score matrix is materialized in
    DRAM between the two matmuls -- this is exactly what rung 10 removes."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    return torch.matmul(softmax(scores), v)


def qkv_project(x: Tensor, qkv_w: Tensor, qkv_b: Tensor,
                heads: int) -> tuple[Tensor, Tensor, Tensor]:
    """Rung 9: one [D -> 3D] GEMM instead of three [D -> D] GEMMs.

    The weight layout must match the reference exactly: rows [0:D] are Q,
    [D:2D] are K, [2D:3D] are V. Getting this wrong produces a model that
    trains fine but loads the checkpoint incorrectly.
    """
    batch, seq, dim = x.shape
    head_dim = dim // heads
    packed = linear(x, qkv_w, qkv_b)
    packed = packed.reshape(batch, seq, 3, heads, head_dim)
    packed = packed.permute(2, 0, 3, 1, 4)
    return packed[0], packed[1], packed[2]


@register(Component.ATTENTION, "triton_qkv_fused")
def attention_qkv_fused(x: Tensor, qkv_w: Tensor, qkv_b: Tensor,
                        heads: int, scale: float) -> Tensor:
    q, k, v = qkv_project(x, qkv_w, qkv_b, heads)
    batch, seq = x.shape[0], x.shape[1]
    out = attention_composed(q, k, v, scale)
    return out.transpose(1, 2).reshape(batch, seq, -1)
