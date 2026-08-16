import torch
from torch import Tensor

from model.kernels.softmax import softmax
from model.registry import Component, register


@register(Component.ATTENTION, "triton_composed")
def attention_composed(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Rung 8: unfused. The [B, H, S, S] score matrix is materialized in
    DRAM between the two matmuls -- this is exactly what rung 10 removes."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    return torch.matmul(softmax(scores), v)
