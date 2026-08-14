"""Pure-PyTorch reference. This is the oracle for every correctness test
and the baseline for every benchmark, so it stays deliberately boring:
plain functional calls, no optimization, no cleverness.
"""
import torch
import torch.nn.functional as F
from torch import Tensor

from model.registry import Component, register


@register(Component.LAYERNORM, "torch")
def layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


@register(Component.GELU, "torch")
def gelu(x: Tensor) -> Tensor:
    return F.gelu(x, approximate="tanh")


@register(Component.SOFTMAX, "torch")
def softmax(x: Tensor) -> Tensor:
    return F.softmax(x, dim=-1)


@register(Component.LINEAR, "torch")
def linear(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    return F.linear(x, weight, bias)


@register(Component.ATTENTION, "torch")
def attention(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    scores = softmax((q @ k.transpose(-2, -1)) * scale)
    return scores @ v


@register(Component.MLP, "torch")
def mlp(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> Tensor:
    return linear(gelu(linear(x, w1, b1)), w2, b2)


@register(Component.BLOCK, "torch")
def block(x: Tensor, ln1_w: Tensor, ln1_b: Tensor,
          qkv_w: Tensor, qkv_b: Tensor, proj_w: Tensor, proj_b: Tensor,
          ln2_w: Tensor, ln2_b: Tensor,
          w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor,
          heads: int, scale: float, eps: float) -> Tensor:
    batch, seq, dim = x.shape
    head_dim = dim // heads

    normed = layernorm(x, ln1_w, ln1_b, eps)
    qkv = linear(normed, qkv_w, qkv_b)
    qkv = qkv.reshape(batch, seq, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attended = attention(q, k, v, scale)
    attended = attended.transpose(1, 2).reshape(batch, seq, dim)
    x = x + linear(attended, proj_w, proj_b)

    return x + mlp(layernorm(x, ln2_w, ln2_b, eps), w1, b1, w2, b2)
