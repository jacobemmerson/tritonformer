"""Rung 13: the fully fused transformer block, the deliberate far end of
the fusion ladder. Predicted to hurt more than Task 16's mega-MLP.

`block_composed` assembles the best individual Triton variant for each
sub-operation. `block_fused` swaps in `mlp_fused` (Task 16's over-fused
MLP) on top of the same fused attention and fused LayerNorm+residual --
it inherits Task 16's 3-4x latency deficit by construction, since
`mlp_fused` is one of its six sub-calls.

A `triton_fused_monolithic` variant that holds both a [BLOCK_M, 768]
hidden tile and a [64, 64] attention tile in a single kernel was
attempted and does not compile on this hardware -- see
docs/findings/05-over-fusion.md for the exact error and the configurations
tried. Task 16 already established the shared-memory wall this hits:
the MLP's own w1/w2 tile alone required 1,048,576 bytes against a 65,536
byte/SM budget, ~16x over, independent of batch tiling. A monolithic
block kernel additionally needs Q/K/V tiles live at the same time, so it
is strictly larger and was never expected to fit. `block_composed` is
therefore the maximum achievable rung.
"""
from torch import Tensor

from model.kernels.attention import attention_flash, qkv_project
from model.kernels.layernorm import layernorm, layernorm_residual
from model.kernels.linear import linear
from model.kernels.mlp import mlp_composed, mlp_fused
from model.registry import Component, register


def _block(x, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b,
           w1, b1, w2, b2, heads, scale, eps, mlp_fn):
    batch, seq, dim = x.shape
    normed = layernorm(x, ln1_w, ln1_b, eps)
    q, k, v = qkv_project(normed, qkv_w, qkv_b, heads)
    attended = attention_flash(q, k, v, scale)
    attended = attended.transpose(1, 2).reshape(batch, seq, dim)
    normed, residual = layernorm_residual(
        linear(attended, proj_w, proj_b), x, ln2_w, ln2_b, eps)
    return residual + mlp_fn(normed, w1, b1, w2, b2)


@register(Component.BLOCK, "triton_composed")
def block_composed(x: Tensor, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b,
                   ln2_w, ln2_b, w1, b1, w2, b2,
                   heads: int, scale: float, eps: float) -> Tensor:
    return _block(x, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b,
                  w1, b1, w2, b2, heads, scale, eps, mlp_composed)


@register(Component.BLOCK, "triton_fused")
def block_fused(x: Tensor, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b,
                ln2_w, ln2_b, w1, b1, w2, b2,
                heads: int, scale: float, eps: float) -> Tensor:
    """Rung 13: maximum fusion. Uses the mega-MLP from rung 12 on top of the
    fused attention and fused LayerNorm+residual, minimizing launch count at
    the cost of the highest register pressure in the project."""
    return _block(x, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b,
                  w1, b1, w2, b2, heads, scale, eps, mlp_fused)
