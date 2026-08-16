import torch
import triton
import triton.language as tl
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


@triton.jit
def _flash_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                  stride_qb, stride_qh, stride_qs, stride_qd,
                  stride_ob, stride_oh, stride_os, stride_od,
                  seq_len, scale,
                  BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr):
    """One program per (batch, head). At S=64, Dh=64 the whole head fits in
    SRAM, so there is no outer tile loop and online rescaling degenerates
    to a single softmax pass. This is a property of the small problem size,
    not a general FlashAttention implementation.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    mask_s = offs_s < seq_len

    base = pid_b * stride_qb + pid_h * stride_qh
    qkv_offsets = base + offs_s[:, None] * stride_qs + offs_d[None, :] * stride_qd
    load_mask = mask_s[:, None]

    q = tl.load(q_ptr + qkv_offsets, mask=load_mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + qkv_offsets, mask=load_mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + qkv_offsets, mask=load_mask, other=0.0).to(tl.float32)

    scores = tl.dot(q, tl.trans(k)) * scale
    # Masked key positions must lose the row max and contribute zero weight.
    scores = tl.where(mask_s[None, :], scores, float("-inf"))
    scores = scores - tl.max(scores, axis=1)[:, None]
    weights = tl.exp(scores)
    weights = tl.where(mask_s[None, :], weights, 0.0)
    weights = weights / tl.sum(weights, axis=1)[:, None]

    out = tl.dot(weights.to(v.dtype), v)
    out_offsets = (pid_b * stride_ob + pid_h * stride_oh
                   + offs_s[:, None] * stride_os + offs_d[None, :] * stride_od)
    tl.store(out_ptr + out_offsets, out, mask=load_mask)


@register(Component.ATTENTION, "triton_flash")
def attention_flash(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, heads, seq_len, head_dim = q.shape
    out = torch.empty_like(q)
    _flash_kernel[(batch, heads)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        seq_len, scale,
        BLOCK_S=triton.next_power_of_2(seq_len),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4, num_stages=2)
    return out
