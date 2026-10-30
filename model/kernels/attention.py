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


@triton.jit
def _bmm_kernel(a_ptr, b_ptr, out_ptr,
                M, N, K, heads, scale,
                stride_ab, stride_ah, stride_am, stride_ak,
                stride_bb, stride_bh, stride_bk, stride_bn,
                stride_ob, stride_oh, stride_om, stride_on,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    """out[b, h] = scale * a[b, h] @ b[b, h], one program per output tile.

    Same tile shape and K-loop as `_linear_kernel`, so the composed_bmm arm
    differs from `attention_composed` only in who runs the GEMM (this kernel
    instead of cuBLAS), not in how it is tiled. Batch and head strides are
    taken separately rather than flattening to 3-D so that a head-split view
    of a packed QKV buffer, whose batch and head axes do not merge, is read
    in place.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)
    pid_b = pid_bh // heads
    pid_h = pid_bh % heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (a_ptr + pid_b * stride_ab + pid_h * stride_ah
              + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr + pid_b * stride_bb + pid_h * stride_bh
              + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc * scale
    out_ptrs = (out_ptr + pid_b * stride_ob + pid_h * stride_oh
                + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def bmm(a: Tensor, b: Tensor, scale: float = 1.0) -> Tensor:
    """[B, H, M, K] @ [B, H, K, N], scaled, through strides: neither operand
    is made contiguous, so `bmm(q, k.transpose(-2, -1))` reads k in place."""
    batch, heads, m, k = a.shape
    n = b.shape[-1]
    out = torch.empty((batch, heads, m, n), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),
                         triton.cdiv(n, meta["BLOCK_N"]),
                         batch * heads)
    _bmm_kernel[grid](
        a, b, out, m, n, k, heads, scale,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2)
    return out


@register(Component.ATTENTION, "triton_composed_bmm")
def attention_composed_bmm(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """`attention_composed` with cuBLAS taken out of the comparison.

    `attention_composed` is "cuBLAS batched GEMM + Triton softmax", so its gap
    to `attention_flash` mixes the cost of fusing with the cost of cuBLAS
    versus a hand-written Triton GEMM. This arm runs the same three stages on
    the same Triton GEMM tiling the fused kernels are built from. The scale is
    applied as the GEMM's alpha epilogue, as cuBLAS applies it, so this is
    three launches to `attention_composed`'s four.
    """
    scores = bmm(q, k.transpose(-2, -1), scale)
    return bmm(softmax(scores), v)


@triton.jit
def _flash_tiled_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                        stride_qb, stride_qh, stride_qs, stride_qd,
                        stride_kb, stride_kh, stride_ks, stride_kd,
                        stride_vb, stride_vh, stride_vs, stride_vd,
                        stride_ob, stride_oh, stride_os, stride_od,
                        heads, seq_len, scale,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_D: tl.constexpr):
    """One program per (BLOCK_M query rows, batch, head); keys and values are
    streamed in BLOCK_N tiles with the online-softmax rescaling that
    `_flash_kernel` could skip because it held the whole [64, 64] head at
    once. Live state per program is q, the accumulator, and one k/v tile,
    instead of six [64, 64] fp32 tiles.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // heads
    pid_h = pid_bh % heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < seq_len

    q_ptrs = (q_ptr + pid_b * stride_qb + pid_h * stride_qh
              + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    row_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, seq_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        k_ptrs = (k_ptr + pid_b * stride_kb + pid_h * stride_kh
                  + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd)
        v_ptrs = (v_ptr + pid_b * stride_vb + pid_h * stride_vh
                  + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(mask_n[None, :], scores, float("-inf"))

        tile_max = tl.maximum(row_max, tl.max(scores, axis=1))
        rescale = tl.exp(row_max - tile_max)
        weights = tl.exp(scores - tile_max[:, None])
        weights = tl.where(mask_n[None, :], weights, 0.0)

        row_sum = row_sum * rescale + tl.sum(weights, axis=1)
        acc = acc * rescale[:, None] + tl.dot(weights.to(v.dtype), v)
        row_max = tile_max

    out = acc / row_sum[:, None]
    out_ptrs = (out_ptr + pid_b * stride_ob + pid_h * stride_oh
                + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, out, mask=mask_m[:, None])


@register(Component.ATTENTION, "triton_flash_tiled")
def attention_flash_tiled(q: Tensor, k: Tensor, v: Tensor, scale: float,
                          block_m: int = 16, block_n: int = 32,
                          num_warps: int = 4) -> Tensor:
    """The tiled test of the flash result: BLOCK_M query rows per program and
    a loop over BLOCK_N-wide key/value tiles, so the grid is
    (seq / BLOCK_M) x batch x heads instead of batch x heads.

    Defaults BLOCK_M=16, BLOCK_N=32: at seq 64 the key loop runs twice, so the
    online rescaling is exercised rather than degenerating to the single pass
    `_flash_kernel` already takes. `block_n=64` reproduces that single pass.
    """
    batch, heads, seq_len, head_dim = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(seq_len, block_m), batch * heads)
    _flash_tiled_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        heads, seq_len, scale,
        BLOCK_M=block_m, BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=num_warps, num_stages=2)
    return out
