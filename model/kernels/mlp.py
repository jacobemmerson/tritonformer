import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from model.kernels.linear import linear, linear_gelu
from model.registry import Component, register


@register(Component.MLP, "triton_composed")
def mlp_composed(x: Tensor, w1: Tensor, b1: Tensor,
                 w2: Tensor, b2: Tensor) -> Tensor:
    return linear(linear_gelu(x, w1, b1), w2, b2)


@triton.jit
def _mlp_fused_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                      M, D, H,
                      stride_xm, stride_xk,
                      stride_w1n, stride_w1k,
                      stride_w2n, stride_w2k,
                      stride_om, stride_on,
                      BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
                      BLOCK_H: tl.constexpr):
    """Rung 12: the deliberate over-fusion.

    Each program owns a [BLOCK_M, D] output tile and must hold the entire
    [BLOCK_M, H] hidden activation to produce it, because the second matmul
    reduces over H. With H=768 that is BLOCK_M*768 floats per program --
    far more than the register file holds, so this is expected to spill to
    local memory. The spill is the finding; check the local_op_ld/st
    counters.

    Deviation from the task brief's reference kernel, recorded here rather
    than silently: the brief's version holds the whole H=768 width at once
    (BLOCK_H = next_power_of_2(768) = 1024) and instructs reducing BLOCK_M
    to 16 then 8 if that overflows shared memory. Empirically (see the
    task report) BLOCK_M has almost no effect on the overflow -- the
    w1/w2 tiles alone are BLOCK_D * BLOCK_H * 4 bytes = 256 * 1024 * 4 =
    1,048,576 bytes, ~16x the 65,536-byte/SM budget, and that term does
    not shrink with BLOCK_M. BLOCK_M=1 still requires 1,053,696 bytes.
    The literal reference kernel cannot compile at any BLOCK_M on this
    card. This version instead loops over H in BLOCK_H-sized tiles (the
    same K-loop structure `_linear_kernel` uses over its reduction
    dimension), accumulating the second matmul as it goes. The hidden
    activation still never reaches DRAM -- each H-tile of it lives only
    in registers between the two `tl.dot` calls before being discarded --
    so this still satisfies "one kernel, hidden dimension never leaves
    the SM." It just does not hold all 768 columns simultaneously, which
    the brief's fixed-BLOCK_H design cannot do on 64KB shared memory
    regardless of batch tiling.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_d[None, :] * stride_xk
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    out = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        w1_ptrs = w1_ptr + offs_h[None, :] * stride_w1n + offs_d[:, None] * stride_w1k
        w1 = tl.load(w1_ptrs, mask=mask_h[None, :] & mask_d[:, None], other=0.0)

        hidden = tl.dot(x, w1)
        hidden += tl.load(b1_ptr + offs_h, mask=mask_h, other=0.0)[None, :]
        inner = 0.7978845608028654 * (hidden + 0.044715 * hidden * hidden * hidden)
        hidden = 0.5 * hidden * (1.0 + libdevice.tanh(inner))
        hidden = tl.where(mask_h[None, :], hidden, 0.0)

        w2_ptrs = w2_ptr + offs_d[None, :] * stride_w2n + offs_h[:, None] * stride_w2k
        w2 = tl.load(w2_ptrs, mask=mask_d[None, :] & mask_h[:, None], other=0.0)

        out += tl.dot(hidden.to(w2.dtype), w2)

    out += tl.load(b2_ptr + offs_d, mask=mask_d, other=0.0)[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])


@register(Component.MLP, "triton_fused")
def mlp_fused(x: Tensor, w1: Tensor, b1: Tensor,
              w2: Tensor, b2: Tensor) -> Tensor:
    shape = x.shape
    x_flat = x.contiguous().reshape(-1, shape[-1])
    m, dim = x_flat.shape
    hidden_dim = w1.shape[0]
    w1, w2 = w1.contiguous(), w2.contiguous()
    out = torch.empty_like(x_flat)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),)
    _mlp_fused_kernel[grid](
        x_flat, w1, b1, w2, b2, out,
        m, dim, hidden_dim,
        x_flat.stride(0), x_flat.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=16,
        BLOCK_D=triton.next_power_of_2(dim),
        BLOCK_H=32,
        num_warps=8, num_stages=1)
    return out.reshape(shape)


@register(Component.MLP, "mlp_fused_lowreg")
def mlp_fused_lowreg(x: Tensor, w1: Tensor, b1: Tensor,
                     w2: Tensor, b2: Tensor) -> Tensor:
    """Experiment 1 (The Flip), `docs/findings/10-register-rule.md`.

    Same kernel body as `_mlp_fused_kernel` (`_mlp_fused_kernel` itself is
    NOT modified -- it is a benchmarked rung baseline every finding cites).
    This variant only shrinks `BLOCK_M` and `num_warps` to test whether
    cutting the `[BLOCK_M, BLOCK_D]` accumulator's register footprint
    restores occupancy enough to flip the fusion from a loss to a win, per
    the register rule in docs/findings/07-retuning.md.

    A register-count sweep across the plan's grid (BLOCK_M in {8,4,2} x
    num_warps in {4,8}, via Triton's compiled-kernel metadata, not ncu)
    found BLOCK_M=2, num_warps=8 is the ONLY config in that grid at
    <=128 regs/thread: 128 regs x 256 threads/block = 32,768 regs/block
    -> 2 blocks/SM by the register budget alone -> the plan's targeted
    50% occupancy, matching `_linear_tuned_kernel`'s regime. This is the
    config registered here; see docs/findings/10-register-rule.md for
    the full sweep table and why shared memory turned out to cap actual
    occupancy at 25% regardless (BLOCK_D x BLOCK_H tile shared-memory
    usage does not shrink with BLOCK_M).
    """
    shape = x.shape
    x_flat = x.contiguous().reshape(-1, shape[-1])
    m, dim = x_flat.shape
    hidden_dim = w1.shape[0]
    w1, w2 = w1.contiguous(), w2.contiguous()
    out = torch.empty_like(x_flat)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),)
    _mlp_fused_kernel[grid](
        x_flat, w1, b1, w2, b2, out,
        m, dim, hidden_dim,
        x_flat.stride(0), x_flat.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=2,
        BLOCK_D=triton.next_power_of_2(dim),
        BLOCK_H=32,
        num_warps=8, num_stages=1)
    return out.reshape(shape)
