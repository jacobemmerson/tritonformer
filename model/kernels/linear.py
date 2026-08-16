import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from model.registry import Component, register


@triton.jit
def _linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                   M, N, K,
                   stride_xm, stride_xk,
                   stride_wn, stride_wk,
                   stride_om, stride_on,
                   HAS_BIAS: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # weight is [N, K], so the N axis walks stride_wn and K walks stride_wk.
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                    other=0.0)
        w = tl.load(w_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_remaining),
                    other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@register(Component.LINEAR, "triton")
def linear(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    leading = x.shape[:-1]
    x_flat = x.contiguous().reshape(-1, x.shape[-1])
    m, k = x_flat.shape
    n = weight.shape[0]
    weight = weight.contiguous()
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),
                         triton.cdiv(n, meta["BLOCK_N"]))
    _linear_kernel[grid](
        x_flat, weight, bias if bias is not None else x_flat, out,
        m, n, k,
        x_flat.stride(0), x_flat.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2)
    return out.reshape(*leading, n)


@triton.jit
def _linear_gelu_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                        M, N, K,
                        stride_xm, stride_xk,
                        stride_wn, stride_wk,
                        stride_om, stride_on,
                        HAS_BIAS: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                    other=0.0)
        w = tl.load(w_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_remaining),
                    other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]

    # The epilogue: the [M, N] tile never leaves registers between the
    # matmul and the activation.
    inner = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
    acc = 0.5 * acc * (1.0 + libdevice.tanh(inner))

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@register(Component.LINEAR, "triton_gelu")
def linear_gelu(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    leading = x.shape[:-1]
    x_flat = x.contiguous().reshape(-1, x.shape[-1])
    m, k = x_flat.shape
    n = weight.shape[0]
    weight = weight.contiguous()
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),
                         triton.cdiv(n, meta["BLOCK_N"]))
    _linear_gelu_kernel[grid](
        x_flat, weight, bias if bias is not None else x_flat, out,
        m, n, k,
        x_flat.stride(0), x_flat.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2)
    return out.reshape(*leading, n)
