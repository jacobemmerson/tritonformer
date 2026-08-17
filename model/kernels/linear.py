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


def _matmul_configs():
    # Grid measured post-merge: committed BLOCK_M=64 is far too small for
    # M = batch*64 (32,768 at batch=512), leaving too few program instances
    # per SM to hide memory latency. GROUP_M=8 is the standard Triton
    # matmul L2 swizzle -- reorders pid_m/pid_n so consecutive programs
    # reuse the same slice of x/w tiles instead of striding across the
    # full row-major grid, which the committed naive kernel does not do.
    configs = []
    for block_m in (64, 128, 256):
        for block_n in (64, 128):
            for block_k in (32, 64):
                for num_warps in (4, 8):
                    for num_stages in (2, 3, 4):
                        configs.append(triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n,
                             "BLOCK_K": block_k, "GROUP_M": 8},
                            num_warps=num_warps, num_stages=num_stages))
    return configs


@triton.jit
def _swizzle_pid(pid, m, n, block_m, block_n, group_m):
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(configs=_matmul_configs(), key=["M", "N", "K"])
@triton.jit
def _linear_tuned_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                         M, N, K,
                         stride_xm, stride_xk,
                         stride_wn, stride_wk,
                         stride_om, stride_on,
                         HAS_BIAS: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    pid_m, pid_n = _swizzle_pid(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
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

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@register(Component.LINEAR, "triton_tuned")
def linear_tuned(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    leading = x.shape[:-1]
    x_flat = x.contiguous().reshape(-1, x.shape[-1])
    m, k = x_flat.shape
    n = weight.shape[0]
    weight = weight.contiguous()
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) *
                         triton.cdiv(n, meta["BLOCK_N"]),)
    _linear_tuned_kernel[grid](
        x_flat, weight, bias if bias is not None else x_flat, out,
        m, n, k,
        x_flat.stride(0), x_flat.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None)
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


@triton.autotune(configs=_matmul_configs(), key=["M", "N", "K"])
@triton.jit
def _linear_gelu_tuned_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                              M, N, K,
                              stride_xm, stride_xk,
                              stride_wn, stride_wk,
                              stride_om, stride_on,
                              HAS_BIAS: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                              BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    pid_m, pid_n = _swizzle_pid(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
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

    inner = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
    acc = 0.5 * acc * (1.0 + libdevice.tanh(inner))

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@register(Component.LINEAR, "triton_tuned_gelu")
def linear_gelu_tuned(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    leading = x.shape[:-1]
    x_flat = x.contiguous().reshape(-1, x.shape[-1])
    m, k = x_flat.shape
    n = weight.shape[0]
    weight = weight.contiguous()
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) *
                         triton.cdiv(n, meta["BLOCK_N"]),)
    _linear_gelu_tuned_kernel[grid](
        x_flat, weight, bias if bias is not None else x_flat, out,
        m, n, k,
        x_flat.stride(0), x_flat.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None)
    return out.reshape(*leading, n)
