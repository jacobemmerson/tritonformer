"""Step 0: determine whether tl.dot is usable on this GPU.

Run this before any kernel work. If tl.dot fails or is catastrophically
slow here, matmul and attention work moves to Modal.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def try_dot(dtype, M=512, N=512, K=512):
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    c = torch.empty((M, N), device="cuda", dtype=dtype)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _matmul_kernel[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                         c.stride(0), c.stride(1),
                         BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    ref = (a.float() @ b.float()).to(dtype)
    max_err = (c.float() - ref.float()).abs().max().item()
    ms = triton.testing.do_bench(
        lambda: _matmul_kernel[grid](a, b, c, M, N, K,
                                     a.stride(0), a.stride(1),
                                     b.stride(0), b.stride(1),
                                     c.stride(0), c.stride(1),
                                     BLOCK_M=64, BLOCK_N=64, BLOCK_K=32))
    tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12
    torch_ms = triton.testing.do_bench(lambda: a @ b)
    torch_tflops = (2 * M * N * K) / (torch_ms * 1e-3) / 1e12
    return max_err, tflops, torch_tflops


def probe():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:             {props.name}")
    print(f"Compute cap:     {props.major}.{props.minor}")
    print(f"SMs:             {props.multi_processor_count}")
    print(f"Total VRAM:      {props.total_memory / 1e9:.2f} GB")
    print(f"Shared mem/blk:  {props.shared_memory_per_block} B")
    print(f"Torch:           {torch.__version__}")
    print(f"Triton:          {triton.__version__}")
    print()

    for dtype in (torch.float32, torch.float16):
        try:
            max_err, tflops, torch_tflops = try_dot(dtype)
            print(f"tl.dot {str(dtype):<16} OK   max_err={max_err:.2e}  "
                  f"triton={tflops:.2f} TFLOPs  torch={torch_tflops:.2f} TFLOPs")
        except Exception as exc:
            print(f"tl.dot {str(dtype):<16} FAIL {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    probe()
