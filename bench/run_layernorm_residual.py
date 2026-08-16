"""Latency benchmark for the fused LayerNorm+residual kernel (Task 11).

Three arms: `torch` (separate add then F.layer_norm), `triton` (Task 7's
unfused kernel plus a separate add), and `triton_residual` (the fused
kernel from model/kernels/layernorm.py). See bench/runner.py for the
sweep / single-shot profile contract shared by every bench/run_*.py module.
"""
import torch

from bench.runner import RunnerSpec, main
from model.baseline.layers import layernorm as layernorm_torch
from model.kernels.layernorm import layernorm as layernorm_triton
from model.kernels.layernorm import layernorm_residual as layernorm_residual_triton

SEQ, DIM = 64, 192


def _arms_for_batch(batch: int, dtype: torch.dtype):
    x = torch.randn(batch, SEQ, DIM, device="cuda", dtype=dtype)
    residual = torch.randn(batch, SEQ, DIM, device="cuda", dtype=dtype)
    w = torch.randn(DIM, device="cuda", dtype=dtype)
    b = torch.randn(DIM, device="cuda", dtype=dtype)

    def torch_arm():
        updated = x + residual
        return layernorm_torch(updated, w, b, 1e-5), updated

    def triton_arm():
        updated = x + residual
        return layernorm_triton(updated, w, b, 1e-5), updated

    return {
        "torch": torch_arm,
        "triton": triton_arm,
        "triton_residual": lambda: layernorm_residual_triton(x, residual, w, b, 1e-5),
    }


def _bytes_theoretical(batch: int) -> int:
    # read x, read residual, write sum, write normed
    return 4 * batch * SEQ * DIM * 4


SPEC = RunnerSpec(kernel="layernorm_residual", arms_for_batch=_arms_for_batch,
                  bytes_theoretical=_bytes_theoretical)


if __name__ == "__main__":
    main(SPEC)
