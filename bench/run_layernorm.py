"""Latency benchmark for the layernorm kernel ladder rung.

See bench/runner.py for the sweep / single-shot profile contract shared by
every bench/run_*.py module.
"""
import torch

from bench.runner import RunnerSpec, main
from model.baseline.layers import layernorm as layernorm_torch
from model.kernels.layernorm import layernorm as layernorm_triton

SEQ, DIM = 64, 192


def _arms_for_batch(batch: int, dtype: torch.dtype):
    x = torch.randn(batch, SEQ, DIM, device="cuda", dtype=dtype)
    w = torch.randn(DIM, device="cuda", dtype=dtype)
    b = torch.randn(DIM, device="cuda", dtype=dtype)
    return {
        "torch": lambda: layernorm_torch(x, w, b, 1e-5),
        "triton": lambda: layernorm_triton(x, w, b, 1e-5),
    }


def _bytes_theoretical(batch: int) -> int:
    return 2 * batch * SEQ * DIM * 4


SPEC = RunnerSpec(kernel="layernorm", arms_for_batch=_arms_for_batch,
                  bytes_theoretical=_bytes_theoretical)


if __name__ == "__main__":
    main(SPEC)
