"""Latency benchmark for the gelu kernel ladder rung.

See bench/runner.py for the sweep / single-shot profile contract shared by
every bench/run_*.py module.
"""
import torch

from bench.runner import RunnerSpec, main
from model.baseline.layers import gelu as gelu_torch
from model.kernels.gelu import gelu as gelu_triton

SEQ, DIM = 64, 768


def _arms_for_batch(batch: int, dtype: torch.dtype):
    x = torch.randn(batch, SEQ, DIM, device="cuda", dtype=dtype)
    return {
        "torch": lambda: gelu_torch(x),
        "triton": lambda: gelu_triton(x),
    }


def _bytes_theoretical(batch: int) -> int:
    return 2 * batch * SEQ * DIM * 4


SPEC = RunnerSpec(kernel="gelu", arms_for_batch=_arms_for_batch,
                  bytes_theoretical=_bytes_theoretical)


if __name__ == "__main__":
    main(SPEC)
