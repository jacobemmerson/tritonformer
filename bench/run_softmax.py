"""Latency benchmark for the softmax kernel ladder rung.

Rows are only 64 floats (256 B), so torch is expected to win at low batch
where launch overhead dominates; the interesting number is the crossover
batch where Triton catches up. See bench/runner.py for the sweep /
single-shot profile contract shared by every bench/run_*.py module.
"""
import torch

from bench.runner import RunnerSpec, main
from model.baseline.layers import softmax as softmax_torch
from model.kernels.softmax import softmax as softmax_triton

HEADS, SEQ = 3, 64


def _arms_for_batch(batch: int, dtype: torch.dtype):
    x = torch.randn(batch, HEADS, SEQ, SEQ, device="cuda", dtype=dtype)
    return {
        "torch": lambda: softmax_torch(x),
        "triton": lambda: softmax_triton(x),
    }


def _bytes_theoretical(batch: int) -> int:
    return 2 * batch * HEADS * SEQ * SEQ * 4


SPEC = RunnerSpec(kernel="softmax", arms_for_batch=_arms_for_batch,
                  bytes_theoretical=_bytes_theoretical)


if __name__ == "__main__":
    main(SPEC)
