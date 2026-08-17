"""Latency benchmark for the fused linear+GeLU epilogue (Task 12).

Three arms: `torch_gelu` (separate F.linear then F.gelu), `triton`
(Task 10's unfused matmul kernel plus a separate Triton gelu -- two
kernel launches per call), and `triton_gelu` (the fused kernel from
model/kernels/linear.py, one launch). Shape matches the model's first
MLP projection: (batch, 64, 192) -> (batch, 64, 768).

Matmul is compute-bound (see bench/run_linear.py), so this module also
re-reads its own rows back out of bench/results/latency.csv and prints a
TFLOPs table alongside the shared GB/s columns.

See bench/runner.py for the sweep / single-shot profile contract shared
by every bench/run_*.py module.
"""
import csv
import sys

import torch

from bench.runner import RunnerSpec, main
from model.baseline.layers import linear_gelu as linear_gelu_torch
from model.kernels.gelu import gelu as gelu_triton
from model.kernels.linear import linear as linear_triton
from model.kernels.linear import linear_gelu as linear_gelu_triton
from model.kernels.linear import linear_gelu_tuned as linear_gelu_triton_tuned
from model.kernels.linear import linear_tuned as linear_triton_tuned

SEQ, K, N = 64, 192, 768
RESULTS_PATH = "bench/results/latency.csv"


def _arms_for_batch(batch: int, dtype: torch.dtype):
    x = torch.randn(batch, SEQ, K, device="cuda", dtype=dtype)
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.05
    b = torch.randn(N, device="cuda", dtype=dtype)
    return {
        "torch_gelu": lambda: linear_gelu_torch(x, w, b),
        "triton": lambda: gelu_triton(linear_triton(x, w, b)),
        "triton_gelu": lambda: linear_gelu_triton(x, w, b),
        "triton_tuned": lambda: gelu_triton(linear_triton_tuned(x, w, b)),
        "triton_tuned_gelu": lambda: linear_gelu_triton_tuned(x, w, b),
    }


def _bytes_theoretical(batch: int) -> int:
    # read x, read weight, read bias, write the [batch, 64, 768] output.
    return (batch * SEQ * K + N * K + N + batch * SEQ * N) * 4


SPEC = RunnerSpec(kernel="linear_gelu", arms_for_batch=_arms_for_batch,
                  bytes_theoretical=_bytes_theoretical)


def _print_tflops_table() -> None:
    with open(RESULTS_PATH, newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["kernel"] == SPEC.kernel]

    print("\n--- TFLOPs summary (2*M*N*K / seconds) ---")
    by_batch: dict[int, dict[str, float]] = {}
    for row in rows:
        batch = int(row["batch"])
        m = batch * SEQ
        seconds = float(row["latency_ms_median"]) * 1e-3
        tflops = 2 * m * N * K / seconds / 1e12
        by_batch.setdefault(batch, {})[row["variant"]] = tflops

    for batch, variants in sorted(by_batch.items()):
        print(f"batch={batch:>4}: " + "  ".join(
            f"{name}={tflops:.3f} TFLOPs" for name, tflops in variants.items()))


if __name__ == "__main__":
    main(SPEC)
    if "--variant" not in sys.argv:
        _print_tflops_table()
