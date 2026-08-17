"""Latency benchmark for the linear (tiled matmul) kernel ladder rung.

Matmul is compute-bound, unlike every bandwidth-bound rung so far
(layernorm, gelu, softmax), so achieved_gbps in the shared CSV is a much
less meaningful number here. After the shared runner records latency,
this module re-reads its own rows back out of bench/results/latency.csv
and prints an achieved-TFLOPs table (2*M*N*K / seconds) for both arms, to
compare against the ~1.86 TFLOPs tl.dot ceiling Task 1 measured on this
GTX 1650 Ti (no tensor cores).

See bench/runner.py for the sweep / single-shot profile contract shared by
every bench/run_*.py module.
"""
import csv
import sys

import torch

from bench.runner import RunnerSpec, main
from model.baseline.layers import linear as linear_torch
from model.kernels.linear import linear as linear_triton
from model.kernels.linear import linear_tuned as linear_triton_tuned

SEQ = 64
RESULTS_PATH = "bench/results/latency.csv"

# (in_features, out_features) actually used by the model.
SHAPES = [(192, 576), (192, 192), (192, 768), (768, 192)]


def _make_spec(k: int, n: int) -> RunnerSpec:
    def _arms_for_batch(batch: int, dtype: torch.dtype):
        x = torch.randn(batch, SEQ, k, device="cuda", dtype=dtype)
        w = torch.randn(n, k, device="cuda", dtype=dtype) * 0.05
        b = torch.randn(n, device="cuda", dtype=dtype)
        return {
            "torch": lambda: linear_torch(x, w, b),
            "triton": lambda: linear_triton(x, w, b),
            "triton_tuned": lambda: linear_triton_tuned(x, w, b),
        }

    def _bytes_theoretical(batch: int) -> int:
        return (batch * SEQ * k + n * k + batch * SEQ * n) * 4

    return RunnerSpec(kernel=f"linear_k{k}_n{n}", arms_for_batch=_arms_for_batch,
                      bytes_theoretical=_bytes_theoretical)


def _print_tflops_table() -> None:
    shape_kernels = {f"linear_k{k}_n{n}": (k, n) for k, n in SHAPES}
    with open(RESULTS_PATH, newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["kernel"] in shape_kernels]

    print("\n--- TFLOPs summary (2*M*N*K / seconds) ---")
    by_shape_batch: dict[tuple[int, int, int], dict[str, float]] = {}
    for row in rows:
        k, n = shape_kernels[row["kernel"]]
        batch = int(row["batch"])
        m = batch * SEQ
        seconds = float(row["latency_ms_median"]) * 1e-3
        tflops = 2 * m * n * k / seconds / 1e12
        by_shape_batch.setdefault((k, n, batch), {})[row["variant"]] = tflops

    for (k, n, batch), variants in sorted(by_shape_batch.items()):
        torch_tflops = variants.get("torch")
        triton_tflops = variants.get("triton")
        ratio = (triton_tflops and torch_tflops
                 and torch_tflops / triton_tflops) or float("nan")
        print(f"k={k:>3} n={n:>3} batch={batch:>4}: "
              f"torch={torch_tflops:.3f} TFLOPs  "
              f"triton={triton_tflops:.3f} TFLOPs  "
              f"torch/triton={ratio:.2f}x")


if __name__ == "__main__":
    specs = [_make_spec(k, n) for k, n in SHAPES]
    main(specs)
    if "--variant" not in sys.argv:
        _print_tflops_table()
