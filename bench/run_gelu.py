"""Latency benchmark for the gelu kernel ladder rung.

Two modes, selected by whether --variant is given:

- Sweep (default, no --variant): times both arms across a batch sweep and
  appends Measurement rows to bench/results/latency.csv. This is what a
  human runs directly.
- Single-shot profile (--kernel --variant --batch --dtype, e.g.
  `--variant triton --batch 8 --dtype float32`): runs exactly one arm at one
  batch size, launching it at least 10 times, and writes no CSV. This is
  the contract bench/profile.py::profile_kernel needs: it invokes this
  module under `ncu --launch-skip 5 --launch-count 1`, so ncu can only
  capture a steady-state launch if there are more than 5 launches to skip
  past. Mirrors bench/run_layernorm.py's contract exactly.
"""
import argparse

import torch

from bench.clocks import locked_clock_mhz
from bench.harness import Measurement, compare, record
from model.baseline.layers import gelu as gelu_torch
from model.kernels.gelu import gelu as gelu_triton

SEQ, DIM = 64, 768
BATCHES = [1, 8, 32, 128, 512]
RESULTS_PATH = "bench/results/latency.csv"

DTYPES = {"float32": torch.float32, "float16": torch.float16}

ARMS = {"torch": gelu_torch, "triton": gelu_triton}


def _make_inputs(batch: int, dtype: torch.dtype, device: str = "cuda"):
    x = torch.randn(batch, SEQ, DIM, device=device, dtype=dtype)
    return (x,)


def run_single(kernel: str, variant: str, batch: int, dtype: str) -> None:
    del kernel  # only one kernel lives in this module
    fn = ARMS[variant]
    (x,) = _make_inputs(batch, DTYPES[dtype])
    for _ in range(10):
        fn(x)
    torch.cuda.synchronize()


def run_sweep() -> None:
    clock = locked_clock_mhz()
    rows: list[Measurement] = []
    ceiling = None
    for batch in BATCHES:
        try:
            (x,) = _make_inputs(batch, torch.float32)
        except torch.cuda.OutOfMemoryError:
            ceiling = batch
            break

        arms = {
            "torch": lambda x=x: gelu_torch(x),
            "triton": lambda x=x: gelu_triton(x),
        }
        try:
            samples = compare(arms)
        except torch.cuda.OutOfMemoryError:
            ceiling = batch
            break

        bytes_theoretical = 2 * batch * SEQ * DIM * 4
        for variant, values in samples.items():
            rows.append(Measurement.build(
                kernel="gelu", variant=variant, batch=batch,
                dtype="float32", samples=values,
                bytes_theoretical=bytes_theoretical,
                locked_clock_mhz=clock))
            print(f"batch={batch:>4} {variant:>6}: "
                  f"{rows[-1].latency_ms_median:.4f} ms  "
                  f"{rows[-1].achieved_gbps:.1f} GB/s")

    if ceiling is not None:
        print(f"OOM reached at batch={ceiling}; sweep truncated.")

    record(rows, RESULTS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="gelu")
    parser.add_argument("--variant", choices=list(ARMS), default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dtype", choices=list(DTYPES), default="float32")
    args = parser.parse_args()

    if args.variant is not None:
        run_single(args.kernel, args.variant, args.batch, args.dtype)
    else:
        run_sweep()


if __name__ == "__main__":
    main()
