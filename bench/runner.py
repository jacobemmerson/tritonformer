"""Shared sweep/profile-mode machinery for the bench/run_*.py kernel runners.

Two modes, selected by whether --variant is given:

- Sweep (default, no --variant): times every arm across a batch sweep and
  appends Measurement rows to bench/results/latency.csv. This is what a
  human runs directly.
- Single-shot profile (--kernel --variant --batch --dtype, e.g.
  `--variant triton --batch 8 --dtype float32`): runs exactly one arm at one
  batch size, launching it at least 10 times, and writes no CSV. This is
  the contract bench/profile.py::profile_kernel needs: it invokes the
  concrete runner module under `ncu --launch-skip 5 --launch-count 1`, so
  ncu can only capture a steady-state launch if there are more than 5
  launches to skip past.

A concrete runner (bench/run_layernorm.py, bench/run_gelu.py, ...) supplies
only what differs between kernels: the kernel name, a callable that builds
the arms dict for a given batch, and a callable that computes
bytes_theoretical for a given batch. Everything else -- the sweep loop, OOM
truncation, CSV recording, single-shot profiling, and CLI parsing -- lives
here so a bug fix in one place reaches every kernel.
"""
import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch

from bench.clocks import locked_clock_mhz
from bench.harness import Measurement, compare, record

BATCHES = [1, 8, 32, 128, 512]
RESULTS_PATH = "bench/results/latency.csv"
DTYPES = {"float32": torch.float32, "float16": torch.float16}


@dataclass(frozen=True)
class RunnerSpec:
    kernel: str
    arms_for_batch: Callable[[int, torch.dtype], dict[str, Callable[[], object]]]
    bytes_theoretical: Callable[[int], int]


def run_single(spec: RunnerSpec, variant: str, batch: int, dtype: str) -> None:
    arms = spec.arms_for_batch(batch, DTYPES[dtype])
    if variant not in arms:
        sys.exit(f"unknown variant {variant!r} for kernel {spec.kernel!r}; "
                 f"available: {sorted(arms)}")
    for _ in range(10):
        arms[variant]()
    torch.cuda.synchronize()


def run_sweep(spec: RunnerSpec) -> None:
    clock = locked_clock_mhz()
    rows: list[Measurement] = []
    ceiling = None
    for batch in BATCHES:
        try:
            arms = spec.arms_for_batch(batch, torch.float32)
            samples = compare(arms)
        except torch.cuda.OutOfMemoryError:
            ceiling = batch
            break

        bytes_theoretical = spec.bytes_theoretical(batch)
        for variant, values in samples.items():
            rows.append(Measurement.build(
                kernel=spec.kernel, variant=variant, batch=batch,
                dtype="float32", samples=values,
                bytes_theoretical=bytes_theoretical,
                locked_clock_mhz=clock))
            print(f"batch={batch:>4} {variant:>6}: "
                  f"{rows[-1].latency_ms_median:.4f} ms  "
                  f"{rows[-1].achieved_gbps:.1f} GB/s")

    if ceiling is not None:
        print(f"OOM reached at batch={ceiling}; sweep truncated.")

    record(rows, RESULTS_PATH)


def main(spec: RunnerSpec) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default=spec.kernel)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dtype", choices=list(DTYPES), default="float32")
    args = parser.parse_args()

    if args.variant is not None:
        run_single(spec, args.variant, args.batch, args.dtype)
    else:
        run_sweep(spec)
