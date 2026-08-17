"""Latency measurement. Counters come from bench/profile.py and are written
to a different file -- ncu serializes and replays kernels, so its durations
are not comparable to anything here.
"""
import csv
import os
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone

import torch
import triton

from bench.clocks import locked_clock_mhz, telemetry

CLOCK_DEVIATION_TOLERANCE = 0.05


def commit_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def compare(arms: dict[str, Callable[[], object]],
            reps: int = 30) -> dict[str, list[float]]:
    """Time each arm, interleaved at the rep level.

    Batching all reps of one arm before the next measures the heatsink:
    the second arm runs hotter. Interleaving spreads thermal drift across
    both arms instead of loading it entirely onto whichever ran second.
    """
    for arm in arms.values():
        for _ in range(5):
            arm()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in arms}
    for _ in range(reps):
        for name, arm in arms.items():
            samples[name].append(triton.testing.do_bench(arm, warmup=0, rep=1))
    return samples


@dataclass(frozen=True)
class Measurement:
    timestamp: str
    commit_sha: str
    gpu: str
    kernel: str
    variant: str
    batch: int
    dtype: str
    latency_ms_median: float
    latency_ms_p10: float
    latency_ms_p90: float
    bytes_theoretical: int
    achieved_gbps: float
    sm_clock_mhz: int
    temp_c: int
    flagged: bool

    @classmethod
    def build(cls, *, kernel: str, variant: str, batch: int, dtype: str,
              samples: list[float], bytes_theoretical: int,
              gpu: str | None = None, sm_clock_mhz: int | None = None,
              temp_c: int | None = None,
              locked_clock_mhz: int | None = None) -> "Measurement":
        ordered = sorted(samples)
        median = statistics.median(ordered)
        if sm_clock_mhz is None or temp_c is None:
            sm_clock_mhz, temp_c = telemetry()
        flagged = False
        if locked_clock_mhz:
            drift = abs(sm_clock_mhz - locked_clock_mhz) / locked_clock_mhz
            flagged = drift > CLOCK_DEVIATION_TOLERANCE
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            commit_sha=commit_sha(),
            gpu=gpu if gpu is not None else gpu_name(),
            kernel=kernel, variant=variant, batch=batch, dtype=dtype,
            latency_ms_median=median,
            latency_ms_p10=ordered[int(0.1 * (len(ordered) - 1))],
            latency_ms_p90=ordered[int(0.9 * (len(ordered) - 1))],
            bytes_theoretical=bytes_theoretical,
            achieved_gbps=bytes_theoretical / (median * 1e-3) / 1e9,
            sm_clock_mhz=sm_clock_mhz, temp_c=temp_c, flagged=flagged)


def record(rows: list[Measurement], path: str) -> None:
    columns = [f.name for f in fields(Measurement)]
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
