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


def _sample_telemetry() -> tuple[int, int]:
    """(sm_clock_mhz, temp_c) via NVML (torch.cuda.clock_rate/temperature),
    falling back to the nvidia-smi subprocess in bench/clocks.py::telemetry()
    when NVML is unavailable (e.g. pynvml/nvidia-ml-py not installed).

    NVML measured ~1.7 ms (clock) + ~0.6 ms (temp) on this host, versus
    ~80 ms for the nvidia-smi subprocess telemetry() shells out to -- see
    compare()'s docstring for why that ~36x difference matters here, not
    just as a speed nicety.
    """
    if torch.cuda.is_available():
        try:
            return torch.cuda.clock_rate(), torch.cuda.temperature()
        except Exception:
            pass
    return telemetry()


@dataclass(frozen=True)
class TelemetrySummary:
    """Worst-case telemetry observed for one arm across a compare() run.

    min_sm_clock_mhz and max_temp_c, not averages: a kernel that throttled
    for even a fraction of its reps ran under degraded conditions for that
    fraction, and averaging would dilute that back below the flag
    threshold. -1 means no sample was ever available (nvidia-smi
    unreachable or unsupported field), matching telemetry()'s sentinel.
    """
    min_sm_clock_mhz: int
    max_temp_c: int


def compare(arms: dict[str, Callable[[], object]],
            reps: int = 30, telemetry_interval: int = 1,
            ) -> tuple[dict[str, list[float]], dict[str, TelemetrySummary]]:
    """Time each arm, interleaved at the rep level.

    Batching all reps of one arm before the next measures the heatsink:
    the second arm runs hotter. Interleaving spreads thermal drift across
    both arms instead of loading it entirely onto whichever ran second.

    Telemetry is sampled INSIDE this loop, not after it returns. Sampling
    once at the end (the original design) reads the clock after the GPU
    has had a moment to idle and recover to its locked target, not the
    clock the kernel actually ran under -- confirmed on this card: rows
    recorded a pristine locked clock while direct nvidia-smi polling
    during the same run showed it swinging down to 300 MHz under thermal
    throttling.

    _sample_telemetry() uses NVML (~1.7 ms clock + ~0.6 ms temp) rather
    than the nvidia-smi subprocess (~80 ms/call). That difference is not
    just speed: an early version of this function sampled via the
    nvidia-smi subprocess every 5th rep, and at 78 ms/call that injected
    ~1.4 s of idle time into a 30-rep loop -- long enough for the card to
    recover between kernel launches, which systematically under-reports
    exactly the throttling this function exists to catch. The instrument
    was perturbing what it measured. NVML is cheap enough to sample every
    rep by default (telemetry_interval=1); the parameter stays as a knob
    in case a future host makes even NVML too expensive to call this often.
    """
    for arm in arms.values():
        for _ in range(5):
            arm()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in arms}
    clock_readings: dict[str, list[int]] = {name: [] for name in arms}
    temp_readings: dict[str, list[int]] = {name: [] for name in arms}
    for rep in range(reps):
        for name, arm in arms.items():
            samples[name].append(triton.testing.do_bench(arm, warmup=0, rep=1))
            if rep % telemetry_interval == 0:
                clock, temp = _sample_telemetry()
                if clock >= 0:
                    clock_readings[name].append(clock)
                if temp >= 0:
                    temp_readings[name].append(temp)

    telemetry_summary = {
        name: TelemetrySummary(
            min_sm_clock_mhz=min(clock_readings[name]) if clock_readings[name] else -1,
            max_temp_c=max(temp_readings[name]) if temp_readings[name] else -1)
        for name in arms
    }
    return samples, telemetry_summary


@dataclass(frozen=True)
class Measurement:
    """sm_clock_mhz and temp_c hold WORST-CASE telemetry observed during
    that arm's reps (minimum clock, maximum temperature), not a point
    sample. This is a semantic change from the merged study's CSV, which
    holds the same 16 columns but populated sm_clock_mhz/temp_c from a
    single nvidia-smi read taken AFTER compare() returned -- by which
    point the card had a moment to idle and recover toward its locked
    target, so that column read as clean even during real throttling.
    The column names and count are unchanged deliberately (record()
    writes a header only when the CSV is new, so a schema change would
    corrupt the existing 300+ rows already in bench/results/latency.csv);
    only what the values MEAN changed. Anyone comparing rows across that
    boundary needs to know this.
    """
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
              sm_clock_mhz: int, temp_c: int,
              gpu: str | None = None,
              locked_clock_mhz: int | None = None) -> "Measurement":
        """sm_clock_mhz and temp_c are required, not sampled here: the
        caller (bench/harness.py::compare(), via its TelemetrySummary) must
        supply the worst clock/temp observed DURING measurement. Sampling
        fresh at build time -- the original design -- reads the clock after
        the GPU has idled and recovered, not what the kernel ran under."""
        ordered = sorted(samples)
        median = statistics.median(ordered)
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
