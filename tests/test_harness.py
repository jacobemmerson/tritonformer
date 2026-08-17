import csv
import subprocess
import time

import torch
import triton

import bench.harness as harness
from bench.clocks import locked_clock_mhz, telemetry
from bench.harness import Measurement, TelemetrySummary, compare, record


def test_compare_runs_arms_interleaved(monkeypatch):
    # triton.testing.do_bench calls fn() several times internally (an
    # initial call plus 5 runtime-estimation calls, before warmup/repeat),
    # so a raw fn()-call order list bursts within a single do_bench
    # invocation regardless of compare()'s interleaving logic. Stub
    # do_bench to a single call so the order list reflects the thing
    # actually under test: whether compare() interleaves arms at the rep
    # level rather than batching all reps of one arm before the next.
    monkeypatch.setattr(triton.testing, "do_bench",
                         lambda fn, **kwargs: fn() or 0.0)

    order = []

    def make(name):
        def arm():
            order.append(name)
        return arm

    compare({"a": make("a"), "b": make("b")}, reps=4)
    # Interleaving means neighbours differ; batching would give aaaabbbb.
    pairs = list(zip(order, order[1:]))
    assert sum(1 for x, y in pairs if x == y) < len(pairs) / 2


def test_compare_returns_samples_per_arm(monkeypatch):
    monkeypatch.setattr(harness, "_sample_telemetry", lambda: (-1, -1))
    samples, _telemetry = compare({"a": lambda: None, "b": lambda: None}, reps=5)
    assert set(samples) == {"a", "b"}
    assert all(len(v) == 5 for v in samples.values())


def test_compare_telemetry_summary_takes_min_clock_and_max_temp(monkeypatch):
    """compare() must sample telemetry DURING its rep loop and report the
    worst (min clock, max temp) seen per arm, not a single post-hoc read --
    a post-hoc read is exactly the bug this replaces: the GPU can idle and
    recover to its locked clock in the moment between compare() returning
    and a later telemetry read, so a row built from that read looks clean
    even when the kernel ran through a real throttle dip."""
    readings = iter([(1830, 60), (1830, 61), (300, 85), (1830, 62)])
    monkeypatch.setattr(harness, "_sample_telemetry", lambda: next(readings))

    samples, telemetry_summary = compare(
        {"a": lambda: None}, reps=4, telemetry_interval=1)

    assert len(samples["a"]) == 4
    assert telemetry_summary["a"] == TelemetrySummary(
        min_sm_clock_mhz=300, max_temp_c=85)


def test_sample_telemetry_falls_back_to_nvidia_smi_when_nvml_unavailable(
        monkeypatch):
    """NVML (torch.cuda.clock_rate/temperature) is ~36x cheaper than the
    nvidia-smi subprocess, so compare() prefers it -- but must not crash
    a whole sweep if NVML is unavailable (e.g. nvidia-ml-py not installed,
    or a driver mismatch raises inside torch's NVML bindings)."""
    def raise_nvml_error(*_args, **_kwargs):
        raise RuntimeError("NVML unavailable")

    monkeypatch.setattr(torch.cuda, "clock_rate", raise_nvml_error)
    monkeypatch.setattr(harness, "telemetry", lambda: (1234, 56))

    assert harness._sample_telemetry() == (1234, 56)


def test_record_writes_header_once(tmp_path):
    path = tmp_path / "out.csv"
    row = Measurement(
        timestamp="2026-08-13T00:00:00", commit_sha="abc1234", gpu="test",
        kernel="layernorm", variant="triton", batch=8, dtype="float32",
        latency_ms_median=1.0, latency_ms_p10=0.9, latency_ms_p90=1.1,
        bytes_theoretical=1024, achieved_gbps=1.0,
        sm_clock_mhz=1500, temp_c=60, flagged=False)
    record([row], str(path))
    record([row], str(path))
    with open(path) as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "timestamp"
    assert len(rows) == 3


def test_unlocked_clock_reports_na_without_raising(monkeypatch):
    """nvidia-smi reports "[N/A]" for clock/temp fields on an unlocked card
    instead of an integer. _query must treat that as "unavailable" rather
    than propagating the ValueError from int() -- this shipped as a bug
    once already."""
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="[N/A]\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert locked_clock_mhz() is None
    assert telemetry() == (-1, -1)


def test_flagged_when_clock_deviates():
    row = Measurement.build(
        kernel="k", variant="v", batch=1, dtype="float32",
        samples=[1.0, 1.0, 1.0], bytes_theoretical=1024,
        gpu="test", sm_clock_mhz=1000, temp_c=80, locked_clock_mhz=1500)
    assert row.flagged is True


def test_locked_clock_mhz_unset_returns_none(monkeypatch):
    """No TRITONFORMER_LOCKED_CLOCK_MHZ declared means "no lock declared",
    not "definitely unlocked" -- GeForce has no queryable read-back of an
    `-lgc` lock, so None is the honest answer either way. Measurement.build
    treats a falsy locked_clock_mhz as "don't flag"."""
    monkeypatch.delenv("TRITONFORMER_LOCKED_CLOCK_MHZ", raising=False)
    assert locked_clock_mhz() is None
    row = Measurement.build(
        kernel="k", variant="v", batch=1, dtype="float32",
        samples=[1.0, 1.0, 1.0], bytes_theoretical=1024,
        gpu="test", sm_clock_mhz=1000, temp_c=80,
        locked_clock_mhz=locked_clock_mhz())
    assert row.flagged is False


def test_locked_clock_mhz_set_and_matching_is_not_flagged(monkeypatch):
    monkeypatch.setenv("TRITONFORMER_LOCKED_CLOCK_MHZ", "1830")
    assert locked_clock_mhz() == 1830
    row = Measurement.build(
        kernel="k", variant="v", batch=1, dtype="float32",
        samples=[1.0, 1.0, 1.0], bytes_theoretical=1024,
        gpu="test", sm_clock_mhz=1830, temp_c=60,
        locked_clock_mhz=locked_clock_mhz())
    assert row.flagged is False


def test_locked_clock_mhz_set_and_drifting_is_flagged(monkeypatch):
    monkeypatch.setenv("TRITONFORMER_LOCKED_CLOCK_MHZ", "1830")
    row = Measurement.build(
        kernel="k", variant="v", batch=1, dtype="float32",
        samples=[1.0, 1.0, 1.0], bytes_theoretical=1024,
        gpu="test", sm_clock_mhz=1200, temp_c=80,
        locked_clock_mhz=locked_clock_mhz())
    assert row.flagged is True
