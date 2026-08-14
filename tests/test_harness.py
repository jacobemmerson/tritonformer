import csv
import time

import triton

from bench.harness import Measurement, compare, record


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


def test_compare_returns_samples_per_arm():
    samples = compare({"a": lambda: None, "b": lambda: None}, reps=5)
    assert set(samples) == {"a", "b"}
    assert all(len(v) == 5 for v in samples.values())


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


def test_flagged_when_clock_deviates():
    row = Measurement.build(
        kernel="k", variant="v", batch=1, dtype="float32",
        samples=[1.0, 1.0, 1.0], bytes_theoretical=1024,
        gpu="test", sm_clock_mhz=1000, temp_c=80, locked_clock_mhz=1500)
    assert row.flagged is True
