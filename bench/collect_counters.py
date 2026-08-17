"""Reproducible counter collection for the full ViT forward pass (Task 19).

Before this driver, counters.csv rows were gathered by ad-hoc interactive
profile_kernel() calls, so the grid could not be regenerated from the
repo. This module fixes that: it determines each arm's launch cardinality
empirically (with torch.profiler, cheaply, in-process -- no ncu involved)
and passes it to profile_kernel(..., expected_kernels=N), so a silent
partial ncu capture raises instead of quietly undercounting an arm's
traffic.

Scope: batch=1 only, one complete forward pass per variant. The
block-level counters already collected via bench/run_block.py cover both
batch extremes (1 and 512) for triton_composed/triton_fused and carry the
over-fusion finding; the vit_forward level mostly adds embed/final-norm/
head work that is identical across variants, and ncu's kernel-replay cost
on a full 512-batch model forward is prohibitive on this disk-constrained
host. A single batch=1 pass per variant is what this driver captures.

One model.forward() call launches many CUDA kernels (patch-embed matmul,
six transformer blocks, final norm and head), not one. ncu's
--launch-skip/--launch-count count individual kernel launches across the
whole profiled process, so the naive per-single-kernel default
(launch_skip=5, launch_count=1) would land inside the FIRST, cold forward
pass -- exactly the autotune/cache-cold launch bench/profile.py's
docstring says to skip past. Instead, this driver measures the number of
CUDA kernel launches in one steady-state forward pass and uses that as
both launch_skip (skip one full warm pass) and launch_count (capture
EXACTLY the next one pass, no more) -- a wider window would double-count
launches across pass boundaries and inflate every derived byte/launch
figure.

Each variant's cycle length is measured in its OWN fresh subprocess
(_measure_cycle), not by looping over variants in this driver's own
process. Measuring several variants back-to-back in one long-lived
process was tried first and produced flaky, drifting launch counts for
whichever variant ran last: PyTorch's cuBLASLt handle caches a
timing-based heuristic per GEMM shape, and calling a differently-shaped
GEMM (a different variant's patch-embed/head linear) earlier in the same
process measurably perturbed that cache for the next variant. ncu's own
capture always launches a brand-new, isolated process per profile_kernel
call, so measuring in an equally fresh, isolated subprocess is what
actually predicts what ncu will see -- confirmed empirically: 10/10
identical launch/distinct counts per variant across repeated fresh
subprocess measurements, versus visible drift when measured back-to-back
in one process.
"""
import argparse
import json
import subprocess
import sys

from bench.profile import profile_kernel, record_counters
from model.registry import Component, variants

RESULTS_PATH = "bench/results/counters.csv"
BATCH = 1

_MEASURE_SCRIPT = """
import json
import torch
from torch.profiler import ProfilerActivity, profile
from bench.run_sweep import CFG, _models

device = torch.device("cuda")
model = _models(device)[{variant!r}]
images = torch.randn({batch}, CFG.in_channels, CFG.image_size, CFG.image_size,
                     device=device)

with torch.inference_mode():
    for _ in range(3):
        model(images)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        model(images)
        torch.cuda.synchronize()

names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
print(json.dumps({{"stride": len(names), "distinct": len(set(names))}}))
"""


def _measure_cycle(variant: str, batch: int) -> tuple[int, int]:
    """Launches in one steady-state forward pass, and the distinct kernel
    count within it, measured in a fresh subprocess -- see module
    docstring for why a shared process across variants is unreliable
    here."""
    script = _MEASURE_SCRIPT.format(variant=variant, batch=batch)
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return payload["stride"], payload["distinct"]


def _capture(variant: str, batch: int, attempts: int = 3) -> list[dict]:
    """Re-measures the cycle length fresh on each attempt. A mismatch
    means cuBLAS's timing-based algorithm heuristic for the degenerate
    (batch=1) patch-embed/head GEMM shapes landed on a different pick
    between the measurement subprocess and ncu's own subprocess -- both
    are legitimate complete single passes, just with a different kernel
    mix, so this is retried (fresh measurement each time) rather than
    treated as fatal on the first flake."""
    for attempt in range(1, attempts + 1):
        stride, distinct = _measure_cycle(variant, batch)
        print(f"variant={variant} attempt={attempt}: "
             f"stride={stride} launches/pass, {distinct} distinct kernels")
        try:
            return profile_kernel(
                "bench.run_sweep", "vit_forward", variant, batch, "float32",
                launch_skip=stride, launch_count=stride,
                expected_kernels=distinct)
        except RuntimeError as exc:
            if attempt == attempts:
                raise
            print(f"  mismatch on attempt {attempt}, remeasuring and retrying: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=RESULTS_PATH)
    args = parser.parse_args()

    total = 0
    for variant in variants(Component.BLOCK):
        captured = _capture(variant, BATCH)
        # Recorded per arm, not batched to the end: a later arm's
        # RuntimeError (after retries are exhausted) should not discard
        # counters already captured for earlier arms.
        record_counters(captured, args.out)
        total += len(captured)
        print(f"  wrote {len(captured)} rows for {variant} batch={BATCH}")

    print(f"wrote {total} rows total to {args.out}")


if __name__ == "__main__":
    main()
