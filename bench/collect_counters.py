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

from bench.profile import (base_kernel_name, profile_kernel,
                           record_counters)
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




# --- per-kernel arm capture -------------------------------------------------
#
# The vit_forward driver above skips one whole steady-state pass, which for a
# model forward is hundreds of launches and therefore clears the benchmark's own
# tensor setup by sheer magnitude. A single-kernel arm has a stride of 1, so the
# same "skip a few strides" reasoning lands *inside* setup: bench/run_mlp.py
# alone launches five `randn` plus two `* 0.05` elementwise multiplies before the
# first arm call, and a five-launch skip profiles a scalar multiply while
# reporting it as the fused kernel. This section therefore measures the setup
# launches explicitly and skips past them, then validates kernel identity rather
# than trusting the arithmetic.

ARM_WARMUP_CALLS = 5

_ARM_MEASURE_SCRIPT = """
import importlib
import json
import torch
from torch.profiler import ProfilerActivity, profile

module = importlib.import_module({module!r})
specs = module.SPEC if isinstance(module.SPEC, list) else [module.SPEC]
spec = [s for s in specs if s.kernel == {kernel!r}][0]

# Create the CUDA context first so its one-off launches are attributed to
# neither the setup count nor the arm's cycle.
torch.zeros(1, device="cuda")
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CUDA]) as setup_prof:
    arms = spec.arms_for_batch({batch}, torch.float32)
    torch.cuda.synchronize()
setup = len([e for e in setup_prof.events() if e.device_type.name == "CUDA"])

arm = arms[{variant!r}]
with torch.inference_mode():
    for _ in range(3):
        arm()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        arm()
        torch.cuda.synchronize()

names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
print(json.dumps({{"setup": setup, "stride": len(names),
                   "distinct": len(set(names)), "names": sorted(set(names))}}))
"""


def measure_arm(module: str, kernel: str, variant: str, batch: int) -> dict:
    """Setup launches, launches per steady-state call, and the kernel names one
    call launches -- measured in a fresh subprocess, for the same reason
    _measure_cycle uses one (see this module's docstring)."""
    script = _ARM_MEASURE_SCRIPT.format(module=module, kernel=kernel,
                                        variant=variant, batch=batch)
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def capture_arm(module: str, kernel: str, variant: str, batch: int,
                dtype: str = "float32") -> list[dict]:
    """Counters for exactly one steady-state call of one per-kernel arm.

    The window is `setup + ARM_WARMUP_CALLS * stride` launches skipped, then
    exactly `stride` captured: past the benchmark's tensor setup, past enough
    warm calls to be in steady state, and covering exactly one whole call. A
    window of exactly one period sums each of the arm's kernels once even if it
    starts mid-cycle, so the totals are comparable across arms with different
    launch counts.
    """
    measured = measure_arm(module, kernel, variant, batch)
    skip = measured["setup"] + ARM_WARMUP_CALLS * measured["stride"]
    print(f"{kernel}/{variant}@{batch}: setup={measured['setup']} launches, "
          f"stride={measured['stride']}, skip={skip}, "
          f"expect={sorted(base_kernel_name(n) for n in measured['names'])}")
    return profile_kernel(module, kernel, variant, batch, dtype,
                          launch_skip=skip, launch_count=measured["stride"],
                          expected_kernels=measured["distinct"],
                          expected_kernel_names=set(measured["names"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=RESULTS_PATH)
    parser.add_argument(
        "--arm", action="append", default=[], metavar="MODULE:KERNEL:VARIANT:BATCH",
        help="profile one per-kernel arm instead of the vit_forward grid; "
             "repeatable, e.g. bench.run_mlp:mlp:triton_fused:128")
    args = parser.parse_args()

    if args.arm:
        total = 0
        for spec in args.arm:
            module, kernel, variant, batch = spec.split(":")
            captured = capture_arm(module, kernel, variant, int(batch))
            record_counters(captured, args.out)
            total += len(captured)
            print(f"  wrote {len(captured)} rows for {kernel}/{variant}")
        print(f"wrote {total} rows total to {args.out}")
        return

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
