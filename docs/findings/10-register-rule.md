# Finding 10: Experiment 1 (The Flip) — pre-registered predictions

This document is created and committed **before any measurement in this experiment
is taken**. Its purpose is to pin down falsifiable predictions so the eventual
result cannot be retrofitted into a narrative. See
`docs/superpowers/plans/2026-08-17-testing-the-register-rule.md`, Experiment 1, and
`docs/findings/07-retuning.md` for the register rule this builds on.

## The rule under test

> Fusion pays when it removes a memory round-trip without spending registers.

`mlp_fused` (`_mlp_fused_kernel` in `model/kernels/mlp.py`, committed config
`BLOCK_M=16, BLOCK_D=next_power_of_2(192)=256, BLOCK_H=32, num_warps=8,
num_stages=1`) already banks a 73% DRAM traffic reduction over `mlp_composed` and
still loses 3.10x-3.83x, measured at 226 regs/thread x 256 threads/block =
57,856 regs/block -> 1 block/SM -> 25.00% occupancy (`docs/findings/05-over-fusion.md`).

The rule predicts that cutting register pressure enough to restore occupancy should
flip the result to a win, since the traffic saving is already banked and unchanged
by the register count. Target: <=128 regs/thread at `num_warps=4` gives
128 x 128 = 16,384 regs/block -> 4 blocks/SM -> 50% occupancy, matching the
committed `_linear_kernel`'s regime from `07-retuning.md`.

## Pre-registered predictions

- **P1.** Reducing `BLOCK_M` 16 -> 8 -> 4 monotonically reduces regs/thread for
  `_mlp_fused_kernel`'s structure (same `_mlp_fused_lowreg_kernel`, `BLOCK_H` and
  `num_warps` held fixed across the comparison).
- **P2.** At <=128 regs/thread, the fused MLP reaches >=50% occupancy
  (`sm__warps_active.avg.pct_of_peak_sustained_active`).
- **P3.** At >=50% occupancy, the fused MLP BEATS `mlp_composed` in latency, because
  its 73% DRAM traffic saving is unchanged while the occupancy penalty that was
  suppressing it is gone.
- **P4.** There exists a register threshold strictly between 128 and 226 regs/thread
  where the fused/composed latency ratio crosses from >1 (loss) to <1 (win).

## What would falsify the rule

If regs/thread drops to <=128, occupancy rises to >=50% (P1+P2 hold), and the fused
kernel **still loses** to composed (P3 fails), the register rule as stated is
incomplete: register-driven occupancy is not sufficient to predict the fusion's
win/loss outcome, and some other factor (H-loop sequential iteration count,
absence of `num_stages` double buffering, or arithmetic intensity too low to
amortize regardless of occupancy) must be doing the work instead. That outcome
will be reported plainly, not softened.

## Method

- New variant `mlp_fused_lowreg` registered alongside `mlp_fused` in
  `model/kernels/mlp.py`, append-only. `_mlp_fused_kernel` and `mlp_fused` are not
  modified.
- Sweep `BLOCK_M` in {8, 4, 2} x `num_warps` in {4, 8}, `BLOCK_H=32` held fixed
  (same H-loop structure as the committed kernel).
- Correctness: `tests/test_mlp.py`, unchanged `TOLERANCES["mlp"]`
  (`rtol=1e-4, atol=1e-4`), including `test_fused_matches_composed_exactly_enough`
  at `rtol=atol=1e-5`. No tolerance is loosened for any config; a config that needs
  slack is reported as broken, not weakened.
- Counters via `bench/collect_counters.py` / `bench/profile.py::profile_kernel`,
  declaring `expected_kernels` explicitly.
- Latency via `bench/run_mlp.py`'s sweep machinery, `TRITONFORMER_LOCKED_CLOCK_MHZ=1300`
  exported for every measurement run. Flagged counts reported per the live
  `flagged` signal from `07-retuning.md`.
- Second route (1d): stage the hidden `[BLOCK_M, BLOCK_H]` tile through shared
  memory instead of registers, as an independent way to hit the same occupancy
  target. If it fails to compile, the exact error and configs tried are recorded.

## Results

(Filled in after measurement — see the rest of this document below this line, and
`.superpowers/sdd/2026-08-17-retune/experiment-1-report.md` for the full writeup.)
