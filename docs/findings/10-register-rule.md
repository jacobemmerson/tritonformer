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

**Environment.** Branch `feat/retune-kernels`, commit `3eecfd4` (predictions) through the
commit landing this section. `TRITONFORMER_LOCKED_CLOCK_MHZ=1300` exported for every
measurement below. All rows cited here are **unflagged** (`sm_clock_mhz` stayed at 1305,
one exception at 1245 noted below — card ran 72-85 C throughout, well short of the
84 C/300 MHz thermal cliff `07-retuning.md` documented).

### The register sweep (Triton compiled-kernel metadata, no ncu)

`BLOCK_M` in {16, 8, 4, 2} x `num_warps` in {4, 8} x `num_stages` in {1, 2}, same
`_mlp_fused_kernel`, called directly (not through the registry) purely to read
`kernel.n_regs` / `kernel.n_spills` / `kernel.metadata.shared` cheaply before spending
ncu time. `num_stages` had **zero effect** on any of these three numbers at any
`(BLOCK_M, num_warps)` pair — see "why the shared-memory route found nothing new" below.

| BLOCK_M | num_warps | regs/thread | spills | shared bytes | blocks/SM by regs | blocks/SM by shared (64KB) | blocks/SM by max-warps | **predicted occupancy** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 255 | 106 | 51,200 | 2 | 1 | 8 | 12.5% |
| 16 | 8 | 226 (committed `mlp_fused`) | 0 | 51,200 | 1 | 1 | 4 | **25.0%** |
| 8  | 4 | 254 | 0 | 41,984 | 2 | 1 | 8 | 12.5% |
| 8  | 8 | 210 | 0 | 41,984 | 1 | 1 | 4 | **25.0%** |
| 4  | 4 | 242 | 0 | 37,376 | 2 | 1 | 8 | 12.5% |
| 4  | 8 | 150 | 0 | 37,376 | 1 | 1 | 4 | **25.0%** |
| 2  | 4 | 166 | 0 | 35,072 | 3 | 1 | 8 | 12.5% |
| 2  | 8 | 128 (**registered `mlp_fused_lowreg`**) | 0 | 35,072 | 2 | 1 | 4 | **25.0%** |

**The decisive fact:** `blocks/SM by shared` is **1 for every single config in the swept
grid**, at every `BLOCK_M`. Shared-memory usage is set by `BLOCK_D` (256) x `BLOCK_H` (32)
— the w1/w2 tile sizes — which the plan's sweep axes (`BLOCK_M`, `num_warps`) never touch.
Occupancy is therefore **never register-bound anywhere in this grid**: it is
shared-memory-bound at exactly `1 block/SM`, and the resulting occupancy percentage is
set entirely by `num_warps` (12.5% at `num_warps=4`, 25.0% at `num_warps=8`) — not by
`BLOCK_M`, and not by the register count. **P2's target of 50% occupancy at <=128
regs/thread is unreachable anywhere in the plan's sweep grid**, because the register
budget was never the binding constraint past `BLOCK_M<=8` — shared memory was, from the
very first config tried.

### ncu-verified counters (batch 128, fp32, `launch_skip=15` to land on a steady-state
launch past all setup/warmup launches, `expected_kernels` declared)

| variant | kernel | regs/thread | occupancy (ncu) | spill | DRAM read | DRAM write | DRAM total |
|---|---|---:|---:|---:|---:|---:|---:|
| `triton_composed` | `_linear_gelu_kernel` + `_linear_kernel` | 168 / 128 | 37.18% / 47.83% | 0 | 76.6M + 80.2M | 24.9M + 6.3M | **187.9 MB** |
| `triton_fused` (`mlp_fused`, committed) | `_mlp_fused_kernel` | 226 | **25.00%** | 0 | 44.4M | 6.4M | **50.8 MB** (-73.0% vs composed) |
| `triton_fused_lowreg` (`mlp_fused_lowreg`, new) | `_mlp_fused_kernel` | **128** | **25.00%** | 0 | 355.7M | 6.9M | **362.6 MB** (**+92.9%** vs composed, **+614%** vs `triton_fused`) |

ncu occupancy for both `triton_fused` (226 regs) and `triton_fused_lowreg` (128 regs)
lands at exactly the value the shared-memory-bound formula above predicted — confirming
the formula, not just the endpoint.

**Why DRAM traffic exploded rather than staying "unchanged" (this breaks P3's premise
directly).** `_mlp_fused_kernel` re-loads the full `w1`/`w2` tiles for every `BLOCK_M`-row
program — there is no cross-program weight reuse (no L2-resident blocking, no persistent
kernel). Shrinking `BLOCK_M` from 16 to 2 multiplies the grid size 8x, and each of those
8x more programs re-reads the same `w1`/`w2` weight tiles from DRAM independently. The
73% DRAM saving `mlp_fused` banks over `mlp_composed` is not a fixed, register-independent
property of "fusion" — it is a property of the specific tile size chosen, and it inverts
into a **traffic loss** the moment `BLOCK_M` shrinks enough to reduce registers. The
rule's premise that "the traffic saving is already banked and unchanged" **does not hold
for this kernel's register-reduction path.**

### Latency (ms, median of interleaved reps, `bench/harness.compare()`, all rows below
unflagged except one noted)

Official sweep (`bench/run_mlp.py`, written to `bench/results/latency.csv`):

| batch | torch | triton_composed | triton_fused (226 regs) | triton_fused_lowreg (128 regs) | fused/composed | lowreg/composed |
|---:|---:|---:|---:|---:|---:|---:|
| 1   | 0.0559 | 0.1753 | 0.5526 | 1.5127  | 3.15x | 8.63x |
| 8   | 0.2316 | 0.5672 | 2.0544 | 12.0513 | 3.62x | 21.25x |
| 32  | 1.0424 | 2.1714 | 8.1942 | 48.1895 | 3.77x | 22.19x |
| 128 | 3.1888 | 8.6283 | 32.7517| 192.7251| 3.80x | 22.34x |
| 512 | 12.6513| 34.4474| 130.9839| 770.8651| 3.80x | 22.37x (composed row min_clock=1245, still within 5% of 1300, unflagged) |

Supplementary points (same `compare()` harness, batches 8 and 128 only, direct calls to
`_mlp_fused_kernel` at the two intermediate grid configs, not written to `latency.csv`
since they are not registered variants):

| BLOCK_M | num_warps | regs/thread | batch=8 ratio vs composed | batch=128 ratio vs composed |
|---:|---:|---:|---:|---:|
| 16 | 8 | 226 | 3.62x | 3.80x |
| 8  | 8 | 210 | 5.92x | 6.21x |
| 4  | 8 | 150 | 14.17x | 14.88x |
| 2  | 8 | 128 | 21.25x | 22.34x |

**The regs-vs-ratio curve (the experiment's deliverable) is monotonic in the WRONG
direction.** As regs/thread falls 226 -> 210 -> 150 -> 128, the fused/composed latency
ratio does not fall toward 1.0x and cross into a win (P4) — it **rises** from 3.8x to
6.2x to 14.9x to 22.3x. Occupancy is flat at 25.0% across all four points (confirmed by
ncu at the two endpoints). Register count went down, occupancy did not move, and latency
got dramatically worse, tracking the DRAM-traffic explosion almost exactly (a ~7.1x DRAM
increase from `triton_fused` to `triton_fused_lowreg` roughly matches the ~5.9x latency
ratio increase at batch 128).

### 1d: the shared-memory route

Two things were tried:

1. **`num_stages` in {1, 2}** across the full grid above: identical `n_regs`,
   `n_spills`, and `shared` bytes at every `(BLOCK_M, num_warps)` pair — `num_stages` had
   no effect at all. Mechanism: the H-loop (`for h_start in range(0, H, BLOCK_H)`) is a
   Python-level loop unrolled at compile time by Triton, not a `tl.range`-style
   pipelined loop construct with loop-carried software-pipelining hints, so there is no
   loop for `num_stages`' double-buffering to apply to.
2. **Explicit manual shared-memory staging of the hidden tile:** not attempted as a
   working kernel, because it is not expressible. `triton.language` 3.6.0 exposes no
   shared-memory allocation primitive (`python -c "import triton.language as tl;
   print([n for n in dir(tl) if 'shared' in n.lower()])"` -> `[]`). Triton's compiler
   already places `tl.dot`'s operands in shared memory automatically when profitable;
   there is no user-facing lever to force staging beyond tile-size choice (`BLOCK_D`,
   `BLOCK_H`) and `num_stages`, both already exhausted above.

**This is itself informative for the project's other headline** ("shared memory bounds
fusion first, at compile time; registers collapse occupancy second"). Here shared memory
bounds occupancy *at runtime*, not compile time — the kernel compiles fine at every
`BLOCK_M`/`num_warps` combination tried — but it still bounds occupancy first: shared
memory pins every config in the grid to the same 1-block/SM ceiling before the register
budget ever becomes the active constraint. Registers only decided *how many spare
registers* each config had left over, never how many blocks fit.

## Verdict on each prediction

- **P1 (BLOCK_M 16->8->4 monotonically reduces regs/thread): HOLDS.** Confirmed at both
  `num_warps=4` (255->254->242->166 down to `BLOCK_M=2`) and `num_warps=8`
  (226->210->150->128).
- **P2 (<=128 regs/thread reaches >=50% occupancy): FAILS.** Occupancy is pinned at
  25.0% (ncu-confirmed) at 128 regs/thread — identical to the 226-regs baseline. Shared
  memory, not registers, sets the blocks/SM ceiling throughout the swept grid; the
  register budget was never binding past `BLOCK_M<=8`.
- **P3 (>=50% occupancy makes fused beat composed): UNTESTABLE as stated — 50% occupancy
  was never reached — but the underlying claim (73% DRAM saving is "unchanged" as
  registers drop) independently FAILS.** DRAM traffic at 128 regs/thread is 362.6 MB,
  a 92.9% *increase* over `triton_composed`'s 187.9 MB, not the 73% *decrease*
  `triton_fused` banks. The traffic saving is a property of `BLOCK_M`, not of "fusion"
  in the abstract, and it evaporates in the exact direction (`BLOCK_M` shrinking) the
  experiment needed to reduce registers.
- **P4 (a threshold between 128 and 226 regs crosses from loss to win): FAILS
  outright.** The measured curve is monotonic in the opposite direction: fused/composed
  ratio *worsens* from 3.8x to 22.3x as regs/thread falls from 226 to 128. There is no
  crossing; the fusion moves further from winning, not closer, at every step.

## The rule is incomplete, stated plainly

**The MLP never flips, at any register count tried, in either direction the plan
considered.** Cutting `mlp_fused`'s registers 226 -> 128 (a 43% reduction, `BLOCK_M`
16 -> 2 at `num_warps=8`) does not restore occupancy (flat at 25.0%, ncu-confirmed) and
makes the fusion **5.9x worse** (3.8x -> 22.3x loss vs composed at batch 128), because
`_mlp_fused_kernel`'s tiling ties `BLOCK_M` to weight-tile reuse: shrinking it to save
registers multiplies redundant `w1`/`w2` DRAM reads across many more grid programs,
destroying the traffic saving the whole rule depends on staying fixed.

The register rule ("fusion pays when it removes a memory round-trip without spending
registers") is **not wrong about `layernorm_residual`** — that kernel's register count
and its DRAM traffic are genuinely independent, so cutting one doesn't touch the other.
It is **incomplete as a general predictive rule** because it implicitly assumes register
pressure and DRAM traffic are independent levers that can be tuned separately. For
`_mlp_fused_kernel`'s design, they are coupled through `BLOCK_M`: the same knob that
buys registers back also buys back the DRAM traffic the fusion existed to save. A
corrected version of the rule needs a second clause: *fusion pays when it removes a
memory round-trip without spending registers, and the mechanism used to reclaim
registers does not itself reopen the round-trip it saved.*

A second, independent finding here: **shared memory, not registers, was the actual
occupancy-limiting resource for `_mlp_fused_kernel` at every config tried** (25.0%
ceiling, invariant to `BLOCK_M`, set by `BLOCK_D` x `BLOCK_H` tile sizes and `num_warps`
alone). This reconciles with, rather than contradicts, the project's other headline
("shared memory bounds fusion first... registers collapse occupancy second") — here
shared memory bounds *occupancy* first, not compile-time viability, and the register
axis this experiment was built to test turns out not to have been the load-bearing
constraint at all. Reducing `BLOCK_H` (kept fixed per the plan's "keep the H-loop
structure" instruction) is the untested lever that would actually move the shared-memory
ceiling; that is the natural next experiment, not a further register cut.

## Correctness and test summary

`tests/test_mlp.py`: 14 passed, including the new
`test_fused_lowreg_matches_composed_exactly_enough` at the same `rtol=atol=1e-5` used for
`test_fused_matches_composed_exactly_enough` — `mlp_fused_lowreg` is numerically correct,
just far slower. Full suite: **148 passed** (143 pre-existing + 5 new: 3 `VARIANTS`
parametrizations x 1 new variant across two tests, plus the new exact-match test).
No tolerance was loosened anywhere.

## Flagged counts

24 total measurement rows collected for this experiment (20 official sweep rows in
`latency.csv` + 4 supplementary curve points): **0 flagged.** Card stayed at 72-85 C,
1245-1305 MHz throughout (one row at 1245 MHz, within 5% of the 1300 MHz target and
correctly not flagged) — well short of the 84 C/300 MHz thermal cliff documented in
`07-retuning.md`. This experiment's short per-call kernels (sub-second even at the
slowest config) apparently did not sustain load long enough to trigger the throttle this
card is prone to under `linear`/`linear_gelu`'s longer sweeps.

## Experiment 1b: the real flip test — pre-registered predictions

Written and committed **before any measurement in this sub-experiment is taken**.
Experiment 1 above found that occupancy was pinned at 25.00% for `_mlp_fused_kernel`
at every `BLOCK_M` tried, because shared memory — not registers — set the blocks/SM
ceiling: `w1`/`w2` tiles at `BLOCK_D=256, BLOCK_H=32` cost 65,536 B, exactly the 64 KB/SM
budget, and that figure is `BLOCK_M`-independent. Cutting `BLOCK_M` also multiplied the
grid, which multiplied redundant `w1`/`w2` re-reads and inverted the fusion's DRAM
saving from -73% to +92.9% — a confound Experiment 1 did not intend and that alone could
explain why it never flipped, independent of occupancy.

This sub-experiment holds `BLOCK_M=16` fixed (the committed `mlp_fused` value) so the
grid size, and therefore the DRAM traffic comparison, stays comparable to `mlp_fused`,
and instead sweeps `BLOCK_H` — the lever that actually sets the shared-memory tile size:

```
BLOCK_H = 32 (current):  65,536 B -> 1 blk/SM ->  25% occupancy |  24 H-loop iterations
BLOCK_H = 16          :  32,768 B -> 2 blk/SM ->  50% occupancy |  48 H-loop iterations
BLOCK_H =  8          :  16,384 B -> 4 blk/SM -> 100% occupancy |  96 H-loop iterations
BLOCK_H =  4          :   8,192 B -> 8 blk/SM -> (warp-capped)  | 192 H-loop iterations
```
(sm_75 caps at 32 warps/SM; at `num_warps=8` that is 4 blocks/SM, so 100% is the ceiling
`BLOCK_H=8` should already reach, and `BLOCK_H=4` cannot exceed it by the warp count even
though the shared-memory arithmetic alone would allow 8 blocks/SM.)

### Pre-registered predictions

- **Q1.** Occupancy rises as `BLOCK_H` falls, tracking the table above: ~50% at
  `BLOCK_H=16`, ~100% at `BLOCK_H=8` (warp-capped, not shared-memory-capped at that
  point).
- **Q2.** DRAM traffic stays close to `mlp_fused`'s -73% vs. `mlp_composed`, because
  `BLOCK_M=16` is unchanged and the grid does not multiply — this is the confound
  Experiment 1's `BLOCK_M` cuts destroyed, held fixed here on purpose.
- **Q3.** There exists a `BLOCK_H` at which the fused MLP's latency beats `mlp_composed`
  — the flip Experiment 1 was looking for and did not find.
- **Q4.** If occupancy rises substantially (Q1 holds) AND the traffic saving is retained
  (Q2 holds) but the fusion still never beats composed (Q3 fails), then neither registers
  nor occupancy is the binding constraint on this kernel at all, and the real cost is the
  H-loop's own serialisation: `_mlp_fused_kernel`'s H-loop is a compile-time-unrolled
  Python `for`, not a `tl.range`-pipelined loop (Experiment 1 already confirmed
  `num_stages` has zero effect on `n_regs`/`n_spills`/`shared` for that reason). Smaller
  `BLOCK_H` trades occupancy for more sequential, loop-carried iterations (24 -> 48 -> 96
  -> 192) with no software pipelining to overlap them. This is tested directly by also
  sweeping `num_stages` in {1, 2, 3} at the best `BLOCK_H`: if pipelining helps once the
  loop has enough iterations to pipeline, latency should improve with `num_stages`; if it
  still does nothing, the loop-unrolling structure itself — not merely its length — is
  the obstacle.

### What each prediction refutes if it fails

- If **Q1 fails** (occupancy does not track the shared-memory table), shared memory is
  not the binding occupancy constraint either, contradicting both Experiment 1's finding
  and the project's headline claim that shared memory bounds fusion first.
- If **Q2 fails** (traffic saving erodes even with `BLOCK_M` fixed), the -73% figure is
  not `BLOCK_H`-independent either, and the traffic accounting needs revisiting beyond
  the `BLOCK_M`-tied redundant-read mechanism Experiment 1 already identified.
- If **Q3 holds**, the register rule's mechanism (occupancy, once uncoupled from the
  DRAM-traffic confound) is vindicated for this kernel, and Experiment 1's failure to
  flip is explained as an artifact of sweeping the wrong knob (`BLOCK_M` instead of
  `BLOCK_H`).
- If **Q3 fails despite Q1 and Q2 holding, Q4 is the result**: the project's
  shared-memory headline is incomplete in the same way the register rule was —
  occupancy is necessary but not sufficient, and the real limiter is loop-carried
  dependency depth in the H-loop, unrelated to any resource-occupancy accounting. This is
  the interesting branch and will be reported plainly, not softened, if it is what the
  data shows.

### Method

- New variant `mlp_fused_blockh` in `model/kernels/mlp.py`, append-only.
  `_mlp_fused_kernel`, `mlp_fused`, and `mlp_fused_lowreg` are not modified — `git diff
  --numstat model/kernels/mlp.py` must show 0 deletions for this sub-experiment.
- Sweep `BLOCK_H` in {16, 8, 4} at `BLOCK_M=16` (fixed), then `num_stages` in {1, 2, 3} at
  whichever `BLOCK_H` from that sweep gives the best latency.
- GeLU constants (`0.7978845608028654`, `0.044715`), `libdevice.tanh`, the fp32
  accumulator, and bias handling (`b1` indexed per H-chunk inside the loop, `b2` added
  once outside it) are preserved exactly from `_mlp_fused_kernel`.
- Correctness: `tests/test_mlp.py`, unchanged `TOLERANCES["mlp"]`, including a new
  exact-match test against `mlp_composed` at `rtol=atol=1e-5`, matching the existing
  `test_fused_matches_composed_exactly_enough` / `test_fused_lowreg_matches_composed_exactly_enough`
  pattern. No tolerance is loosened.
- Counters via `bench/collect_counters.py` / `bench/profile.py::profile_kernel` per
  `BLOCK_H`; latency via `bench/run_mlp.py`'s sweep machinery.
  `TRITONFORMER_LOCKED_CLOCK_MHZ=1300` exported for every measurement. Flagged counts
  reported per the live `flagged` signal.
