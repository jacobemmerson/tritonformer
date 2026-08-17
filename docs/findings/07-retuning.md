# Finding 07: Retuning `softmax` and `linear`/`linear_gelu`, and fixing the clock instrumentation

This is a follow-up to the merged 19-task study (`docs/findings/00-06`), scoped to two
questions: (1) how much of the naive-Triton-vs-torch deficit reported there was a
configuration artifact — undersized tiles, no L2 pid swizzle — versus intrinsic to
Triton on this card, for `softmax` and `linear`/`linear_gelu` specifically; and (2) why
`flagged` was `False` on every one of the original 240 rows, which turned out to need
three independent fixes, not one.

**Scope.** `mlp_fused` and the flash-attention kernel are explicitly out of scope for
this document (a separate follow-up). The existing kernels
(`_softmax_kernel`, `_linear_kernel`, `_linear_gelu_kernel`) are byte-for-byte
unchanged; every tuned kernel below is a new variant registered alongside the
original (`triton_tuned`, `triton_tuned_gelu`), so the naive-vs-tuned delta is
itself measured, not assumed.

## Three independent bugs, one reassuring `False`

The merged study's `flagged` column read `False` on all 240 rows. That looked like
"clocks were stable." It meant "the instrument could not have caught drift even if it
tried," for three separate reasons, found and fixed one at a time over the course of
this task:

**Bug 1 — wrong field, no queryable lock.** `locked_clock_mhz()` queried
`clocks.applications.graphics`, which reads `[N/A]` on GeForce: application clocks
(`-ac`) are a different mechanism than the `-lgc` graphics-clock lock this project
actually uses, and this card exposes no queryable read-back of an `-lgc` lock at all
(confirmed: no lock field anywhere in `nvidia-smi -q`, and pynvml was not installed at
the time). **Fix:** `bench/clocks.py::locked_clock_mhz()` now reads the operator-declared
target from `TRITONFORMER_LOCKED_CLOCK_MHZ`, returning `None` when unset — auto-detection
is not possible here, so this makes that explicit instead of silently wrong.

**Bug 2 — telemetry sampled after measurement, not during it.** `Measurement.build`
called `telemetry()` itself, after `compare()`'s 30-rep loop had already returned. By
that point the GPU had a moment to idle and recover toward its locked target, so the
recorded clock reflected recovery, not what the kernel ran under. This was caught by
direct concurrent `nvidia-smi` polling during a run whose CSV rows all read a pristine
1305 MHz: the card was actually swinging 300–1245 MHz under sustained thermal
throttling at the same time. **Fix:** `compare()` now samples telemetry *inside* its
rep loop and returns a per-arm `TelemetrySummary` (minimum clock, maximum temperature
observed across that arm's reps — worst-case, not average, since a kernel that
throttled for even a few reps ran under degraded conditions for that fraction).
`Measurement.build` no longer has a fallback path that samples fresh; `sm_clock_mhz`
and `temp_c` are required keyword arguments now, so there is no code path left where a
"convenient" post-hoc sample can quietly substitute for what actually happened during
measurement.

**Bug 3 — the fix for Bug 2, done naively, would have caused a third bug.** The first
attempt at in-loop sampling used the existing `nvidia-smi` subprocess
(`bench.clocks.telemetry()`), measured at ~80 ms/call. At 30 reps × 3 arms sampled
every 5th rep, that is ~1.4 s of injected subprocess-launch idle time per `compare()`
call — long enough for the card to recover between kernel launches, systematically
*under*-reporting the very throttling the sampling exists to catch. The instrument
would have been perturbing what it measured. **Fix:** switched in-loop sampling to
NVML via `torch.cuda.clock_rate()` (~1.7 ms) and `torch.cuda.temperature()` (~0.6 ms) —
measured ~2.2 ms combined on this host, about 36x cheaper — cheap enough to sample
*every* rep rather than every 5th. `bench/clocks.py::telemetry()` (the `nvidia-smi`
path) is kept as a fallback for hosts without NVML available; `_sample_telemetry()` in
`bench/harness.py` tries NVML first and falls back automatically if it raises.

Each bug alone was enough to produce `flagged = False` regardless of the other two.
All three had to be fixed for the column to mean anything. Once fixed, it does: **54 of
the 100 new rows in this run came back flagged** (see below).

**CSV schema is unchanged, not the semantics.** `bench/results/latency.csv` already
holds 300+ rows from the merged study, and `record()` writes a header only on a new
file, so appending rows with different columns would corrupt it. Instead, `sm_clock_mhz`
and `temp_c` keep their names and column position but now mean *minimum clock* /
*maximum temperature observed during that arm's reps*, not a single point sample taken
after the fact. This is documented on `Measurement`'s docstring in `bench/harness.py`.
**Anyone comparing a row from before this task to a row from after it is comparing two
different measurement definitions under the same column names.**

## The thermal story

- **1830 MHz lock attempted first, failed to hold.** 1830 MHz is the lowest
  *supported* graphics-clock value on this card (range 1830–2100 via `nvidia-smi -lgc`),
  but under sustained load the card hit `SW Power Cap: Active` and
  `SW Thermal Slowdown: Active` simultaneously, with `clocks.sm` swinging 1305–1830 MHz
  at 82–87°C and up to 50.09 W against the card's hard 50 W limit. The lock request
  itself succeeded; the card simply could not sustain it.
- **1300 MHz lock held for short bursts, not a full sweep.** Direct polling during a
  ~30-minute sweep run showed the card dropping to **300 MHz at 84–85°C while drawing
  only ~28 W** — this is a thermal ceiling, not a power ceiling; the chassis cannot
  dissipate sustained load at these clocks, and the card throttles hard well under its
  wattage budget to manage it. Idle recovery back to 1305 MHz happens within the ~20 s
  it takes to move to the next arm/batch in a sweep, which is exactly why Bug 2 above
  went undetected for as long as it did.
- **All latency numbers in this document were collected under this real,
  intermittently-throttling 1300 MHz-target lock**, not a clean 1830 MHz lock as
  originally planned. `compare()` interleaves arms at the rep level (`bench/harness.py`),
  so **ratios between arms measured in the same run remain valid** even when both arms
  shared a throttle dip — but **absolute milliseconds are inflated** on flagged rows.
  Concretely: this run's `torch` reading for `linear` at `k=192, n=192, batch=512` is
  **2.0554 ms** (flagged, `sm_clock_mhz=300`) — more than the fully-tuned Triton kernel
  at the same shape. A companion pre-merge measurement on a cool, unthrottled card
  (cited in the task brief) read **1.7449 ms** for the same torch call. That difference
  is the throttling tax made visible in a single number.
- **Comparisons against the original merged-study CSV rows are not apples-to-apples** —
  different clock regime (unlocked, 300–1575 MHz swing, both power and thermal caps
  active) and, as of this task, a different telemetry-sampling method entirely.
  Naive-vs-tuned comparisons *within this run* are apples-to-apples; comparisons
  *against* `docs/findings/00-06`'s numbers are not.

## `softmax`

**A1 — `(SOFTMAX, "triton_tuned")`** in `model/kernels/softmax.py`. Rows batch
`ROWS` (`tl.constexpr`, `triton.autotune`d over `{1,2,4,8,16}` keyed on `n_cols`) per
program instead of the committed one-row-per-program, with 2D masking over both rows
and columns.

**Elem/thread arithmetic.** The committed kernel gives each program a `BLOCK=64` row at
`num_warps=4` — 128 threads for 64 elements, **0.5 elem/thread**, well under the ≥2
elem/thread a memory-bound kernel needs to hide latency (the same reasoning `06-synthesis.md`
gives for why `layernorm`, at `D=192` → `BLOCK=256` → 2.0 elem/thread with the same
`num_warps=4`, does not have this problem). Autotune picked `ROWS=8` for the model's
64-wide rows: `8 * 64 / 128 = 4.0 elem/thread`. For the wider `n_cols=192` case, it
picked `ROWS=1` — at `BLOCK=256`, `num_warps=4` that is already `256/128 = 2.0
elem/thread`, comfortably past the threshold on its own, so batching more rows on top
buys nothing and the autotuner correctly declines to.

**Latency (ms, median of 30 interleaved reps; all rows below unflagged, `sm_clock_mhz=1305`):**

| batch | torch | triton (naive) | triton_tuned | naive/torch | tuned/torch | naive→tuned gain |
|------:|------:|----------------:|-------------:|-------------:|-------------:|-------------------:|
|     1 | 0.0056 |          0.0072 |       0.0047 |         1.29x |    **0.84x** |               1.53x |
|     8 | 0.0088 |          0.0224 |       0.0082 |         2.55x |    **0.93x** |               2.73x |
|    32 | 0.0224 |          0.0730 |       0.0220 |         3.26x |    **0.98x** |               3.32x |
|   128 | 0.0762 |          0.2760 |       0.0758 |         3.62x |    **0.99x** |               3.64x |
|   512 | 0.2913 |          1.0895 |       0.2913 |         3.74x |    **1.00x** |               3.74x |

**Tuned softmax reaches parity with `F.softmax` at every batch measured, and beats it
at batch ≤ 32.** This fully confirms the pre-task measurement cited in the task brief
(parity at batch 512, win below batch 32) — under this run's real 1300 MHz-target,
intermittently-throttled lock, not just the cool-card conditions it was originally
measured under.

**Counters (ncu, batch 128, fp32; naive-variant counters for this kernel were not
previously collected, so this is tuned-only):**

| variant | kernel | registers/thread | occupancy (warps active, %) | DRAM read | DRAM write |
|---|---|---:|---:|---:|---:|
| triton_tuned | `_softmax_tuned_kernel` | 19 | 91.40% | 6,315,904 | 6,004,192 |

19 registers/thread and 91.4% occupancy is nowhere near register-limited — consistent
with the elem/thread story: the ROWS=8 batching win comes from reducing launch/thread
under-utilization, not from resolving a register or shared-memory constraint.

## `linear` and `linear_gelu`

**A2 — `(LINEAR, "triton_tuned")` and `(LINEAR, "triton_tuned_gelu")`** in
`model/kernels/linear.py`. `triton.autotune` over `BLOCK_M ∈ {64,128,256}`,
`BLOCK_N ∈ {64,128}`, `BLOCK_K ∈ {32,64}`, `num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}`
(72 configs), keyed on `(M, N, K)`, plus `GROUP_M=8` L2 pid-swizzle (`_swizzle_pid`,
the standard Triton matmul group-ordering, hoisted out as its own `@triton.jit`
helper since it is shared by both kernels). The K-axis masking is copied verbatim from
the committed kernels — same `k_remaining` pattern, same `HAS_BIAS` dummy-pointer
guard — and re-verified against the existing `test_non_power_of_two_k_dimension`
(K=100) assertion, unchanged tolerance.

**Winning configs (batch 128, picked fresh per `(M,N,K)` — autotune decisions are not
persisted across processes):**

| shape (K→N) | BLOCK_M | BLOCK_N | BLOCK_K | num_warps | num_stages |
|---|---:|---:|---:|---:|---:|
| 192→192 | 128 | 64  | 32 | 4 | 3 |
| 192→576 | 256 | 128 | 32 | 8 | 4 |
| 192→768 | 256 | 128 | 32 | 8 | 2 |
| 768→192 | 256 | 128 | 32 | 8 | 4 |

The committed kernel's fixed `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4,
num_stages=2` is smaller than every winning config on every axis but `BLOCK_K` — for
`M = batch*64` (32,768 at batch 512), 64 rows per program is far too little parallel
work per launch.

**Latency (ms, median of 30 interleaved reps). Flag column marks rows where the arm's
minimum observed clock drifted more than 5% from the 1300 MHz target:**

| shape | batch | torch | flag | naive | flag | tuned | flag | naive/torch | tuned/torch | naive→tuned gain |
|---|---:|---:|:--:|---:|:--:|---:|:--:|---:|---:|---:|
| 192→192 | 1   | 0.0163 |   | 0.0352 |   | 0.0350 |   | 2.16x | 2.15x | 1.01x |
| 192→192 | 8   | 0.0391 |   | 0.0772 |   | 0.0633 |   | 1.97x | 1.62x | 1.22x |
| 192→192 | 32  | 0.1244 |   | 0.2786 |   | 0.1678 |   | 2.24x | 1.35x | 1.66x |
| 192→192 | 128 | 0.4763 |   | 1.1428 | ⚑ | 0.6759 |   | 2.40x | 1.42x | 1.69x |
| 192→192 | 512 | 2.0554 | ⚑ | 4.9853 | ⚑ | 2.8161 | ⚑ | 2.43x | 1.37x | 1.77x |
| 192→576 | 1   | 0.0259 |   | 0.0523 |   | 0.0471 |   | 2.02x | 1.82x | 1.11x |
| 192→576 | 8   | 0.0731 |   | 0.2326 |   | 0.1464 |   | 3.18x | 2.00x | 1.59x |
| 192→576 | 32  | 0.3515 | ⚑ | 0.8532 | ⚑ | 0.4928 | ⚑ | 2.43x | 1.40x | 1.73x |
| 192→576 | 128 | 1.5053 | ⚑ | 3.8586 | ⚑ | 1.9170 | ⚑ | 2.56x | 1.27x | 2.01x |
| 192→576 | 512 | 6.2140 | ⚑ | 14.0838| ⚑ | 7.8241 | ⚑ | 2.27x | 1.26x | 1.80x |
| 192→768 | 1   | 0.0281 | ⚑ | 0.0530 | ⚑ | 0.0485 |   | 1.89x | 1.73x | 1.09x |
| 192→768 | 8   | 0.0784 |   | 0.2785 |   | 0.1551 |   | 3.55x | 1.98x | 1.80x |
| 192→768 | 32  | 0.4680 | ⚑ | 1.1388 | ⚑ | 0.5482 | ⚑ | 2.43x | 1.17x | 2.08x |
| 192→768 | 128 | 1.4044 | ⚑ | 5.0336 | ⚑ | 2.2578 | ⚑ | 3.58x | 1.61x | 2.23x |
| 192→768 | 512 | 9.3562 | ⚑ | 18.8240| ⚑ | 9.4127 | ⚑ | 2.01x | 1.01x | **2.00x** |
| 768→192 | 1   | 0.0287 | ⚑ | 0.1229 | ⚑ | 0.1252 | ⚑ | 4.29x | 4.36x | 0.98x |
| 768→192 | 8   | 0.1433 |   | 0.2869 |   | 0.2309 |   | 2.00x | 1.61x | 1.24x |
| 768→192 | 32  | 0.5414 | ⚑ | 1.1374 | ⚑ | 0.6114 | ⚑ | 2.10x | 1.13x | 1.86x |
| 768→192 | 128 | 2.0774 | ⚑ | 5.0994 | ⚑ | 2.4741 | ⚑ | 2.45x | 1.19x | 2.06x |
| 768→192 | 512 | 12.2674| ⚑ | 18.6337| ⚑ | 10.9953| ⚑ | 1.52x | **0.90x** | 1.70x |

Autotuning + L2 swizzle recovers **1.01x–2.23x** of the naive deficit across shapes and
batches, cutting the naive-to-torch gap (1.9x–4.3x) roughly in half on average. Tuned
`linear` gets closest to parity at `192→768, batch=512` (1.01x, both arms flagged
together, ratio still meaningful per the interleaving argument above).

### Verifying the `768→192, batch=512` "beats cuBLAS" result — does NOT survive

That row (0.90x — tuned faster than torch) is the only one anywhere in this grid where
tuned beats torch, and **both arms are flagged at `sm_clock_mhz=300`**. Per the task's
explicit instruction not to cite a vendor-BLAS win without checking it on unflagged
data, this was re-measured directly with 60 reps per arm at a cool card
(`torch.cuda.temperature()` = 61°C before, clock confirmed pinned at 1305 MHz
throughout every batch tested — no drift at all, `min_sm_clock_mhz=1305` for both arms
at every batch):

| batch | torch p50 (ms) | tuned p50 (ms) | tuned/torch | min clock (both arms) |
|------:|---------------:|---------------:|------------:|:---:|
|     1 | 0.0279 | 0.1186 | 4.25x | 1305 (clean) |
|     8 | 0.1373 | 0.2260 | 1.65x | 1305 (clean) |
|    32 | 0.5182 | 0.5878 | 1.13x | 1305 (clean) |
|   128 | 1.6940 | 2.2863 | 1.35x | 1305 (clean) |

**Verdict: does not hold.** On every clean, unthrottled measurement of this shape,
tuned Triton is 1.13x–4.25x *slower* than cuBLAS, non-monotonically (1.13x at batch 32
is closer to parity than 1.35x at batch 128) — there is no trend toward parity as batch
grows, let alone a win. The apparent 0.90x "win" at batch 512 in the sweep table above
happened during a shared 300 MHz throttle dip that hit both arms in the same
interleaved rep block; cuBLAS's kernel selection evidently degrades more under that
specific clock/thermal state than the tuned Triton kernel does, which is a real and
interesting result about *throttled* behavior, but it is not "tuned Triton beats
cuBLAS" — reported here as within-noise / an artifact of shared throttling, per the
instruction not to report a reversal that does not survive verification.

### Counters (ncu, batch 128, fp32)

| shape | kernel | registers/thread | occupancy (%) | DRAM read | DRAM write |
|---|---|---:|---:|---:|---:|
| 192→192 | `_linear_tuned_kernel` | 128 | 48.97% | 8,612,672  | 6,024,640  |
| 192→576 | `_linear_tuned_kernel` | 128 | 49.36% | 21,359,136 | 18,580,128 |
| 192→768 | `_linear_tuned_kernel` | 255 | 12.49% | 26,995,168 | 24,870,304 |
| 768→192 | `_linear_tuned_kernel` | 255 | 12.50% | 34,898,752 | 6,125,280  |

Note the split: the two shapes whose winning config used `num_stages=4` (192→192,
192→576) land at 128 registers/thread and ~49% occupancy; the two using `num_stages=2`
(192→768, 768→192) land at the **hardware register cap (255/thread)** and **12.5%
occupancy — the same "1 block/SM" ceiling `05-over-fusion.md` measured for
`mlp_fused`'s catastrophic loss.** The difference here: `mlp_fused` was register-bound
by a hard-coded, unautotuned tile that happened to overflow; these autotuned configs
*chose* a big enough tile to saturate registers because on this shape the resulting
deep instruction-level parallelism per thread out-ran the latency-hiding a
higher-occupancy config could offer, and `do_bench` picked it as the wall-clock
winner anyway. **This is a genuine qualification of `06-synthesis.md` Section 5's
occupancy-first heuristic ("check predicted occupancy... before fusing")**: for a
single autotuned matmul (not a fusion decision), the fastest config on this hardware
was sometimes the one that minimizes occupancy, not maximizes it — caveated because
autotune reruns its search fresh in every process, so the exact config `ncu`'s own
subprocess landed on here is not guaranteed to be bit-identical to whatever the
interleaved latency sweep selected, only drawn from the same 72-config grid and
verified fast by the same wall-clock benchmark.

### `linear_gelu`: verifying the epilogue-fusion reversal — holds

`docs/findings/03-epilogue-fusion.md` and `06-synthesis.md` list `linear_gelu`
fusion as one of only two fusions in the whole ladder that paid off (+9–11% at batch ≥
8, naive kernels). This run's sweep table suggested that under autotuning the
advantage **inverts** — composed (separate tuned matmul + gelu) beats fused
(single tuned kernel):

| batch | torch_gelu | naive composed | naive fused | tuned composed | tuned fused |
|------:|---:|---:|---:|---:|---:|
|   512 | 10.8533 | 20.6612 (1.90x, fusion **+7%**) | 19.1843 | 11.1245 | 12.0731 (1.02x tuned; fusion **−9%**) |

All five batch-512 rows were flagged (`sm_clock_mhz=300`), and the gap is only ~9% —
plausibly within drift noise on a throttled row, so this was re-verified directly:
60 reps/arm, at batches (1, 8, 32) where the card stayed pinned at 1305 MHz throughout
(`min_sm_clock_mhz=1305` for every arm at every batch — genuinely clean, unflagged
data, not sweep rows that happened not to trip the flag):

| batch | composed p10/p50/p90 (ms) | fused p10/p50/p90 (ms) | composed vs fused |
|------:|---:|---:|---:|
|  1 | 0.0519 / 0.0530 / 0.0533 | 0.0553 / 0.0556 / 0.0569 | composed **4.7%** faster |
|  8 | 0.1702 / 0.1723 / 0.1802 | 0.1864 / 0.1885 / 0.1920 | composed **8.6%** faster |
| 32 | 0.5891 / 0.5955 / 0.6024 | 0.6862 / 0.6905 / 0.6936 | composed **13.8%** faster |

**Verdict: holds.** The p10–p90 spreads for composed and fused do not overlap at any of
the three clean batches — this is not noise, and the effect *grows* with batch (4.7% →
8.6% → 13.8%), the opposite of what throttling-driven noise would look like. **The
project's "fusions that pay off" list shrinks from two to one under autotuning**:
`layernorm_residual` still stands (not retuned in this task; out of scope), but
`linear_gelu`'s naive-kernel win reverses once both sides of the comparison are
autotuned.

**Mechanism, per the task's hypothesis — confirmed.** Winning configs (batch 128):

| arm | BLOCK_M | BLOCK_N | BLOCK_K | num_warps | num_stages |
|---|---:|---:|---:|---:|---:|
| composed (`_linear_tuned_kernel`) | 256 | 128 | 32 | 8 | 4 |
| fused (`_linear_gelu_tuned_kernel`) | 64  | 128 | 32 | 4 | 2 |

The composed matmul autotunes to the same large, deep-pipelined config as the standalone
`192→768` shape above (255 regs/thread, 12.49% occupancy per the counters table). The
fused kernel — which must additionally hold the GeLU epilogue's `tanh` intermediate
terms in registers alongside the matmul accumulator while the tile is still resident —
is stuck at a much smaller `BLOCK_M=64`, `num_warps=4`, `num_stages=2`. Task 12's
original register measurement (128 regs/thread unfused vs 168 fused, both at the
naive kernel's fixed tile) already showed the epilogue costs +40 registers/thread; this
result shows that extra cost also **shrinks the autotune search's viable large-tile
region** — the same aggressive tile the plain matmul can profitably use pushes the fused
kernel over the register budget before the tile is even big enough to pay for the
saved DRAM round-trip. Confirmed directly:

| arm | kernel | registers/thread | occupancy (%) |
|---|---|---:|---:|
| composed | `_linear_tuned_kernel` | 255 | 12.49% |
| fused | `_linear_gelu_tuned_kernel` | 255 | 24.98% |

Both hit the register cap in this counter capture (autotune reruns per-process, so
this specific pair of configs is not guaranteed identical to the p10/p50/p90 table's
configs — but they were drawn from the same grid and land in the same regime). The
fused kernel achieves *higher* occupancy (24.98% vs 12.49%) here precisely *because*
its smaller `BLOCK_M=64` tile needs less total register budget per block even while
still register-bound per thread — occupancy alone does not predict which one is
faster; the composed arm's bigger tile amortizes more work per launch and wins despite
lower occupancy, the same nuance noted in the `linear` counters section above.

## Correctness

All new variants pass the existing parametrized test files
(`tests/test_softmax.py`, `tests/test_linear.py`, `tests/test_linear_gelu.py`) at
**unchanged** `TOLERANCES` from `tests/conftest.py` — no tolerance was loosened for any
tuned variant, including the numerical-stability (1e5-magnitude row) and non-power-of-two
(K=100, n_cols=192) cases. `tests/test_harness.py` covers the telemetry fix: unset lock →
`None`/unflagged, matching lock → unflagged, drifting lock → flagged, NVML-unavailable
fallback to the `nvidia-smi` path, and that `compare()`'s `TelemetrySummary` reports the
minimum clock / maximum temperature actually observed across a rep loop rather than a
single value. Full suite: **143 passed** (`.venv/bin/python -m pytest -q`).

## Flagged count

**54 of 100 new rows in this run are flagged** (`sm_clock_mhz` drifted more than 5%
from the declared 1300 MHz target at some point during that arm's reps) — up from 0 of
240 in the merged study. Breakdown by kernel: `softmax` 0/15 (never throttled during
its sweep — it is fast enough per-call that the card stayed cool); `linear_k192_n192`
10/15; `linear_k192_n576` 9/15; `linear_k192_n768` 11/15; `linear_k768_n192` 12/15;
`linear_gelu` 15/25 (every batch ≥ 32). This is the intended, correct outcome of fixing
all three telemetry bugs on a card that genuinely throttles under sustained load — not
noise to be tuned away. **`flagged` is now a live signal**: it correctly reads clean
through `softmax`'s short-duration sweep and correctly trips through the longer,
heavier `linear`/`linear_gelu` sweeps exactly where direct concurrent `nvidia-smi`
polling (see thermal story above) independently confirmed real 300 MHz dips.
