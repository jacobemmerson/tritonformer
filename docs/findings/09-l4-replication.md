# Finding 09: Experiment 3a — pre-registered L4 predictions

This document is created and committed **before any measurement on the L4 in this
experiment is taken**. Its purpose is to pin down falsifiable predictions so the eventual
result cannot be retrofitted into a narrative. See
`docs/superpowers/plans/2026-08-17-testing-the-register-rule.md`, Experiment 3, and
`docs/findings/10-register-rule.md` (Experiments 1 and 1b) for the rule and the mechanism
this replication tests.

## The rule under test, and where it stands going into this experiment

All prior measurement in this project was taken on a GTX 1650 Ti (sm_75, 16 SMs, ~1 MB
L2, no tensor cores). The rule derived there, in its original form:

> **Fusion pays when it removes a memory round-trip without spending registers.**

Experiment 1 (`10-register-rule.md`) falsified that rule's mechanism, not its headline
observation: cutting `mlp_fused`'s registers via `BLOCK_M` left occupancy pinned at
25.00% and made latency *worse*, because registers and DRAM traffic are coupled through
`BLOCK_M` on this kernel. Experiment 1b then cut shared memory directly via `BLOCK_H`,
holding `BLOCK_M` fixed to remove that confound, and got the same null result: occupancy
still did not move (still 25.00%, ncu-confirmed), the DRAM traffic saving held exactly
steady (-72.9% both configurations), and the fusion still lost — worse, in fact, tracking
the H-loop's iteration count (24 -> 48). The reconciled headline as of commit `3e28816`:

> Shared memory sets the compile-time feasibility bound, but the mega-MLP's actual
> latency loss comes from the H-loop's unpipelined serial reduction, not from register or
> occupancy collapse.

So going into this experiment, **occupancy is already known not to be the binding
mechanism for the mega-MLP fusion on sm_75.** What Experiment 3 asks is a different
question: is any of this sm_75-specific? A 16-SM, ~1 MB-L2, no-tensor-core laptop card is
an unusual profile; the L4 (sm_89) is close to the opposite on every axis that plausibly
matters.

## Hardware delta being tested

| | GTX 1650 Ti (sm_75) | L4 (sm_89) |
|---|---|---|
| SMs | 16 | 58 |
| L2 | ~1 MB | ~48 MB |
| bandwidth | ~192 GB/s | ~300 GB/s |
| shared mem / block | 64 KB | ~99 KB |
| registers / SM | 65,536 | 65,536 (same) |
| tensor cores | none | 4th gen |
| cooling | 50 W laptop, throttles | datacenter, no throttle |

## Pre-registered predictions

1. **The rule's premise may collapse.** A `[128,3,64,64]` fp32 intermediate is 6.29 MB
   (`128 x 3 x 64 x 64 x 4 = 6,291,456 B`; **corrected from an earlier draft's 12.6 MB
   before any L4 measurement was taken** — the original figure appears to have costed a
   write+read round trip rather than the tensor's own size, an error traced to the
   pre-registration's source plan) and fits in 48 MB of L2, so the unfused arm may never
   reach DRAM. If so, **even register-free fusion stops paying** — which would make
   `layernorm_residual`'s 22-25% win an artifact of a small-L2 card. Testable:
   `dram__bytes_read.sum` for composed arms should collapse. The correction does not
   change the prediction's direction: 6.29 MB fits inside 48 MB of L2 even more
   comfortably than 12.6 MB did, so if anything the prediction is strengthened, not
   weakened, by the fix.
2. **Register arithmetic transfers unchanged.** Same 65,536 regs/SM, so `mlp_fused` at
   226 regs x 256 threads -> 1 block/SM -> 25% should reproduce exactly.
3. **The matmul comparison needs controlling.** cuBLAS gets tensor cores; our fp32
   `tl.dot` does not. Report strict-fp32 AND TF32-enabled numbers, or the comparison
   measures precision policy rather than kernel quality.
4. **The monolithic block kernel still fails to compile.** It needs 262,144 B; Ada allows
   ~99 KB/block. The bound moves, it does not disappear.

### Occupancy denominator differs between the two cards — read before scoring prediction 2

| | sm_75 (Turing) | sm_89 (Ada) |
|---|---|---|
| max resident warps / SM | 32 | 48 |
| max resident threads / SM | 1,024 | 1,536 |
| 32-bit registers / SM | 65,536 | 65,536 (unchanged) |

`mlp_fused` at 226 regs/thread x 256 threads/block = 57,856 regs/block; `65,536 //
57,856 = 1` block/SM on **both** cards — the register-file arithmetic that decides block
count genuinely transfers, because it depends only on the register file size, which is
unchanged. But the occupancy *percentage* does not, because it divides by max resident
warps/SM, which differs: `1 block x 8 warps / 32 = 25.00%` on sm_75, versus `1 block x 8
warps / 48 = 16.67%` on sm_89. The same register-driven *block-count* math applies across
both cards; the occupancy *percentage* denominator does not.

Prediction 2 is kept **verbatim** above and will be scored as literally written — "25%
should reproduce exactly" is expected to read FALSE on the L4, landing at 16.67% instead.
Scoring it must separate two distinct claims bundled in that one sentence: (a) **"1
block/SM transferred"** — the substantive claim about the register file, expected to
hold — from (b) **"the occupancy percentage matched 25%"** — a denominator-dependent
figure expected to read ~16.67% instead. A pre-registered alternative (16.67%) is named
here so the eventual result lands on a number that was predicted in advance, not a
post-hoc explanation invented after seeing the measurement.

## Pre-registered expectation (not a fifth prediction, not scored)

`F.scaled_dot_product_attention` will select a different backend on sm_89 than it did on
sm_75. On the 1650 Ti it selected `fmha_cutlassF_f32_aligned_64x64_rf_sm75`, which already
beat our hand-written flash kernel by 3.26x. Real FlashAttention kernels are available on
sm_89 (Ada); which backend SDPA actually picks there will be recorded as an observation,
not scored as held/broken.

## Known measurement risk — stated before any measurement

`ncu` requires `NVreg_RestrictProfilingToAdminUsers=0` set at the host kernel-module level
(see `bench/profile.py:8`). Modal's shared GPU hosts are not expected to grant this, so
`ERR_NVGPUCTRPERM` is a live possibility. **Prediction 1 is the only one of the four that
depends on hardware counters** (`dram__bytes_read.sum` via ncu). If counters are
unavailable when this experiment runs, prediction 1 will be recorded **UNRESOLVED —
neither held nor broken** — it will not be quietly re-scored against weaker evidence
(e.g. latency alone) to manufacture a verdict.

Predictions 2, 3 and 4 do not depend on ncu:
- Prediction 2's registers and spills come from Triton's compiled kernel metadata
  (`n_regs` / `n_spills`), and occupancy from arithmetic on those, the same as
  `10-register-rule.md`'s method.
- Prediction 3 is pure latency, no counters needed.
- Prediction 4 is a compile attempt: it either compiles or raises, not a runtime measurement.

Declaring this in advance is what stops a counter-permission failure from being
retrofitted into a narrative later — e.g. treating a missing prediction 1 result as if it
had supported whichever direction the other three predictions leaned.

## What happens next

Experiment 3's measurement phase (Experiment 3b) runs on Modal against an L4, scores each
of the four predictions above against what was actually observed, and reports the
SDPA-backend expectation. No GPU time is spent until this document is committed.

---

# Results

Measured on Modal against an NVIDIA L4 (sm_89), source tree at commit `936b4f3` — the
pre-registration commit above, so no kernel changed between predicting and measuring.
Raw measurement record: `.superpowers/sdd/2026-08-17-testing-the-register-rule/task-2-report.md`.
Rows landed in `bench/results/latency.csv` (220 new) and `bench/results/counters.csv`
(154 new, of which 77 are the superseded first capture — see below), distinguished by the
`gpu` column (`NVIDIA L4`).

## Environment

```
device_name        NVIDIA L4
compute_capability (8, 9)
sm_count           58
total_memory_GB    23.66
shared_mem_per_block_B    49,152   (default)
shared_mem_per_block_optin_B 101,376
l2_cache_B         50,331,648
torch              2.11.0+cu128
triton             3.6.0
ncu                2025.1.1.0
matmul_allow_tf32  False
cudnn_allow_tf32   True
```

**Provenance of the two `allow_tf32` lines, because they carry weight below.** They were
printed by the `smoke` invocation's `_describe_device()`, not by the sweep process itself.
They are the container image's process defaults, and no measurement body other than
`tf32_body` ever writes those flags — `sweep_body`, `counter_grid_body` and `sdpa_body`
leave them alone, and `tf32_body` restores `False` before returning. So the sweeps ran at
`matmul_allow_tf32 = False`, on the evidence of the same image's recorded default rather
than a read-back inside the sweep process. That is good evidence and not a direct
observation; the distinction is kept wherever the value is load-bearing.

Torch and Triton match the 1650 Ti host exactly. Python differs (3.13.3 in the container
vs 3.14.4 locally) because Modal's `debian_slim` offers no 3.14; the pinned wheels are the
same builds. Full suite on the L4: **153 passed**, matching sm_75 test for test — but only
under `TRITON_F32_DEFAULT=ieee`, which is a finding in itself and is reported below.

**Counters were available.** The pre-registered risk of `ERR_NVGPUCTRPERM` did not
materialise: Modal's L4 grants counter permission. What failed was ncu's attempt to lock
GPU clocks, worked around with `--clock-control none` through a new opt-in env var
(`TRITONFORMER_NCU_CLOCK_CONTROL`, commit `210c17f`; unset reproduces the exact ncu command
every sm_75 measurement used). **Prediction 1 was therefore measured, not recorded
UNRESOLVED**, and an earlier ruling that pre-authorised an UNRESOLVED verdict for it is
void.

## Measurement-regime warning — read this before comparing any two rows

`bench/results/latency.csv` now holds rows from at least three measurement regimes. The
`gpu` column is what separates the L4 rows; nothing in the file separates the first two.

| regime | clock lock | `flagged` | note |
|---|---|---|---|
| merged study (findings 01-06) | none | dead by construction | `TRITONFORMER_LOCKED_CLOCK_MHZ` unset, clocks 300-1575 MHz |
| retune (findings 07-10) | 1300 MHz declared | live | ~60% of heavy rows drifted |
| L4 (this finding) | none | dead by construction | lock correctly unset per plan; clocks 660-2040 MHz |

`sm_clock_mhz` and `temp_c` also changed meaning between the first two regimes, from a
point sample to min/max observed. Anyone comparing rows across regimes without knowing
this will draw false conclusions.

### Throttle disclosure (required before quoting any cross-card ratio)

| card | rows | flagged | `sm_clock_mhz` range |
|---|---:|---:|---|
| GTX 1650 Ti | 405 | **60 (14.8%)** | 300-1950 |
| NVIDIA L4 | 220 | 0 | 660-2040 |

**`flagged=False` means different things on the two cards.** On the L4 no clock lock was
declared, so `bench/clocks.py::locked_clock_mhz()` returns `None` and the drift test never
ran: all 220 rows are unflagged *by construction, not by measurement*, while the card's
clocks in fact vary by ~3x. On the 1650 Ti `flagged=True` is a real signal and fires on
14.8% of rows.

Consequences, applied rather than re-measured (re-measurement was ruled out on budget):

- **Six `linear_gelu` cross-card speedups are WITHDRAWN.** Their sm_75 side ran at
  300 MHz, ~23% of nominal; they measure a throttled laptop card, not a card difference.
- `mlp` @ 512 (`torch`, `triton_composed`) and four `vit_forward` cells (`torch_compile`
  @ 128 and @ 512, `torch` @ 512, `triton_composed` @ 512) are throttle-inflated in the
  L4's favour on the sm_75 side. Treat their magnitudes as upper bounds.
- **Large-margin rank conclusions survive**, because a throttled sm_75 side can exaggerate
  an L4 advantage but cannot invent an ordering *within* the sm_75 card. Tight ratios from
  flagged cells should not be quoted.

### Counter comparability

L4 counter rows were gathered with clocks unfixed; sm_75 counter rows were gathered with
ncu's default clock locking (the earlier claim that the sm_75 *latency* rows were locked at
1300 MHz is false — that applied to counter collection and to campaigns that declared the
lock). **Byte, sector and warp counts are frequency-invariant and remain comparable across
the two cards; nothing rate- or duration-shaped is.** Every cross-card counter figure below
is a byte count, for that reason. One asymmetry to state plainly: only the L4 side is
kernel-identity-validated. The sm_75 rows came from earlier campaigns whose capture windows
were chosen per experiment, so they are *intended*-equivalent rather than produced by
identical driver code.

## Prediction 1 — BROKE

> **1.** The rule's premise may collapse. [...] a `[128,3,64,64]` fp32 intermediate is
> 6.29 MB and fits in 48 MB of L2, so the unfused arm may never reach DRAM. If so, **even
> register-free fusion stops paying** [...] Testable: `dram__bytes_read.sum` for composed
> arms should collapse.

`dram__bytes_read.sum`, batch 128, fp32, **per one steady-state call**, L4 rows under
`TRITON_F32_DEFAULT=ieee`:

| fusion | sm_75 composed | sm_75 fused | sm_75 advantage | L4 composed | L4 fused | L4 advantage |
|---|---:|---:|---:|---:|---:|---:|
| `layernorm_residual` | 18,905,696 | 12,599,456 | **1.50x** | 21,093,888 | 13,609,088 | **1.55x** |
| `attention` | 52,252,608 | 21,183,232 | **2.47x** | 43,265,664 | 21,370,624 | **2.02x** |
| `mlp` | 156,931,392 | 44,364,432 | **3.54x** | 40,823,680 | 8,262,528 | **4.94x** |

The same data as the cross-card change per arm:

| fusion | composed arm, L4 vs sm_75 | fused arm, L4 vs sm_75 |
|---|---:|---:|
| `layernorm_residual` | **+11.6%** | +8.0% |
| `attention` | -17.2% | +0.9% |
| `mlp` | -74.0% | -81.4% |

**BROKE on both halves.**

- **The composed arms did not collapse as a class.** `layernorm_residual`'s composed arm
  read **11.6% more** on the L4; `attention`'s fell 17.2%; only `mlp`'s fell substantially
  (-74.0%).
- **Fusion did not stop paying.** Its DRAM advantage persisted on every rung, at the same
  order of magnitude: 1.50x -> 1.55x, 2.47x -> 2.02x, 3.54x -> 4.94x.

**The buried assumption that failed**, and the part worth carrying forward: the prediction
assumed a bigger cache would preferentially help the arm with an intermediate to absorb.
It did not. On the one rung where traffic fell substantially (`mlp`) *both* arms fell, the
fused one further (-74.0% composed, -81.4% fused), so the ratio survived; on the other two
rungs four of the four arms moved by -17.2% to +11.6%, i.e. not in the direction or
magnitude a cache-absorption story predicts, and three of the six measured arms read
**more** DRAM on the 48 MB-L2 card than on the ~1 MB one.

**No mechanism is offered for why the ratio survived, because none was measured.** "A
larger L2 helps both arms" would be a story fitted to one rung and contradicted by the
other two. What the counters establish is the negative result — the composed arms did not
collapse and fusion kept paying — not the reason for it. `mlp`'s fused arm in particular
fell **-81.4%** while having no intermediate to absorb at all; that collapse is
unaccounted for and is listed with this document's other unexplained results below.

`layernorm_residual` — the rung the prediction explicitly named, whose 22-31% win it
proposed to reveal as a small-L2 artifact — is the rung that moved **least** (1.50x ->
1.55x). **The "artifact of a small-L2 card" hypothesis is unsupported for its own target
rung.**

This is a broken prediction, not a reversal. Nothing inverted.

## Prediction 2 — split verdict: 1 block/SM transferred, the numbers in the sentence did not

> **2. Register arithmetic transfers unchanged.** Same 65,536 regs/SM, so `mlp_fused` at
> 226 regs x 256 threads -> 1 block/SM -> 25% should reproduce exactly.

`mlp_fused`'s committed configuration (`BLOCK_M=16, BLOCK_H=32, num_warps=8`), from
Triton's compiled-kernel metadata. Device properties read live on each card.

| card | precision | n_regs | spills | shared B | regs/block | blocks/SM by regs | blocks/SM by shared | blocks/SM | occupancy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1650 Ti (sm_75) | ieee (only option) | **226** | 0 | 51,200 | 57,856 | 1 | 1 | **1** | **25.00%** |
| L4 (sm_89) | ieee | **210** | 0 | 51,200 | 53,760 | 1 | 2 | **1** | **16.67%** |
| L4 (sm_89) | tf32 | 106 | 0 | 51,200 | 27,136 | 2 | 2 | 2 | 33.33% |

Scored as the three claims the one sentence bundles:

1. **1 block/SM transferred — HELD** (under IEEE, the precision that matches the sm_75
   record). The register file is a binding constraint on both cards.
2. **226 regs/thread did not reproduce — BROKE.** The same kernel source compiles to
   **210** regs/thread on the L4 under IEEE. Register count is a codegen output, not a
   property of the source, and it moved with the backend.
3. **25.00% did not reproduce — BROKE, at exactly the pre-registered alternative.**
   Measured **16.67%** = 8 warps / 48, the sm_89 denominator named in advance above. The
   register arithmetic that decides *block count* transferred; the *percentage* in the
   prediction's wording was an sm_75 bookkeeping artifact.

This is neither a pass nor a fail. The substantive claim held and the stated numbers did
not, and the prediction's wording conflated them — which is a result about how the
prediction was written, and is reported rather than edited away.

**A fourth result, not pre-registered at all: shared memory stopped binding.** On sm_75
the register file and shared memory were **co-binding** — the table shows `blocks/SM by
regs = 1` *and* `by shared = 1`, so registers were already a binding constraint there. On
Ada, 100 KB/SM of shared memory admits 2 blocks, so shared memory no longer binds at all,
leaving the register file as the sole constraint holding `mlp_fused` to 1 block/SM.
**Registers did not start binding; shared memory stopped.** The phrasing "the binding
resource swapped" would be an overstatement and is not used here.

That matters for Experiment 1's conclusion, which turned on shared memory being the
constraint `BLOCK_M` could not move. On the L4 that constraint is gone and the register
file — which `BLOCK_M` *does* move — is what remains. Experiment 1's lever would be live
on this card; whether pulling it flips the result was not measured, and is not claimed.

For continuity with Experiments 1/1b (L4, ieee):

| config | n_regs | shared B | blocks/SM | occupancy |
|---|---:|---:|---:|---:|
| `BLOCK_M=2, BLOCK_H=32` (`mlp_fused_lowreg`) | 110 | 35,072 | 2 | 33.33% |
| `BLOCK_M=16, BLOCK_H=16` (`mlp_fused_blockh`) | 121 | 33,792 | 2 | 33.33% |

## Prediction 3 — BROKE; the premise is false on Ada

> **3. The matmul comparison needs controlling.** cuBLAS gets tensor cores; our fp32
> `tl.dot` does not. Report strict-fp32 AND TF32-enabled numbers, or the comparison
> measures precision policy rather than kernel quality.

The instruction was right and was followed. **The premise it rests on is wrong.** Our
`tl.dot` *does* get tensor cores on sm_89 — by default, without asking — because Triton's
NVIDIA backend hardcodes `default_dot_input_precision = "tf32"`. The controlling knob is
Triton's `TRITON_F32_DEFAULT`, not only torch's `allow_tf32`, and the pre-registration
knew about only the latter.

Batch 128, `triton.testing.do_bench`, TFLOPs = 2*M*N*K/s, shape k=192 n=768. Not written to
`latency.csv`: its `dtype` column would read `float32` for all four combinations with
nothing to distinguish them, and adding a column would break `record()`'s header contract.

| Triton `tl.dot` | torch `allow_tf32` | torch | triton | triton_tuned |
|---|---|---:|---:|---:|
| ieee | False | 10.71 TF | 5.63 TF | 8.11 TF |
| ieee | True | 17.03 TF | 5.64 TF | 8.13 TF |
| tf32 | False | 10.82 TF | **13.26 TF** | **16.19 TF** |
| tf32 | True | 16.84 TF | 13.25 TF | 16.12 TF |

Full grid, `triton/torch` latency ratio (>1 means Triton slower), batch 128:

| Triton | torch allow_tf32 | k192/n576 | k192/n192 | k192/n768 | k768/n192 |
|---|---|---:|---:|---:|---:|
| ieee | False | 1.83x | 1.36x | 1.90x | 1.54x |
| ieee | True | 2.26x | 1.75x | 3.02x | 2.49x |
| tf32 | False | **0.88x** | **0.84x** | **0.82x** | **0.85x** |
| tf32 | True | 1.11x | 1.03x | 1.27x | 1.45x |

**Read only the two matched-policy rows.** The four-row grid crosses two independent
knobs, and just two of its rows compare like with like:

| matched policy | grid row | k192/n576 | k192/n192 | k192/n768 | k768/n192 |
|---|---|---:|---:|---:|---:|
| both IEEE | Triton `ieee`, torch `allow_tf32=False` | 1.83x | 1.36x | 1.90x | 1.54x |
| both TF32 | Triton `tf32`, torch `allow_tf32=True` | 1.11x | 1.03x | 1.27x | 1.45x |

**At matched precision cuBLAS still wins — the Triton kernel is 36-90% slower with both at
IEEE and 3-45% slower with both at TF32.** The two remaining rows are mismatches and must
not be read as kernel comparisons: the `Triton tf32 / torch allow_tf32=False` row
(0.82-0.88x) puts our kernel on tensor cores against a cuBLAS forbidden them, and the
`Triton ieee / torch allow_tf32=True` row (1.75-3.02x) does the reverse. That torch's flag
genuinely binds is visible in the TFLOPs table: 10.82 TF at `False` against 16.84 TF at
`True`, same shape, same call.

So the deficits split in two. **The 1.75-3.02x band is mismatch-driven** and shrinks to
1.03-1.45x once both sides are on tensor cores. **The 1.36-1.90x band is not** — it is a
genuine matched-IEEE deficit with no instrumentation artifact behind it. `triton_tuned`
closes most of the remaining gap at the one shape measured for it (k192/n768: 16.12 TF
against TF32-enabled cuBLAS's 16.84 TF, ~4% short — parity, not a win); the other three
shapes were not measured for that arm, so no claim is made about them.

**Cost of IEEE relative to TF32, from the controlled measurement: ~2.3x** (`triton` at
k=192/n=768, 5.63 -> 13.26 TF; 2.0-2.4x across the other three shapes). The full test
suite also ran 415 s under `tf32` versus 1,299 s under `ieee`, but that ~3x is weaker
evidence — the two runs differ in 70 pass/fail outcomes and include every non-`tl.dot`
test — and should not be quoted as the figure.

Verdict: **prediction 3's premise BROKE.** Its methodological instruction — report both
precisions, never average them — was necessary and is vindicated: without it every L4
number in this document would silently be a different arithmetic from every sm_75 number in
the corpus, and the mismatched 0.82-0.88x row would read as a kernel win. The performance
conclusion underneath is the unexciting one. **cuBLAS still beats our GEMM at matched
precision, on Ada as on Turing** — `07-retuning.md`'s parity-at-best finding survives the
card change; only its stated *reason* ("on a card with no tensor cores cuBLAS is hard to
beat") turns out not to be the operative one, since cuBLAS wins on a card with them too.

## Prediction 4 — HELD; the bound moved and did not disappear

> **4. The monolithic block kernel still fails to compile.** It needs 262,144 B; Ada
> allows ~99 KB/block. The bound moves, it does not disappear.

```
BLOCK_H=32: OutOfResources: out of resource: shared memory, Required: 278528, Hardware limit: 101376.
BLOCK_H=16: OutOfResources: out of resource: shared memory, Required: 278528, Hardware limit: 101376.
```

Two details that must travel with this result:

- **The limit the compiler enforced is 101,376 B — sm_89's opt-in dynamic shared-memory
  maximum, not the 49,152 B default** that `torch.cuda.get_device_properties()` reports as
  `shared_memory_per_block`. Both were printed in the run for exactly this reason. The
  bound moved 65,536 B -> 101,376 B (+54.7%), and the requirement still overshoots it by
  2.75x. As on sm_75, the requirement is invariant in `BLOCK_H`.
- **The kernel tested is a reconstruction, requiring 278,528 B**, not the uncommitted
  original's 262,144 B recorded in `05-over-fusion.md`. The dominant term is identical and
  drives both figures — the `[BLOCK_D, BLOCK_D]` = 256x256x4 = 262,144 B output-projection
  weight tile — with this version additionally holding a 16,384 B attention tile live
  alongside it. It was validated against the local 1650 Ti first, where it reproduces
  `Hardware limit: 65536`. Do not read 278,528 B as the original prototype's figure.

## The SDPA backend on sm_89 (observation, not scored)

```
fmha_cutlassF_f32_aligned_64x64_rf_sm80(
    PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params)
```

**The backend differs only in its SM target, not in its family.** sm_75 selected
`fmha_cutlassF_f32_aligned_64x64_rf_sm75`; sm_89 selects the `_sm80` build of the same
CUTLASS memory-efficient kernel. PyTorch did **not** switch to a FlashAttention backend
despite Ada having one — expected, since its FlashAttention kernels do not serve fp32. The
pre-registered expectation ("SDPA will select a different backend") is technically
satisfied and substantively not: it is the same kernel, recompiled.

Latency at batch 128, heads 3, seq 64, head_dim 64, fp32. The column heading is the
**Triton** precision only: `torch.backends.cuda.matmul.allow_tf32` was never written by the
SDPA measurement, so SDPA ran at the container's recorded default of `False` in both
columns (see Environment). **Only the `ieee` column is a matched comparison.**

| arm | ieee (matched) | tf32 (Triton only — NOT matched) |
|---|---:|---:|
| `F.scaled_dot_product_attention` | 0.1508 ms | 0.1504 ms |
| `attention_flash` (ours) | 0.2727 ms | 0.1322 ms |
| `attention_composed` (ours) | 0.2566 ms | 0.2559 ms |
| **flash / sdpa** | **1.81x slower** | 0.88x — **mismatched, not a win** |

**At matched precision our flash kernel still loses to SDPA on the L4, by 1.81x** — down
from 3.26x on sm_75, which is the result. The 0.88x in the second column **is not a kernel
comparison and must not be quoted as one**: it puts our kernel on tensor cores against an
SDPA that is not using them, the same mismatch prediction 3's grid contains. SDPA's own
timing is unaffected by the Triton knob (0.1508 vs 0.1504 ms), as it must be, which is
consistent with — but on its own no proof of — SDPA being off the tensor cores; the
recorded `matmul_allow_tf32 False` is what supports that.

## The largest finding is none of the four: the kernels are not TF32-safe

This was not predicted, is not one of the four, and is more consequential than any of them.

Triton's NVIDIA backend hardcodes `default_dot_input_precision = "tf32"`
(`triton/backends/nvidia/compiler.py:123`), overridable by `TRITON_F32_DEFAULT`
(`triton/knobs.py:484`). **sm_75 has no TF32 tensor cores, so that default had nothing to
lower to and silently produced IEEE fp32.** On sm_89 it binds for real.

| Triton fp32 precision | full suite on L4 |
|---|---|
| `tf32` (Triton's default on Ada) | **70 failed, 83 passed** |
| `ieee` | **153 passed** — matches sm_75 test for test |

Same kernels, same torch reference, batch 8, `torch.backends.cuda.matmul.allow_tf32 = False`
in both runs so torch is not the variable; tolerances in `tests/conftest.py` untouched
(`rtol=1e-4, atol=1e-4`):

| kernel | `ieee` max_abs | `ieee` outside tol | `tf32` max_abs | `tf32` outside tol |
|---|---:|---:|---:|---:|
| `linear` | 1.669e-06 | 0 / 98,304 | 2.528e-03 | 70,470 / 98,304 |
| `linear_tuned` | 1.669e-06 | 0 / 98,304 | 2.528e-03 | 70,470 / 98,304 |
| `mlp_composed` | 5.484e-06 | 0 / 98,304 | 5.124e-03 | 86,551 / 98,304 |
| `mlp_fused` | 6.199e-06 | 0 / 98,304 | 5.122e-03 | 86,539 / 98,304 |

Four claims, each independently evidenced:

1. **This project's entire measurement history is IEEE fp32 only by accident of hardware.**
   No kernel, test, or finding in this repository ever chose IEEE; the card chose it.
2. **The kernels are not TF32-safe at the accuracy this project declares** — one to two
   orders of magnitude outside tolerance, on 70-88% of elements, under Triton's own
   default.
3. **Nothing in the kernels declares a precision policy.** Not one `tl.dot` call site in
   `model/kernels/` passes `input_precision`, and no test asserts which precision is in
   force. The policy is inherited from a backend default that differs by hardware
   generation and is invisible at every call site.
4. **Every failure identified in the measurement record is a `tl.dot` kernel**, and every
   elementwise and reduction kernel named in that record passed under both settings —
   subject to the same 8-test gap below, which leaves the universal unproven in both
   directions. Failures as recorded:
   `test_linear` (24), `test_mlp` (16), `test_linear_gelu` (8), `test_block` (8),
   `test_attention` (5, all `attention_flash`), `test_end_to_end` (1).

   **That enumeration sums to 62, against the 70 reported failed. Eight failures are
   unaccounted for.** The gap is inherited from the measurement record, which lists the
   per-file counts without the full failure list, and it cannot be closed without re-running
   the suite on an L4 — which this experiment will not do. So claim 4 is **evidenced for
   the 62 enumerated failures and unverified for the remaining 8**. It is stated at that
   strength rather than weakened to fit: the plausible homes for the missing 8 are
   `test_qkv` and `test_vit_baseline`, both `tl.dot`-bearing, but that is a guess and is
   not offered as evidence. Anyone who re-runs the `tf32` suite should close this.

This is a portability defect in the project's kernels, found by moving hardware, and it
must not be softened into a tolerance problem. It also gates everything else in this
document: **every L4 latency and counter row recorded here was taken under
`TRITON_F32_DEFAULT=ieee`**.

That knob governs `tl.dot`, and therefore only the Triton arms. **The `torch` and
`torch_compile` arms route through cuBLAS, which `TRITON_F32_DEFAULT` does not touch at
all**; their precision is set by `torch.backends.cuda.matmul.allow_tf32`, which no sweep
body writes. Those arms are IEEE on the strength of the recorded container default
(`matmul_allow_tf32 False`, Environment above) — the same default the sm_75 record ran
under. Taken together the L4 rows are arithmetic-for-arithmetic comparable to sm_75, with
the caveat that the torch-side half of that claim rests on a value recorded by `smoke`
rather than read back inside the sweep process. Had the default been `True`, every
within-card `triton/torch` ratio here would carry the same mismatch prediction 3 warns
about — worth 1.56x on torch at k192/n768 (10.82 -> 16.84 TF), which is not a small
number. A future campaign should read the flag back per sweep rather than inherit it.

TF32 numbers appear only in tables explicitly labelled with a precision —
prediction 2's register table, prediction 3's matmul tables, the `precision_check` table
above, and the SDPA table — and nowhere else.

## `vit_forward` end-to-end — the comparison with full coverage on both cards

All four arms at all five batches on both cards. Every cell is like-for-like on the
Triton side (`TRITON_F32_DEFAULT=ieee`, read back per run) and like-for-like on the torch
side to the extent the recorded `matmul_allow_tf32 False` default is trusted — see the
precision note above. Ratios
against each card's *own* `torch` baseline, which is how the fusion ladder's conclusions
are stated (>1 means slower than torch):

| card | batch | `torch_compile` | `triton_composed` | `triton_fused` |
|---|---:|---:|---:|---:|
| 1650 Ti | 1 | 0.92x | 2.26x | 5.01x |
| 1650 Ti | 8 | 1.07x | 2.15x | 5.27x |
| 1650 Ti | 32 | 0.77x | 1.95x | 5.07x |
| 1650 Ti | 128 | 0.83x † | 2.27x | 5.93x |
| 1650 Ti | 512 | 1.19x † | 1.91x † | 4.74x † |
| L4 | 1 | 0.63x | **2.05x** | **1.80x** |
| L4 | 8 | 0.71x | 1.80x | 1.58x |
| L4 | 32 | 0.83x | 1.67x | 3.95x |
| L4 | 128 | 1.01x | 1.38x | 2.68x |
| L4 | 512 | 1.31x | 1.43x | 2.57x |

Absolute medians (ms) and the L4's speedup over the 1650 Ti at the same arm and batch:

| batch | `torch` | `torch_compile` | `triton_composed` | `triton_fused` |
|---:|---|---|---|---|
| 1 | 0.8281 -> 1.5094 (**0.55x**) | 0.7614 -> 0.9569 (0.80x) | 1.8739 -> 3.0930 (0.61x) | 4.1472 -> 2.7228 (1.52x) |
| 8 | 2.8652 -> 1.7664 (1.62x) | 3.0684 -> 1.2585 (2.44x) | 6.1619 -> 3.1821 (1.94x) | 15.0967 -> 2.7868 (5.42x) |
| 32 | 11.5983 -> 1.9236 (6.03x) | 8.9847 -> 1.6026 (5.61x) | 22.6526 -> 3.2046 (7.07x) | 58.8083 -> 7.5976 (7.74x) |
| 128 | 39.5250 -> 8.7685 (4.51x) | 32.9621 † -> 8.8253 (3.73x) | 89.6230 -> 12.0924 (7.41x) | 234.3975 -> 23.5372 (9.96x) |
| 512 | 203.2131 † -> 39.1327 (5.19x) | 241.7115 † -> 51.3992 (4.70x) | 389.1505 † -> 55.9514 (6.96x) | 963.1414 -> 100.5942 (9.57x) |

**†** marks a cell whose sm_75 side involves a `flagged` row, per the throttle disclosure
above: `torch_compile` @128 (675 MHz), `torch` @512 (915 MHz), `torch_compile` @512
(630 MHz), `triton_composed` @512 (300 MHz). Every batch-512 ratio against the sm_75 torch
baseline inherits that baseline's flag, which is why the whole 1650 Ti @512 ratio row is
marked. **No ratio carrying a † should be quoted as a measurement**, per this document's
own rule. Batches 1, 8 and 32 are unflagged on the sm_75 side throughout.

Reading it:

- **The headline verdict survives the card change.** On both cards, at every batch,
  `triton_fused` is slower than `triton_composed` is slower than eager `torch` — with
  exactly one exception, below. **The over-fusion penalty is not an artifact of a small,
  slow, tensor-core-less laptop GPU.**
- **The penalty shrinks substantially but does not invert.** `triton_fused` costs 5.93x
  torch at batch 128 on the 1650 Ti and 2.68x on the L4; `triton_composed` goes 2.27x ->
  1.38x. The L4 is kinder to both Triton arms and kindest to the most over-fused one.
- **One rank flip, at batch 1 — measured; its cause is not.** On the L4 `triton_fused`
  (1.80x torch) is *faster* than `triton_composed` (2.05x torch); on the 1650 Ti it was
  decisively slower (5.01x vs 2.26x). **No row in this comparison is flagged on either
  card**, so the flip is not a throttling artifact. The obvious reading — at batch 1 the
  model is launch-bound, and the fused block launches fewer kernels (73 against 77, from
  `06-synthesis.md`), so its launch saving finally exceeds its serialisation cost — is
  **an untested hypothesis, labelled as such.** This experiment measured no launch count,
  no per-launch overhead and no kernel-time-versus-wall-time split on the L4; it ruled out
  clocks and nothing else. It is the same evidentiary position as `attention_flash` below
  and gets the same treatment. What would settle it: a `torch.profiler` launch-count and
  kernel-time breakdown for both arms at batch 1 on the L4, which is cheap and was not
  run.
- **The L4 is *slower* than the 1650 Ti at batch 1** for three of four arms (0.55x, 0.61x,
  0.80x). A 58-SM datacenter card cannot show its advantage on a batch-1 forward pass of a
  tiny ViT: the work is latency-bound, so per-launch overhead dominates and the extra SMs
  and bandwidth have nothing to bite on. This is *not* a clock effect — all four batch-1
  L4 rows record `sm_clock_mhz=2040`, the highest in the entire run.
- **`torch_compile` beats eager at small batch on both cards** (0.92x/0.77x on the
  1650 Ti at batches 1/32, 0.63x/0.71x/0.83x on the L4 at 1/8/32, all unflagged), which is
  the trustworthy half of Experiment 2's Inductor finding replicating across cards. **The
  "and loses at batch 512 on both" half does not survive the throttle disclosure.** The
  sm_75 1.19x cell has `torch_compile` at 630 MHz against a `torch` baseline at 915 MHz —
  a 1.45x clock gap that could account for the whole ratio on its own, and
  `08-inductor.md`'s own caveats section already flagged that figure as imprecise. The
  batch-512 loss is established **on the L4 only** (1.31x, no flagged row involved on
  either side of the comparison).

## Per-kernel rungs

Median latency (ms), latest capture per cell, L4 rows IEEE. Cells whose sm_75 side is
`flagged` are marked; the six `linear_gelu` cross-card speedups are withdrawn entirely and
are not listed.

| kernel | variant | batch | 1650 Ti | L4 | L4 speedup |
|---|---|---:|---:|---:|---:|
| `mlp` | `torch` | 128 | 3.1990 | 0.6390 | 5.01x |
| `mlp` | `triton_composed` | 128 | 8.6534 | 0.9339 | 9.27x |
| `mlp` | `triton_fused` | 128 | 32.8346 | 3.3449 | 9.82x |
| `mlp` | `triton_fused_blockh` | 128 | 52.2351 | 4.8246 | 10.83x |
| `mlp` | `triton_fused_lowreg` | 128 | 193.1410 | 17.6133 | 10.97x |
| `mlp` | `torch` | 512 | 12.6581 | 2.7761 | 4.56x (sm_75 flagged) |
| `mlp` | `triton_composed` | 512 | 34.5503 | 5.0186 | 6.88x (sm_75 flagged) |
| `mlp` | `triton_fused` | 512 | 131.2155 | 12.6909 | 10.34x |
| `block` | `torch` | 128 | 5.5702 | 1.3322 | 4.18x |
| `block` | `triton_composed` | 128 | 11.1456 | 2.0792 | 5.36x |
| `block` | `triton_fused` | 128 | 27.6488 | 4.3786 | 6.31x |
| `attention` | `torch` | 128 | 0.4632 | 0.2335 | 1.98x |
| `attention` | `triton_composed` | 128 | 0.5939 | 0.2580 | 2.30x |
| `attention` | `triton_flash` | 128 | 1.0016 | 0.2785 | 3.60x |
| `attention` | `triton_flash` | 512 | 4.0630 | 0.7736 | 5.25x |
| `layernorm_residual` | `triton` | 128 | 0.1938 | 0.1132 | 1.71x |
| `layernorm_residual` | `triton_residual` | 128 | 0.1516 | 0.1260 | 1.20x |
| `layernorm_residual` | `triton` | 512 | 0.7660 | 0.4485 | 1.71x |
| `layernorm_residual` | `triton_residual` | 512 | 0.5913 | 0.4413 | 1.34x |

- **The fusion ladder's ordering survives on the per-kernel rungs too.** On both cards
  `torch` beats `triton_composed` beats `triton_fused` for `mlp` and `block` at every batch.
- **`layernorm_residual`'s fusion win narrows and changes sign at batch 128.** On the
  1650 Ti the fused arm is 21.8% faster (0.1516 vs 0.1938 ms). On the L4 it is **11%
  *slower*** (0.1260 vs 0.1132 ms), though it stays marginally ahead at batch 512 (0.4413
  vs 0.4485). This is the only rung where the L4 changes a sign — and it is the rung
  prediction 1 was about. **But the mechanism prediction 1 proposed is not what did it:**
  the DRAM traffic ratio that was supposed to explain the win barely moved (1.50x ->
  1.55x). The win eroded while its stated cause stayed put, so this project has measured
  the erosion without explaining it. That is left unresolved rather than given a story.
- **`attention_flash` gains most of all** (3.60x at batch 128, 5.25x at 512) — **and no
  DRAM-traffic explanation is offered, because the counters support none.** That arm reads
  21,183,232 B on sm_75 and 21,370,624 B on the L4: **+0.9%, essentially unchanged**, while
  sitting 2.0-2.5x below the composed arm on both cards. Its traffic did not move, so its
  speedup is not a traffic effect. A compute/occupancy explanation (58 SMs vs 16, on a
  kernel that was occupancy-starved on the smaller card) is available and **is labelled an
  untested hypothesis, not a measurement.**

## What the rule becomes on hardware where L2 might have made fusion pointless

The plan posed this experiment as a fork: on a big-L2, tensor-core card, does the L2 make
fusion *pointless* rather than registers making it *costly*? **Neither.** Both halves of
the fork are refuted by the measurement.

- **L2 did not make fusion pointless.** Fusion's DRAM-read advantage survived intact on
  all three rungs (1.55x, 2.02x, 4.94x) on a card with ~48x the L2. Why it survived is not
  established — see prediction 1; three of the six measured arms read *more* DRAM on the
  bigger-L2 card, so no cache-absorption account fits the data.
- **Registers did not become the cost either.** The mega-MLP still loses on the L4, at
  every batch except the launch-bound batch-1 `vit_forward` case — while its occupancy
  bookkeeping changed (25.00% -> 16.67%) and shared memory stopped binding entirely.
  The loss survived a change in *every* resource the register rule invokes.

So the reconciled headline from `10-register-rule.md` stands and is now cross-card:

> Shared memory sets the compile-time feasibility bound, but the mega-MLP's actual latency
> loss comes from the H-loop's unpipelined serial reduction, not from register or occupancy
> collapse.

**L4 corroborates it, and sharpens the first clause into a hardware-dependent one.** The
feasibility bound is a property of the card: it moved 64 KB -> 100 KB per SM, which was
enough to stop shared memory binding `mlp_fused` at all (2 blocks by shared, 1 by
registers) and not nearly enough to let the monolithic block kernel compile (278,528 B
required against 101,376 B). The *cost* clause is the card-independent one: the H-loop's
serial reduction cost the fusion its win on a card with 3.6x the SMs, 48x the L2, ~1.6x the
bandwidth, and tensor cores. **The feasibility bound travels with the hardware; the
serialisation cost travels with the kernel.**

**How strong that second sentence is, stated precisely.** The H-loop was instrumented on
sm_75 (Experiment 1b: iteration count 24 -> 48, latency ratio 3.8x -> 6.0x, with occupancy
and traffic held constant) and **was not instrumented on the L4.** What the L4 contributes
is eliminative: the fusion still loses while registers, occupancy and shared-memory binding
all changed, and its DRAM advantage grew rather than shrank, so none of those can be the
cost on this card either. The attribution to the H-loop is carried over from sm_75 as the
only surviving candidate, not re-measured on Ada.

Two clauses the rule does not have — one a hypothesis, one supported:

- **A candidate launch-count clause, not yet earned.** The batch-1 rank flip is the one
  place the L4 changes an ordering, and it is the one place memory traffic plainly cannot
  be the explanation. Launch count is the obvious candidate — it is the one axis on which
  fusion is unambiguously ahead and which the register rule never mentions. **But this
  experiment did not measure it on the L4**, so the clause is recorded as a hypothesis
  worth testing, not as an amendment the evidence supports. Adding it to the rule on the
  strength of one unexplained rank flip would be exactly the manufactured significance
  this document refuses elsewhere.
- **A precision-policy clause.** "Fusion pays when..." presupposes the two arms compute
  the same arithmetic. On sm_75 they did, silently and by accident. On sm_89 the default
  changes underneath unchanged source. Any cross-card claim about kernel quality is
  meaningless until the precision policy is declared on both sides.

## The instrumentation bug this replication caught

The first counter measurement was wrong, and the corrected table above is the second.

`counter_grid_body` used `launch_skip = 5 * stride`. `bench/run_mlp.py` launches **7** CUDA
kernels during benchmark tensor setup (5 `randn` + 2 `* 0.05`) before the first arm call,
and `bench/run_layernorm_residual.py` launches 4. For a single-launch fused arm `stride ==
1`, so the window landed at launch 6 — **inside setup**. The L4's fused arms profiled a
torch elementwise multiply instead of the kernel under test. Because a scalar multiply
touches a fraction of the bytes the real kernel does, the fused arms appeared to read
almost nothing, and prediction 1 was scored broken in a dramatic direction it did not
actually break in. **Every figure from that capture is withdrawn, and none of them is
reproduced here** — the corrected table above is the only counter result this experiment
has.

Three things worth recording about it:

- **It was caught because the two cards disagreed implausibly.** A two-order-of-magnitude
  gap between the two cards' fusion advantages is not a hardware difference, and it was
  only visible because the replication put the two side by side. A single-card measurement
  would have shipped the bug as a finding.
- **The root fix is identity, not counting.** `bench/profile.py` gained an optional
  `expected_kernel_names` parameter and a `base_kernel_name()` normaliser, and
  `bench/collect_counters.py` now measures each arm's setup launches and per-call stride in
  a fresh subprocess before choosing its window. A launch-*count* check could never have
  caught this: a fused kernel and a setup multiply both satisfy "exactly one distinct
  kernel".
- **A second defect fell out of the re-audit.** The sm_75 baseline for
  `mlp/triton_fused` (capture `01:39:51`) spans **two** launches summed as one call
  (88,728,864 B). Per call it is 44,364,432 B, agreeing to within 0.2% with the independent
  single-launch capture at `01:06:21`. The sm_75 fused/composed ratio is therefore **3.54x,
  not the 1.77x** the doubled figure implied. `mlp/triton_fused_blockh` at `01:39:57` has
  the same structure.

The three composed arms, re-measured with the corrected window, reproduce their pre-fix
numbers to within 0.3% — which is both the evidence that the defect was confined to the
single-launch arms and an incidental run-to-run reproducibility check on the counter
measurement.

### Superseded counter rows are still in `counters.csv` — how to identify them

The wrong-kernel L4 rows were **not deleted**, because deleting measured rows rewrites the
record, and a `superseded` marker column would break `record()`'s header contract for
2,367 prior rows.

- **Superseded rows: the first counter capture, timestamps `2026-08-18T22:45:*` and
  `22:46:*` (77 rows).** The wrong-kernel rows within it — 11 by metric row — are the
  `attention/triton_flash` and `mlp/triton_fused` arms.
- **Corrected rows: timestamps `2026-08-19T16:09`-`16:10`.**
- **Identification rule:** the superseded rows are self-identifying — their `kernel_name`
  holds a torch `vectorized_elementwise_kernel` where a Triton kernel belongs. Any analysis
  of L4 counter rows must filter on `kernel_name`, which is exactly the check that should
  have been present from the start.

## Scoped reconciliation with the existing corpus (ruling R2)

Experiment 4, the corpus-wide consistency pass, ran at commit `3e28816` — **before** this
experiment, contrary to the plan's ordering. Rather than re-running it, this section names
only the claims the L4 results affect, and where they live. **No measured number in
`docs/findings/00`-`10` or `README.md` is altered by this experiment**; every one of them
remains a correct record of what was measured on the card and configuration it names. What
changes is the *scope* those claims may be stated at.

| claim | where | what the L4 changes |
|---|---|---|
| "Experiment 3 (replicating on an L4 GPU) was never run [...] The rule's card-specificity remains untested." | `README.md`, register-rule status list | **Was false; corrected.** It ran, and this document is the result. The `README.md` bullet was replaced in a follow-up commit — the only edit this experiment made outside this file. |
| "on this card, fusion pays only when it removes a DRAM round-trip without adding register or loop-serialization cost" | `README.md` headline | The "on this card" hedge can now be lifted for the *ordering* (fused > composed > torch survives on sm_89 at every batch but one) and must be kept for the *magnitudes* (5.93x -> 2.68x at batch 128) and for `layernorm_residual`'s win. |
| `layernorm_residual` "22-31% faster" every batch | `02-layernorm-fusion.md`, restated in `08-inductor.md`'s recap table as "WINS 22-25%" | sm_75-scoped. On the L4 the fused arm is **11% slower** at batch 128 and only 1.6% faster at 512. Its DRAM advantage is unchanged (1.50x -> 1.55x), so the erosion has no measured cause. |
| `attention_flash` "~3.26x slower than SDPA" | `04-flash-attention.md` (two places) | sm_75-scoped. On the L4 the gap narrows to **1.81x at matched IEEE**. The 0.88x figure in the SDPA table is *not* a matched comparison — it is our kernel on tensor cores against an SDPA that is not using them — and must not be quoted as a win. |
| monolithic block "262,144 bytes required against a 65,536-byte budget" | `05-over-fusion.md`, `06-synthesis.md` | Still fails on sm_89, against a **101,376 B** opt-in limit. The 262,144 B figure is the uncommitted original's; the L4 test used a 278,528 B reconstruction. |
| "our best tuned Triton GEMM only reaches cuBLAS parity, never beats it" | `07-retuning.md`, restated in `08-inductor.md` | **Survives the card change.** At matched precision cuBLAS still wins on sm_89 (36-90% at IEEE, 3-45% at TF32); `triton_tuned` reaches ~4% short of TF32 cuBLAS at the one shape measured. What does not survive is the *reason* `08-inductor.md` gave — "on a card with no tensor cores cuBLAS is hard to beat" — since cuBLAS also wins on a card with them. |
| "`torch_compile` beat every arm this project hand-built at every batch except 8" | `README.md`, `08-inductor.md` | Replicates in shape, not in detail. On the L4 `torch_compile` beats eager at batches 1/8/32, ties at 128 (1.01x) and loses at 512 (1.31x), so the batch-8 exception is sm_75-specific. The batch-512 loss is now established on the L4; the sm_75 1.19x that was its only prior evidence is throttle-contaminated on both sides and should not be leaned on. |
| "shared memory bounds fusion first, at compile time; registers collapse occupancy second" | `10-register-rule.md` | The first clause is hardware-dependent. For `mlp_fused` on sm_89 shared memory does not bind at all; the register file is the sole constraint. The H-loop conclusion is unaffected and corroborated. |
| every `tl.dot` latency and accuracy number in the corpus | `04`, `05`, `07`, `08`, `10`, `00-hardware.md`'s probe | All of them are **IEEE fp32 by accident of sm_75 having no TF32 hardware**, not by choice. They remain valid, and they are not the numbers the same source produces on tensor-core hardware. |

Nothing in this table is a correction to a measurement. Everything in it is a scope
condition that was invisible while the project had one card.

## Cost

~3,951 s of client-side wall time around `.remote()` calls, an **upper bound** on billed
GPU seconds (it includes queueing, container start and image pull). At $0.000222/s that is
**~$0.88**, against the user's $5 limit. 537 s of it (13.6%) is waste from two aborted test
runs — one killed in error, one killed deliberately when it stalled downloading CIFAR-10 at
billed GPU rates. Image builds ran on Modal's CPU builders and consumed no GPU time; every
version pin, import and the presence of `ncu` was validated CPU-only before the first L4
container started.

## Test summary

**153 passed on the L4 under `TRITON_F32_DEFAULT=ieee`**, matching sm_75 test for test.
No tolerance in `tests/conftest.py` was changed (0-line diff) and no kernel in
`model/kernels/` was modified (`git diff --stat 936b4f3..HEAD -- model/kernels/` is empty).
Under Triton's own default on Ada the same suite is **70 failed, 83 passed** — see the
TF32 section, which is a finding, not a test failure to be fixed by loosening anything.

## Concerns and caveats

- **The L4's `flagged` column carries no information.** All 220 rows are unflagged because
  no lock was declared, not because the clocks were stable; they span 660-2040 MHz. Any
  future L4 campaign that wants a live drift signal must declare
  `TRITONFORMER_LOCKED_CLOCK_MHZ`.
- **The sm_75 side of every cross-card comparison is not identity-validated.** Its counter
  captures predate the `expected_kernel_names` guard. The `mlp` double-counting defect
  found here is exactly the class of error that guard exists to catch, and it was found in
  the sm_75 baseline by hand.
- **Four results in this document are measured and unexplained**, and are left that way
  rather than given a plausible story: `mlp`'s fused arm reading -81.4% fewer bytes with
  no intermediate to absorb (prediction 1); `layernorm_residual`'s fusion win eroding to
  -11% at batch 128 while its traffic ratio barely moved (per-kernel rungs);
  `attention_flash`'s 3.60x/5.25x L4 speedup with traffic flat at +0.9% (same section);
  and the batch-1 `vit_forward` rank flip (`vit_forward` section). **Two of the four carry
  a named candidate hypothesis**, each labelled untested — `attention_flash` (58 SMs vs 16)
  and the rank flip (launch count). `mlp`'s traffic collapse and `layernorm_residual`'s
  eroded win carry none at all.
- **Prediction 2's `BLOCK_M` lever is live on sm_89 and was not pulled.** With shared
  memory no longer binding, Experiment 1's register sweep would be a genuine occupancy
  experiment on this card rather than the null it was on sm_75. That is the obvious next
  measurement and this experiment did not make it.
- **Only `vit_forward` has verified full four-arm, five-batch coverage on both cards.**
  Per-kernel cross-card ratios were checked against `latency.csv` individually; any new
  one must be checked the same way before it is quoted.
