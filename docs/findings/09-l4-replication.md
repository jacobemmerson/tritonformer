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
