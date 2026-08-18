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

1. **The rule's premise may collapse.** A `[128,3,64,64]` fp32 intermediate is 12.6 MB and
   fits in 48 MB of L2, so the unfused arm may never reach DRAM. If so, **even
   register-free fusion stops paying** — which would make `layernorm_residual`'s 22-25%
   win an artifact of a small-L2 card. Testable: `dram__bytes_read.sum` for composed arms
   should collapse.
2. **Register arithmetic transfers unchanged.** Same 65,536 regs/SM, so `mlp_fused` at
   226 regs x 256 threads -> 1 block/SM -> 25% should reproduce exactly.
3. **The matmul comparison needs controlling.** cuBLAS gets tensor cores; our fp32
   `tl.dot` does not. Report strict-fp32 AND TF32-enabled numbers, or the comparison
   measures precision policy rather than kernel quality.
4. **The monolithic block kernel still fails to compile.** It needs 262,144 B, Ada allows
   ~99 KB/block. The bound moves, it does not disappear.

Note on prediction 2's relevance: because Experiment 1 already showed occupancy is *not*
the mechanism behind `mlp_fused`'s loss on sm_75, prediction 2 reproducing exactly on the
L4 would confirm that Triton's register-arithmetic model transfers across architectures
(same register file size, same compiled kernel, same occupancy formula) — it would not,
by itself, say anything about whether the L4 reproduces the loss, or whether the loss
still traces to H-loop serialisation there. Those are separate questions this experiment
does not pre-register predictions for.

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
