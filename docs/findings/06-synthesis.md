# Finding 06: Synthesis — when does kernel fusion help on a GTX 1650 Ti?

This is the project's headline document. Tasks 7-18 built a fusion ladder
from single-op kernels up to a fully-fused transformer block and measured
each rung against its unfused neighbor. This document answers the
research question directly: on this card (sm_75, 16 SMs, no tensor
cores, 4GB, 64KB shared mem/SM, 65,536 regs/SM, 255 regs/thread max,
~192 GB/s peak DRAM bandwidth), when does fusing Triton kernels help, and
when does it hurt?

Every number below already exists in `docs/findings/00-05` or in
`bench/results/latency.csv` / `bench/results/counters.csv`. No new claim
is introduced.

## A note on this run's measurement conditions

Two things affect how the numbers in this document (and specifically the
Task 19 batch sweep and counter grid) should be read.

**Clocks were run unlocked.** `scripts/lock_clocks.sh` requires `sudo`,
unavailable on this host, and its 1200 MHz default is below this card's
1830-2100 MHz supported graphics-clock range regardless. `locked_clock_mhz()`
therefore returns `None` for every row in this run, so `Measurement.build`'s
`flagged` column is `False` by construction — not because clocks were
stable, but because the drift check that sets it never had a locked
target to compare against. Checking `bench/results/latency.csv`:
**0 of 240 rows are flagged.** That count carries no thermal-guard signal
here; it would read exactly the same whether the card ran rock-steady or
wildly unstable. The honest thermal record is the per-row `sm_clock_mhz`
and `temp_c` columns instead, and they show real movement: across the
`vit_forward` sweep rows, `sm_clock_mhz` spans 1530-1905 and `temp_c`
spans 69-84.

**The card throttled during the sweep.** A 3h22m training run had
thermally soaked the card before this session (it was measured throttling
to 690 MHz at 85C); the sweep was not started until the card cooled to
61C with SM clock idling at 300 MHz. But under the sustained 100% load of
the sweep itself, `nvidia-smi -q -d PERFORMANCE` recorded SM clock
oscillating **300-1575 MHz — a 5x swing — at 80-84C**, with two throttle
reasons simultaneously active: `SW Power Cap: Active` (this is a 50W
card) and `SW Thermal Slowdown: Active` (`HW Thermal Slowdown` was not
active). Consequences:

1. **Absolute figures in this run (ms, GB/s, TFLOPs) are depressed and
   noisy** — they are lower bounds on this card's capability, not a
   clean measurement of it.
2. **Relative comparisons (the ratios this whole document is built from)
   remain valid.** `bench/harness.py::compare` interleaves every arm at
   the rep level by design (Task 5) specifically so thermal drift lands
   on all arms in a run roughly equally, rather than loading entirely
   onto whichever arm happened to run second. Every conclusion below is
   a ratio between arms measured this way, so it survives the throttling
   even though the absolute numbers underneath it do not.

## 1. Which fusions helped, at which batch sizes, and by how much

| Fusion | Rung | Helped at | Margin | Traffic ratio (measured vs predicted) |
|---|---|---|---|---|
| `layernorm_residual` (fold `x + residual` into LayerNorm) | 2 | every batch 1-512 | **+22-31% faster** than the unfused Triton pair (23% at batch 512) | 0.8045 vs 0.80 predicted |
| `linear_gelu` (fuse the GeLU epilogue into the linear kernel) | 3/epilogue | every batch 1-512 | **+9-11% faster** than the unfused pair | 0.6695 vs 0.6676 predicted |
| `triton_qkv_fused` (one [D→3D] GEMM replacing three [D→D] GEMMs) | 9 | **batch 1 only** | 0.0548ms vs 0.0868ms unfused (a win); at batch 512 it is a wash/loss (13.1648ms vs 13.1332ms) | not traffic-limited; launch-count win at small batch only |

Both traffic-based wins (`layernorm_residual`, `linear_gelu`) matched
their array-pass arithmetic predictions to within 0.6-0.3%, and both
eliminated a genuine intermediate-tensor DRAM round trip without pushing
register pressure far enough to cost occupancy: `layernorm_residual`
introduced no new occupancy story at all (same `BLOCK=256`/`num_warps=4`
shape as the unfused LayerNorm kernel), and `linear_gelu` only raised
registers/thread from 128 to 168 (occupancy 49.37% → 37.18%, zero
spilling).

**A related, non-fusion result worth citing here**: rung 1's Triton
LayerNorm alone (not yet fused with the residual) already *beat* torch by
1.89x, reaching 153 GB/s against a ~169-170 GB/s raw-copy ceiling on this
card, because `F.layer_norm`'s generic multi-pass kernel only reached
60-81 GB/s. This is why rung 2 had headroom to win at all — GeLU's rung
(Task 8) had no such headroom, since `F.gelu` was already a single fused
elementwise kernel with nothing left to eliminate, and showed no
improvement (the correct, predicted result, not a failure to optimize).

## 2. Which fusions hurt, and which counter explains each

| Fusion / kernel | Loss | Counter that explains it |
|---|---|---|
| `softmax` (standalone rung) | ~2.5-2.7x slower than `F.softmax` | occupancy starvation — 0.5 elements/thread at `BLOCK=64`/`num_warps=4` — **not launch overhead** |
| `linear` (standalone Triton matmul) | 2.26-3.42x slower than cuBLAS | 0.808 vs 2.727 TFLOPs achieved; no tensor cores on this card, so a hand-written matmul cannot approach cuBLAS's throughput |
| `attention_flash` (fused QK^T→softmax→PV) | 1.49-2.24x slower despite -37.2% DRAM traffic | register-spill traffic (60,555,264 B local-memory ld+st) was **2.13x** the DRAM traffic it saved (28,367,200 B) — the fusion's own spill cost outweighed the round-trip it eliminated |
| `mlp_fused` (mega-MLP, both matmuls + GeLU in one kernel) | 3.10-3.83x slower, despite -73% DRAM and **zero** spilling | occupancy collapse: 226 regs/thread × 256 threads/block = 57,856 regs/block; 65,536/57,856 → 1 block/SM → 25.00% occupancy (8 of 32 warps/SM) |
| `block_fused` (whole transformer block, mega-MLP inside) | 2.16-2.50x slower, despite having the **fewest launches** of any block arm (11, vs 12 for `triton_composed`, vs 16-17 for the torch reference) | inherits the mega-MLP's occupancy collapse — fewer launches did not compensate |

The mega-MLP result is the project's **failed prediction that matters**:
the plan predicted over-fusion would fail via register spilling, the way
`attention_flash` did (255 regs/thread, hardware max, genuine spill).
`mlp_fused` never crossed that line — 226 regs/thread, under the 255
cap, zero measured spill (`l1tex__t_bytes_pipe_lsu_mem_local_op_ld/st`
both zero) — and still lost 3-4x. The compiler collapsed occupancy
instead of spilling, a different, earlier failure mode than the one the
brief predicted.

The occupancy method itself validated cleanly across every rung it was
checked against: 128 regs/thread → 50.0% predicted / 49.37% measured;
168 regs → 37.5% / 37.18%; 226 regs → 25.0% / 25.00% (exact); the flash
attention kernel's 255 regs was actually **shared-memory-limited**
(12.5% predicted / 12.48% measured), not register-limited, the one case
in the ladder where a different resource bound the same low number.

**Caveat carried from Task 15**: local-memory (spill) counters are
measured at L1, and some of that traffic may be absorbed by L1/L2 before
reaching DRAM. They are not byte-for-byte comparable to
`dram__bytes_read/write.sum`; the 2.13x ratio above compares like units
(both L1-measured local-memory traffic vs. the DRAM-traffic delta), but
should be read as directionally decisive, not as an exact accounting.

## 3. Where the crossover sits

**There is no crossover at the block level.** The brief's working
hypothesis was "fusion wins small (fewer launches dominate), loses large
(bandwidth work dominates)." The measured `block_fused` vs
`block_composed` ratio is 2.16x at batch 1 and drifts *upward*, not
across 1.0, to 2.50x at batch 512 — `block_fused` loses at every batch,
including batch 1, where fewer launches were supposed to matter most.
The same non-crossover shows up one level down at the mega-MLP (3.10x at
batch 1 → 3.83x at batch 512): fusion "never starts" paying rather than
"stops" paying at some point along the sweep.

The one place a genuine small-batch/large-batch split *does* appear is
`triton_qkv_fused`: it wins at batch 1 (launch-count reduction dominates
when each launch's fixed overhead is a large fraction of total time) and
is a wash or a loss by batch 512 (bandwidth/compute work dominates once
launch overhead is amortized). That is the one rung in the ladder that
matches a textbook launch-bound → bandwidth-bound transition.

**The actual break-even boundary sits at the QKV-projection rung — several
rungs before attention itself** (`docs/findings/05-over-fusion.md`'s
stated conclusion). Attention was already on the losing side of that
boundary, not sitting at it: `attention_flash` loses 1.49-2.24x across
the whole sweep, mega-MLP loses 3.10-3.83x, and the fully fused block
loses 2.16-2.50x. Nothing past the QKV-projection rung ever crosses back
to a win.

**This project's established conclusion, which this document does not
revise:** *on this card, fusion stops paying once shared memory forces
tiles too small to amortize the fusion, not once registers run out.* The
evidence for shared memory (not registers) as the real wall is the
monolithic block kernel that was attempted and never compiled: a single
kernel combining even one head's Q/K/V tiles with the MLP's weight tiles
required 262,144 bytes of shared memory against the 65,536-byte/SM
budget — a 4x overflow, independent of any batch or block-size tuning
tried. The compiler declining that fusion outright, at compile time,
before any runtime register or occupancy question could even arise, IS
the result: it is a harder, earlier wall than the register-pressure
story the brief predicted, and it is why `block_composed` (six
best-individual-kernel launches) is the maximum achievable rung on this
hardware, not the fully monolithic kernel.

## 4. The 1650 Ti versus a second GPU

**Not collected.** No second GPU was available for this run. This
project's entire measurement grid — the batch sweep and the counter
grid — was collected on a single GTX 1650 Ti. No cross-architecture
comparison is made or implied anywhere in this document, and none should
be inferred from it; a different GPU (with tensor cores, different
shared-memory/register budgets, or different DRAM bandwidth) could shift
every boundary described in Section 3, but that is not something this
project measured.

## 5. What would be fused differently, knowing the results

- **Check predicted occupancy from register count and shared-memory tile
  size before fusing, not just predicted DRAM-traffic savings.** Both
  wins here (`layernorm_residual`, `linear_gelu`) eliminated a real DRAM
  round trip *and* stayed comfortably under the occupancy cliff (37-50%,
  zero spill). Both catastrophic losses (`mlp_fused`, `attention_flash`)
  eliminated real DRAM traffic too (-73% and -37.2% respectively) but
  paid for it in occupancy or spill traffic that outweighed the saving.
  DRAM-traffic prediction alone is not a sufficient fusion heuristic on
  this hardware; the occupancy/spill counters are.

- **Stop at `block_composed`, not `block_fused`.** The fully fused block
  loses at every batch and inherits the mega-MLP's collapse by
  construction. Given the mega-MLP result was already known before Task
  17 ran, `block_fused` was foreseeably a loss; the honest fusion
  strategy for this MLP shape on this hardware is Task 12's
  `linear_gelu` epilogue fusion followed by a separate `linear`, not the
  full two-matmul mega-kernel.

- **Never attempt the monolithic single-kernel block.** Its shared-memory
  requirement (262,144 bytes vs. a 65,536-byte budget) was foreseeable
  from the MLP's own w1/w2 tile size alone (Task 16 already showed one
  `[BLOCK_D, BLOCK_H]` tile needs ~16x the SM's shared-memory budget);
  adding live Q/K/V tiles on top could only make that worse. This is a
  case where the arithmetic that predicted the mega-MLP's shared-memory
  floor should have been carried forward to rule out the block-level
  attempt before writing any kernel code for it.

- **Use `triton_qkv_fused` conditionally on batch size**, since it is the
  one fusion in the ladder with a genuine small-batch win and a
  large-batch loss — the kind of batch-aware dispatch this project's
  registry (`VariantConfig`) already supports structurally, even though
  no task wired up batch-conditional variant selection.

- **Do not chase attention fusion further on this card without tensor
  cores.** `attention_flash`'s spill traffic (2.13x the DRAM it saved)
  is a direct consequence of holding Q/K/V tiles in registers/shared
  memory on a 255-register-max, no-tensor-core SM; a card with more
  register/shared-memory headroom or native flash-attention hardware
  support would change this calculus, but this project measured only
  the 1650 Ti (Section 4).

## Correctness, for context

None of the above changes the correctness picture Task 18 already
established: the end-to-end accuracy/prediction-agreement gate passed
for both block variants (`triton_composed`, `triton_fused`) at
10000/10000 test-set prediction agreement with the reference
implementation, at the checkpoint's recorded accuracy of 0.8489. That
gate was proven non-vacuous — a deliberate 1% weight corruption dropped
agreement to 99.61%, well below the ≥99.9% threshold the real gate
requires. Every fusion discussed above, including the ones that lose on
latency, produces numerically correct output.

## Appendix: the Task 19 measurement grid

### End-to-end `vit_forward` latency sweep (median of 30 interleaved reps, fp32)

All five batch sizes fit in 4GB for every block variant — no OOM
truncation was hit.

| batch | torch (ms) | triton_composed (ms) | triton_fused (ms) | composed/torch | fused/torch |
|------:|-----------:|----------------------:|--------------------:|---:|---:|
|     1 |     0.6071 |                 1.3454 |               2.9298 | 2.22x | 4.83x |
|     8 |     2.1370 |                 4.5357 |              10.7511 | 2.12x | 5.03x |
|    32 |     9.7724 |                17.1314 |              41.9381 | 1.75x | 4.29x |
|   128 |    52.6370 |                95.1882 |             210.4312 | 1.81x | 4.00x |
|   512 |   201.9738 |               389.7883 |             867.7132 | 1.93x | 4.30x |

`sm_clock_mhz` on these rows spans 1530-1905, `temp_c` spans 69-84, and
every row is `flagged=False` (see caveat above — that column carries no
signal under unlocked clocks). This end-to-end view is consistent with
Section 3: `triton_composed`/`triton_fused` never beat the torch
reference at the whole-model level, at any batch, because every block
variant carries the same sub-fusion losses established in Sections 2-3.

### Counter grid captured for this task (`bench/collect_counters.py`)

Scope: `vit_forward` at **batch 1 only**, one complete forward pass per
block variant. `vit_forward` at batch 512 was not captured — `ncu`'s
kernel-replay cost on a full 512-batch model forward was prohibitive on
this disk-constrained host (98% full), and it would have been
scientifically redundant: the block-level counters already collected via
`bench/run_block.py` cover both batch extremes (1 and 512) for
`triton_composed`/`triton_fused` and are what carry the over-fusion
finding in Section 2-3. The `vit_forward` level mostly adds
embed/final-norm/head work that is identical across block variants.

One complete forward pass launches a different number of CUDA kernels
per variant (measured empirically via `torch.profiler`, not assumed —
each block variant's launch count differs by construction, and even the
torch-backed patch-embed/head GEMM's cuBLAS algorithm choice at this
degenerate batch=1 shape can vary between process launches):

| variant | launches/pass | distinct kernel names | counter rows captured |
|---|---:|---:|---:|
| `torch` | 103 | 14 | 721 (103 × 7 metrics) |
| `triton_composed` | 77 | 11 | 539 (77 × 7 metrics) |
| `triton_fused` | 73 | 12 | 511 (73 × 7 metrics) |

Each capture is `profile_kernel(..., expected_kernels=N)`-verified to be
exactly one complete pass with no partial or double-counted window.
