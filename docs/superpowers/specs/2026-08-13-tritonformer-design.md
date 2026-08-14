# Tritonformer: Forward-Pass ViT in Triton with Fusion Analysis

**Date:** 2026-08-13
**Status:** Design approved, pending spec review

## Goal

Implement the forward pass of a shallow vision transformer entirely in Triton kernels, then use Nsight to measure **when kernel fusion helps performance and when it hurts**. The trained model runs CIFAR-10 inference; accuracy is a correctness gate, not the objective.

The research question is the deliverable. A fusion that loses is as valuable a result as one that wins, provided the profiler explains why.

## Scope

**In scope:** forward-pass kernels (LayerNorm, GeLU, Softmax, Linear, attention), fusion variants of each, a benchmark harness, a scripted Nsight profiling workflow, and per-kernel findings documents.

**Out of scope:** backward passes, training loops, and optimizers in Triton. These are a separate project with their own spec.

**Rationale for deferring training.** Backward passes roughly double the kernel count and introduce the hardest kernel in the project (fused LayerNorm backward, requiring cross-program-instance reduction). More importantly, they change the research question: fusion decisions invert under training, because an intermediate that forward-only can discard may be needed for the backward pass. That is a genuinely interesting second question, but it is not the same question, and answering it well requires the forward kernels to already be solid.

The reference model is trained in stock PyTorch and its checkpoint frozen. Every accuracy comparison for the life of the project uses that one file.

## Hardware

| Target | Role |
|---|---|
| GTX 1650 Ti (laptop, TU117, sm_75) | Daily driver. Unlimited iteration, root access for profiling counters. |
| Modal (L4 or A10G) | Matmul/attention work if needed, plus the second-architecture data point. |

**The 1650 Ti has no tensor cores.** Turing tensor cores shipped on TU102/104/106; the 16-series parts had them fused off. Consequences:

- Memory-bound kernels (LayerNorm, GeLU, Softmax, residual adds) profile normally at ~192 GB/s. The full fusion story is learnable here with no compromise.
- `tl.dot` emits `mma.sync` on its fast path, which requires tensor cores. On sm_75 without them, behaviour depends on Triton version and dtype, ranging from "slow FMA fallback" to "fails to compile."

**Step 0 is a hardware capability probe** (`scripts/probe_hardware.py`): run `tl.dot` in fp32 and fp16, confirm compilation, record achieved TFLOPs. This resolves in minutes whether matmul work happens locally or moves to Modal, before any design depends on the answer.

The two-architecture comparison is a deliberate experiment, not a fallback. The compute-to-bandwidth ratio determines where fusion's break-even point sits, so the same kernel can legitimately win on one card and lose on the other. Explaining that divergence is a stronger result than any single number.

**The GPU is in a laptop and will thermally throttle.** Mitigations are designed into the harness rather than discovered as noise later (see Benchmark Harness).

## Model Configuration

| Parameter | Value |
|---|---|
| Input | CIFAR-10, 32×32×3 |
| Patch size | 4×4 → 8×8 = **64 tokens** |
| Pooling | Mean over patches (**no CLS token**) |
| Embedding dim | 192 |
| Depth | 6 blocks |
| Heads | 3 × head_dim 64 |
| MLP hidden | 768 (ratio 4) |
| Norm placement | Pre-norm |
| Primary dtype | fp32 |

**Mean pooling instead of a CLS token** is a deliberate deviation. A CLS token makes sequence length 65, which forces either padding to 128 (wasting half the compute on masked-out values) or a boundary-mask branch in every kernel. Sequence length 64 tiles perfectly against power-of-two block sizes and removes a whole class of masking bugs, at no measurable accuracy cost on CIFAR-10.

**fp32 is primary; fp16 is a later experiment.** Without tensor cores, fp16 buys no math speedup — only halved memory traffic, which does help bandwidth-bound kernels. But it also introduces accumulation-order and stability concerns that would masquerade as kernel bugs during early development. Correctness in fp32 first, then fp16 as a measured experiment.

**Batch size is an experimental axis, not a constant.** Standard sweep: **{1, 8, 32, 128, 512}**, VRAM permitting (4GB).

This axis is central to the research question. At batch 1 a 64×192 tensor is 48KB; the GPU is nearly idle and *kernel launch overhead dominates*. Fusion looks miraculous because it eliminates launches, not memory traffic. At batch 512 the workload is genuinely bandwidth-bound and fusion helps for the textbook reason. Same kernels, same fusions, opposite explanations — and the crossover point is exactly where over-fusion starts to hurt.

## Architecture

### Variant registry

One mechanism serves four purposes: incremental swap-in, fusion-ladder experiments, A/B benchmarking, and parametrized tests. Building separate mechanisms for each is how the repo degrades.

```
model/
  reference/     pure PyTorch twin of every component — oracle + baseline
  kernels/       Triton kernels, one file per component
  registry.py    (component, variant) -> callable
  vit.py         assembled model; takes a variant config
```

Primitives (`layernorm`, `gelu`, `softmax`, `linear`) register variants such as `torch` and `triton`. Composites (`mlp`, `attention`, `block`) register variants spanning the fusion ladder: `composed` (built from independently-selectable primitives) through `fused_epilogue` to `fused_full` (single mega-kernel).

The model takes a config naming one variant per component, defaulting to `torch` throughout. That config drives promotion of a kernel, benchmark sweeps, and test parametrization alike.

**Variant selection is an enum per component, not a set of booleans.** Flags like `fuse_gelu` / `fuse_bias` / `fuse_residual` admit combinations for which no kernel exists. An enum makes the valid set exactly the set of registered variants, so an invalid configuration fails at construction rather than silently producing a wrong result.

**Composites do not reach into primitives' internals.** A fused MLP kernel is a peer of the composed one, not a subclass or special case. Their shared contract is the reference implementation — same inputs, same outputs, same declared tolerance. This is what makes `fused_full` writable without touching working code.

### Repository layout

```
model/         reference/, kernels/, registry.py, vit.py
bench/         harness.py (do_bench -> CSV), profile.py (ncu -> CSV), results/
tests/         pytest, allclose vs reference, parametrized over variants
scripts/       probe_hardware.py, train_reference.py, lock_clocks.sh
docs/findings/ per-kernel markdown: what the numbers meant and why
data/          CIFAR-10 + frozen reference checkpoint
```

## Build Strategy

**Reference-first with incremental swap-in**, with the fusion ladder as the measurement pattern applied within each layer.

1. Build the PyTorch ViT; train to a checkpoint; freeze and commit it (~20MB fp32).
2. Per component: write tests → write the unfused Triton kernel → verify against reference → benchmark → then climb the fusion ladder, profiling each rung.

This keeps a working, accuracy-validated model available at all times, and makes any regression attributable to the single kernel just swapped in. It also front-loads risk: the `tl.dot` probe and the training run both complete before significant kernel investment.

The rejected alternative was kernel-first (write all kernels, assemble at the end), where integration failures arrive all at once with no bisection path.

## Kernel Inventory and Fusion Ladder

Shapes: B × 64 tokens × 192 dim, 3 heads × 64 head_dim, MLP hidden 768, depth 6.

**A shape fact drives the attention design.** At S=64 and head_dim=64, a head's Q, K, and V are 16KB each in fp32 — 48KB total, within Turing's 64KB shared memory. The entire attention computation for a head fits in SRAM with no outer tile loop, so FlashAttention's tiling and online rescaling collapse to a single block. This makes the headline kernel substantially more approachable than usual. The cost: 48KB/SM permits only one resident block per SM, so occupancy is 1 — a fact to measure rather than assume benign.

| # | Component | Rung | Hypothesis / settling metric |
|---|---|---|---|
| 1 | LayerNorm | standalone | Matches torch; establishes bandwidth-bound floor |
| 2 | LayerNorm | + residual add fused | `dram__bytes` drops ~⅓ (eliminates one read + one write of B·S·D) |
| 3 | GeLU | standalone | **Expect no win.** Both implementations bandwidth-bound. Negative result: Triton alone buys nothing. |
| 4 | Softmax | standalone | Rows are 64 floats = 256B. Launch-overhead dominated; expect torch to win at low batch. |
| 5 | Linear | tiled matmul | The `tl.dot` risk. Expect to lose to cuBLAS. |
| 6 | Linear | + bias epilogue | Free win |
| 7 | Linear | + GeLU epilogue | Eliminates a round-trip of B·64·768 |
| 8 | Attention | composed | 5 launches + materialized S×S scores |
| 9 | Attention | QKV as one GEMM | 3 launches → 1; higher arithmetic intensity |
| 10 | Attention | flash-style fused | Scores never reach DRAM. Expect the largest single win. |
| 11 | MLP | fc1 + bias + GeLU fused | Standard epilogue fusion |
| 12 | MLP | **whole MLP one kernel** | **Expect this to hurt.** 768-wide intermediate → register spills |
| 13 | Block | **fully fused** | **Expect this to hurt more.** Deliberate over-fusion |

Rows 12–13 are the purpose of the project, not an afterthought. The predicted signature is specific and falsifiable: `launch__registers_per_thread` rises, `sm__warps_active` falls, `dram__bytes` falls, and local-memory traffic becomes non-zero — fusion trading memory traffic for occupancy at a net loss. On 16 SMs that trade should go bad early; on an A10G it may still pay. If rows 12–13 do not hurt, that is equally a finding, and the batch sweep locates the crossover.

Rows 3 and 4 are deliberately unglamorous. A project reporting only wins has not measured anything.

## Benchmark Harness

The harness's primary job is producing numbers that survive thermal drift on a laptop GPU.

**Interleave A/B comparisons at the rep level.** Running all reps of variant A then all reps of variant B measures the heatsink: the second variant runs hotter. Interleaving distributes thermal drift across both arms rather than loading it entirely onto whichever ran second. Medians are taken per variant afterward.

**Lock clocks and record them.** `nvidia-smi -lgc` / `-lmc` before a run; sample SM clock and temperature *during* the run and write them into every row. Rows whose clock deviated more than 5% from locked are flagged rather than silently averaged in — throttling should be visible in the data, not hidden in variance.

**Medians with percentiles, never means.** Use `triton.testing.do_bench`, which handles warmup, median reporting, and L2 flushing between reps. The L2 flush is essential: without it the input stays cache-resident and the measurement reflects a cache hit rather than the kernel.

CSV schema, one row per measurement:

```
timestamp, commit_sha, gpu, kernel, variant, batch, dtype,
latency_ms_median, latency_ms_p10, latency_ms_p90,
bytes_theoretical, achieved_gbps, sm_clock_mhz, temp_c, flagged
```

`commit_sha` allows a months-old row to be traced to the kernel version that produced it. `bytes_theoretical` is derived by hand per kernel — the traffic an ideal implementation must move — and dividing it by measured latency yields achieved bandwidth. Compared against the card's ~192 GB/s peak, that ratio indicates remaining headroom. **Achieved-bandwidth fraction, not raw speedup, is the signal that a kernel is finished.**

## Profiling Workflow

**Two tools answering two different questions.**

- **`nsys` — the gaps *between* kernels.** Launch count, launch overhead, idle gaps, CPU-side stalls. This is the whole story at batch 1, where tensors are tiny and the workload is launch-bound. It is what demonstrates that a batch-1 fusion win came from eliminating launches rather than memory traffic.
- **`ncu` — what happens *inside* one kernel.** Memory traffic, occupancy, register pressure. This is the story at batch 512, where the workload is bandwidth-bound.

Running both against the same fusion at both ends of the batch sweep is what turns "this fusion helped" into "this fusion helped *because*."

**Never quote `ncu` timings as performance.** `ncu` serializes execution and replays kernels to collect counter sets, so its durations are inflated and not comparable to anything. Latency comes from `do_bench`; counters come from `ncu`. These are written to **separate CSV files**, not separate columns, so conflating them is structurally difficult.

Setup: profiling counters require elevated permissions — set `NVreg_RestrictProfilingToAdminUsers=0` via modprobe config and reboot, once. Scripted invocation targets a single steady-state launch via `--launch-skip` / `--launch-count`, avoiding cold-start effects.

Metric set:

```
dram__bytes_read.sum                                 traffic — did fusion cut it
dram__bytes_write.sum                                traffic
l1tex__t_sector_hit_rate.pct                         locality
sm__warps_active.avg.pct_of_peak_sustained_active    achieved occupancy
launch__registers_per_thread                         register pressure
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum          spills
l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum          spills
```

The local-memory counters are the critical addition. Register spills go to local memory, which is physically DRAM. Non-zero local traffic is direct, unambiguous evidence of over-fusion — the smoking gun for the rows 12–13 hypothesis. Without it, occupancy would be seen to fall with no visible cause.

## Testing and Validation

Tests are written before each kernel. Because tests parametrize over `(component, variant)`, **every new variant automatically inherits the full correctness suite** — registering `fused_full` subjects it to the same oracle as everything else with no new test code.

**The reference implementation is the oracle and must stay boring.** Plain `torch.nn.functional` calls, no optimization. The moment it becomes interesting it stops being trustworthy as ground truth.

**Tolerances are declared per kernel and justified, never tuned until green.** Reduction order differs from torch's, so bitwise equality is impossible; fp32 LayerNorm and matmul land near `rtol=1e-4`. If a test requires a looser tolerance than declared, that is a finding to investigate, not a number to edit — it is usually a real bug such as a missing mask or a wrong accumulator dtype.

Edge cases reflecting actual usage:

- **D=192 is not a power of two.** Every kernel needs masked loads on the feature axis; a shape exercising the partial block is mandatory. This is the most likely source of silent garbage.
- **Non-contiguous inputs.** Attention reshapes and transposes to split heads, so kernels receive strided tensors. Tests use real post-transpose layouts, not freshly-allocated contiguous ones.
- **Batch 1 and large batch**, since grid logic differs at the extremes.
- **Numerical stress:** softmax with large-magnitude inputs (max-subtraction correctness); LayerNorm with near-zero variance (eps handling).

**End-to-end gate:** the assembled Triton ViT must match the frozen reference checkpoint's CIFAR-10 test accuracy within 0.1% **and** agree on ≥99.9% of individual predictions. Accuracy alone is too coarse — two models can reach identical accuracy while disagreeing on many images, hiding a real kernel bug.

## Deliverables

1. Working forward-pass ViT in Triton, matching the reference checkpoint on CIFAR-10.
2. Benchmark CSVs covering the kernel × variant × batch × dtype × GPU grid.
3. Nsight counter CSVs for each fusion rung.
4. Per-kernel findings documents in `docs/findings/` recording the *interpretation* — the numbers alone will not later explain why a fusion lost.

Plots are deferred until the CSVs exist and the comparisons that matter are known.

## Open Questions

- Whether `tl.dot` is usable on the 1650 Ti. Resolved by step 0 before any dependent work begins.
- Whether batch 512 fits in 4GB VRAM at fp32. If not, the sweep truncates to the largest batch that fits, and the ceiling is recorded.
