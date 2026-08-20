# Finding 03: Fusing bias and GeLU into the matmul epilogue

Rungs 6 and 7 of the fusion ladder. Task 10 measured Triton's tiled matmul
losing to cuBLAS by 2.26-3.42x at batch 128 on this card (no tensor
cores). This document asks whether folding the bias add and the GeLU
activation into the matmul's epilogue -- eliminating a full DRAM
round-trip of the `[batch, 64, 768]` intermediate -- recovers any of that
gap, and separately, what it costs in register pressure.

## Setup

- GPU: GTX 1650 Ti, sm_75, 16 SMs, no tensor cores, 4 GB, peak DRAM
  bandwidth ~192 GB/s. Clocks unlocked; `sm_clock_mhz`/`temp_c` recorded
  per row from live telemetry.
- Shape: `(batch, 64, 192) -> (batch, 64, 768)`, fp32 -- the model's first
  MLP projection, matching Task 10's `k=192, n=768` shape exactly.
- Three arms, all forward-only fp32:
  - `torch_gelu`: `F.gelu(F.linear(x, w, b), approximate="tanh")` --
    cuBLAS SGEMM then a separate elementwise kernel.
  - `triton`: Task 10's unmodified `_linear_kernel`, then Task 8's
    separate `_gelu_kernel` -- two kernel launches per call, unmodified
    from their original tasks.
  - `triton_gelu`: the new fused `_linear_gelu_kernel`, one launch --
    bias add and GeLU computed on the `[M, N]` tile while it is still in
    registers, before it is ever written to DRAM.
- 30 interleaved reps per arm via `bench/harness.py::compare`.
- Commit at time of measurement: `da7c74e`.

## Measured latency (median of 30 interleaved reps, fp32)

| batch | torch_gelu ms | triton (composed) ms | triton_gelu (fused) ms | torch_gelu TFLOPs | triton TFLOPs | triton_gelu TFLOPs |
|------:|---------------:|----------------------:|-------------------------:|-------------------:|---------------:|----------------------:|
|     1 |         0.0225 |                 0.0410 |                   0.0400 |               0.838 |          0.461 |                  0.472 |
|     8 |         0.0747 |                 0.2093 |                   0.1931 |               2.021 |          0.722 |                  0.782 |
|    32 |         0.4026 |                 0.8363 |                   0.7638 |               1.500 |          0.722 |                  0.791 |
|   128 |         1.2140 |                 3.4625 |                   3.0925 |               1.990 |          0.698 |                  0.781 |
|   512 |         5.2675 |                13.7300 |                  12.5199 |               1.835 |          0.704 |                  0.772 |

TFLOPs = `2 * (batch*64) * 768 * 192 / seconds`, printed by
`bench/run_linear_gelu.py`'s own summary (matmul is compute-bound, so
GB/s alone -- also recorded in `bench/results/latency.csv` -- is
misleading here; see below).

At every batch, fusion beats the composed two-launch Triton arm
(9-11% faster at batch >= 8), and at batch 128 the fused kernel
(3.0925 ms) is even slightly faster than Task 10's *pure* unfused matmul
alone (3.1613 ms, no GeLU at all) -- the fused single launch outruns the
unfused matmul-only kernel plus its own launch overhead, before GeLU is
even added back in.

## Measured DRAM traffic (ncu, batch 128, fp32)

The composed `triton` arm launches two kernels per call
(`_linear_kernel` then `_gelu_kernel`); Part A's `expected_kernels=2`
confirmed both were captured rather than silently landing on one:

| arm | kernel(s) | bytes read | bytes written | total |
|---|---|---:|---:|---:|
| triton (composed) | `_linear_kernel` | 76,467,296 | 24,950,752 | |
| triton (composed) | `_gelu_kernel` | 25,184,672 | 24,831,328 | |
| **triton (composed) total** | | **101,651,968** | **49,782,080** | **151,434,048** |
| triton_gelu (fused) | `_linear_gelu_kernel` | 76,466,400 | 24,913,056 | **101,379,456** |

Measured traffic ratio (fused / composed) = 101,379,456 / 151,434,048 =
**0.6695** -- a 33.1% reduction, saving 50,054,592 bytes.

## Predicted vs. measured

Predicted savings, per the brief: `2 * batch * 64 * 768 * 4` bytes --
one avoided write of the GeLU input plus one avoided read of it back in.
At batch=128: `2 * 128 * 64 * 768 * 4` = 50,331,648 bytes.

Predicted ratio = (151,434,048 - 50,331,648) / 151,434,048 = **0.6676**.

Measured ratio: **0.6695**. This matches the array-pass prediction to
within 0.27% (measured savings 50,054,592 bytes vs predicted 50,331,648)
-- the same tight agreement Task 11's residual-LayerNorm fusion showed
(0.8045 measured vs 0.80 predicted). The arithmetic model continues to
hold.

## Register cost

| kernel | registers/thread | warps_active (%peak) | local ld/st (spill) |
|---|---:|---:|---:|
| `_linear_kernel` (Task 10, unfused, this run) | 128 | 49.37% | 0 / 0 |
| `_gelu_kernel` (Task 8 reference) | 27 | 90.99% | 0 / 0 |
| `_linear_gelu_kernel` (fused, this task) | 168 | 37.18% | 0 / 0 |

The epilogue adds **+40 registers/thread** over Task 10's unfused matmul
(128 -> 168, +31%), and occupancy drops correspondingly (49.37% ->
37.18% warps active). No spilling occurred (`l1tex__..._local_op_ld/st`
both 0 for both kernels) -- the extra registers are absorbed within the
no-spill budget, but at a real occupancy cost. This is the first data
point on the register-pressure curve Tasks 16-17 test: fusing more work
into an epilogue keeps growing per-thread register demand, and at some
point (not yet reached here) that will force spilling rather than just
reduced occupancy.

## Did fusion close the gap to cuBLAS?

**Partially, on latency; not on TFLOPs, and it did not close the gap
outright.**

- At batch 128, the ratio of fused-Triton to torch's *equivalent
  two-op workload* (`torch_gelu`) is 3.0925 / 1.2140 = **2.55x**,
  narrower than Task 10's 3.42x ratio for the pure matmul alone at this
  same shape. Fusion narrowed the gap by removing the composed arm's
  second launch and its DRAM round-trip, and the composed arm's own
  ratio (3.4625 / 1.2140 = 2.85x) already improves over pure-matmul-only
  3.42x, since torch_gelu also pays for its own second kernel.
- But TFLOPs tell the more honest story for a compute-bound kernel:
  fused Triton reaches only 0.781 TFLOPs at batch 128, statistically the
  same as Task 10's *unfused* matmul-alone TFLOPs (0.764) and nowhere
  near torch_gelu's 1.990 TFLOPs or Task 10's cuBLAS-alone 2.617 TFLOPs.
  Removing a DRAM round-trip does not touch the compute-bound ceiling
  this kernel was already running against -- `tl.dot` throughput here is
  gated by the same hand-tiled, no-tensor-core SGEMM Task 10 measured,
  and the epilogue fusion doesn't change how the matmul itself computes.

**The gap to cuBLAS is real and fusion did not close it.** What fusion
bought here is smaller than Task 11's LayerNorm fusion win: LayerNorm was
bandwidth-bound with genuine bandwidth headroom to reclaim, so removing a
DRAM pass moved the needle on latency directly. This matmul is
compute-bound (Task 10 established that), so removing a DRAM pass
removes real traffic -- verified above, matching the arithmetic model
almost exactly -- but traffic was never the bottleneck for the matmul
itself; it only mattered for the separate GeLU pass that fusion now
avoids launching. The result is a fused kernel that is a strict
improvement over the composed two-launch Triton path (faster, less
traffic, same correctness), but still loses to cuBLAS by more than 2x,
and the epilogue's +40 register cost is the first concrete evidence that
further fusion (attention, the second MLP projection) will keep pushing
register pressure up before it ever threatens to spill.

> **Correction (Experiment 4, 2026-08-18):** the win reported above (+9-11%
> at batch >= 8) is for the **naive, untuned** kernels only, and remains the
> correct record of what was measured at that tuning level.
> `docs/findings/07-retuning.md` found this **reverses once both the
> composed and fused kernels are autotuned**: the tuned composed arm is
> 4.7-13.8% faster than the tuned fused arm, verified on unthrottled data,
> with the effect growing with batch — the opposite of what measurement
> noise would produce. Mechanism: the autotuned fused kernel
> (`_linear_gelu_tuned_kernel`) is forced into a smaller `BLOCK_M` than the
> autotuned composed kernel (`_linear_tuned_kernel`) because it must also
> hold the GeLU epilogue's intermediate terms live alongside the matmul
> accumulator, which shrinks the autotune search's viable large-tile
> region — the same register cost identified in this document, now shown
> to bind the *tuned* search space too. See `07-retuning.md` for the full
> verification.
