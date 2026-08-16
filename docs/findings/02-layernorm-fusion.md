# Finding 02: Fusing the residual add into LayerNorm

Rung 2 of the fusion ladder, and the first *real* fusion in this project
(Task 8's GeLU rewrite fused nothing — there was nothing to fuse). This
document measures what folding `x + residual` into the LayerNorm kernel
actually buys, in both latency and DRAM traffic, and checks the
measurement against the arithmetic prediction rather than the brief's
initial "roughly a third" hypothesis.

## Setup

- GPU: GTX 1650 Ti, sm_75, 16 SMs, no tensor cores, 4 GB, peak DRAM
  bandwidth ~192 GB/s. Clocks unlocked (`scripts/lock_clocks.sh` not run —
  needs sudo); `sm_clock_mhz`/`temp_c` recorded per row from live
  telemetry.
- Shape: `(batch, 64, 192)` fp32. `bytes_theoretical = 4 * batch * 64 * 192
  * 4` (read x, read residual, write the summed residual, write the
  normed output).
- Three arms, all forward-only fp32:
  - `torch`: `updated = x + residual`, then `F.layer_norm(updated, ...)`.
  - `triton`: same separate add, but Task 7's unfused `_layernorm_kernel`
    for the normalize step. Unmodified — Task 7's kernel is untouched by
    this task.
  - `triton_residual`: the new fused `_layernorm_residual_kernel`, one
    kernel launch, one read of `x`, one read of `residual`, one write of
    the summed residual, one write of the normed output.
- 30 interleaved reps per arm via `bench/harness.py::compare`.
- Commit at time of measurement: `ae221bd`.

## Measured latency (median of 30 interleaved reps, fp32)

| batch | torch ms | triton ms | fused ms | torch GB/s | triton GB/s | fused GB/s | fused vs triton |
|------:|---------:|----------:|---------:|-----------:|------------:|-----------:|-----------------:|
|     1 |   0.0096 |    0.0083 |   0.0057 |       20.5 |        23.6 |       34.3 |             0.69x |
|     8 |   0.0209 |    0.0171 |   0.0129 |       75.3 |        92.1 |      122.1 |             0.75x |
|    32 |   0.0712 |    0.0532 |   0.0410 |       88.3 |       118.2 |      153.6 |             0.77x |
|   128 |   0.2656 |    0.1938 |   0.1516 |       94.7 |       129.9 |      166.1 |             0.78x |
|   512 |   1.0623 |    0.7660 |   0.5913 |       94.8 |       131.4 |      170.3 |             0.77x |

("fused vs triton" = fused latency / unfused-triton latency; lower is
better for the fused kernel.)

At batch 512 the fused kernel reaches 170.3 GB/s achieved, essentially at
the same ~169-170 GB/s ceiling GeLU and the Task 7 LayerNorm both
converged to — this is a raw-copy-bound ceiling on this card, not
something specific to LayerNorm. Fusion gets there in one launch instead
of two-plus-`F.layer_norm`'s-internal-kernels.

## Measured DRAM traffic (ncu, batch 128, fp32)

`profile_kernel` captures one steady-state launch window. The unfused
`triton` arm issues two kernel launches per call (a `torch.add`
elementwise kernel, then `_layernorm_kernel`), so `launch_count=2` was
used to capture both kernels in the same profiling window rather than
just whichever one the skip count happened to land on:

| arm | kernel(s) | bytes read | bytes written | total |
|---|---|---:|---:|---:|
| triton (unfused) | `torch.add` elementwise | 12,592,960 | 6,011,424 | |
| triton (unfused) | `_layernorm_kernel` | 6,312,736 | 6,002,656 | |
| **triton (unfused) total** | | **18,905,696** | **12,014,080** | **30,919,776** |
| triton_residual (fused) | `_layernorm_residual_kernel` | 12,599,456 | 12,274,816 | **24,874,272** |

Measured traffic ratio (fused / unfused) = 24,874,272 / 30,919,776 =
**0.8045** — a 19.6% reduction.

## Predicted vs. measured

The brief's initial hypothesis was "DRAM traffic drops by roughly a
third." The correct arithmetic, worked from array-passes rather than
guessed, says otherwise:

- Unfused: (read x, read residual, write sum) + (read sum, write normed)
  = 5 full-array passes.
- Fused: (read x, read residual, write sum, write normed) = 4 full-array
  passes.
- Predicted ratio = 4/5 = **0.80**, i.e. a 20% reduction — not a third.

Measured ratio: **0.8045**. This matches the 0.80 array-pass prediction
to within 0.6%, not the ⅓ figure from the brief. The "roughly a third"
hypothesis was simply the wrong arithmetic — it looks like it assumed the
unfused path materializes the sum tensor 1.5x (3 passes over it) relative
to the fused path's 1x, which isn't what either implementation does; both
touch the summed-residual tensor exactly once per read and once per
write, unfused or fused. There's no L2-absorption story needed here: the
tensors at batch 128 (128×64×192×4 bytes ≈ 6.3 MB per tensor) are larger
than this card's L2 (a few MB), so the intermediate genuinely round-trips
through DRAM in the unfused path, and removing that round trip is exactly
what the byte counters show.

## Did fusion help?

**Yes, measurably, on both axes:**

- **Latency**: the fused kernel is 22-31% faster than the unfused Triton
  pair across the sweep (23% at batch 512), and reaches the same ~170
  GB/s ceiling the unfused arm only approaches from below (131 GB/s).
- **Traffic**: DRAM bytes drop by 19.6%, matching the 20% array-pass
  prediction almost exactly.

This is a different outcome from Task 8's GeLU rung, where there was no
second pass to eliminate and Triton could not beat `F.gelu`'s
already-single-pass kernel. Here there genuinely were two passes over the
same intermediate tensor (write it, then read it back), and collapsing
them into one kernel removed exactly the pass the arithmetic said it
would. Unlike Task 9's softmax, which lost because of occupancy
starvation, this kernel keeps the same one-row-per-program structure and
the same `BLOCK=256`/`num_warps=4` shape as Task 7's verified-correct
LayerNorm, so there was no separate occupancy story to introduce — the
win is attributable to the traffic reduction alone.

The unfused Triton arm was NOT already sitting at the bandwidth ceiling
(131 GB/s vs a ~169-170 GB/s ceiling) the way GeLU was, which is exactly
why there was room for a fusion win here: two real kernel launches with a
real intermediate tensor to eliminate, not a single already-optimal
elementwise pass.
