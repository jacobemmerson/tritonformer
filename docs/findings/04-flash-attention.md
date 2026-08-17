# Finding 04: Flash-style fused attention

Rung 10, the headline kernel. At S=64 and head_dim=64 an entire head's Q,
K, V is 48KB in fp32 -- inside Turing's 64KB shared memory -- so the
tiling loop that real FlashAttention needs for longer sequences collapses
to a single block: one program loads the whole head, computes the score
matrix, softmax, and the output matmul entirely in registers/shared
memory, and the `[B, H, 64, 64]` score matrix never reaches DRAM.

## Framing correction (carried over from Task 13)

An earlier draft of Task 13's report wrongly claimed
`F.scaled_dot_product_attention` was unfused on this sm_75 card.
Re-verified with `torch.profiler`: SDPA runs **one** kernel,
`fmha_cutlassF_f32_aligned_64x64_rf_sm75`, PyTorch's memory-efficient
(xFormers-derived) fused backend available on sm_75 (true FlashAttention
needs sm_80+). It does not materialize the score matrix. Only
`model.baseline.layers.attention` (the `torch` arm in
`bench/run_attention.py`) is unfused by construction (four separate
kernels: softmax, scale-mul, two GEMMs).

Consequently, this kernel's DRAM-traffic win is credited against
`triton_composed`'s measured 76,306,272 bytes at batch 128 (Task 13), not
against an "unfused torch" strawman. Its latency is compared honestly
against both the unfused `torch` baseline and the fused
`fmha_cutlassF...` kernel below, and **it loses to both.**

## Setup

- GPU: GTX 1650 Ti, sm_75, 16 SMs, no tensor cores, 4 GB, ~192 GB/s peak
  DRAM bandwidth, 64KB shared memory/SM.
- Shape: `[B, 3, 64, 64]` q/k/v, fp32, `BATCHES = [1, 8, 32, 128, 512]`.
- `attention_flash`: one Triton program per `(batch, head)`, loading Q,
  K, V for the whole head (`BLOCK_S=64`, `BLOCK_D=64`), computing
  `scores = (q @ k.T) * scale`, a single-pass softmax, and `weights @ v`,
  all before ever writing to DRAM. `num_warps=4, num_stages=2`.
- Commit at time of measurement: this task's commit (see report).

## Latency across the batch sweep (median of 30 interleaved reps, fp32)

| batch | torch ms | triton_composed ms | triton_flash ms | triton_qkv_fused ms | triton_qkv_unfused ms |
|------:|---------:|--------------------:|------------------:|----------------------:|------------------------:|
|     1 |   0.0192 |               0.0206 |             0.0307 |                 0.0548 |                   0.0868 |
|     8 |   0.0369 |               0.0456 |             0.0763 |                 0.2343 |                   0.2288 |
|    32 |   0.1180 |               0.1493 |             0.2478 |                 0.8011 |                   0.8108 |
|   128 |   0.4632 |               0.5939 |             1.0016 |                 3.1862 |                   3.2282 |
|   512 |   1.8135 |               2.3900 |             4.0631 |                13.1648 |                  13.1332 |

**`triton_flash` is slower than every other arm in the table, at every
batch, including the unfused `triton_composed` baseline it was meant to
beat on latency.** At batch 128 it is 1.69x slower than `triton_composed`
(1.0016ms vs 0.5939ms) and 2.16x slower than the unfused `torch` baseline
(1.0016ms vs 0.4632ms).

A separate, informal (not CSV-recorded) comparison against
`F.scaled_dot_product_attention` at batch 128, same q/k/v tensors, using
`triton.testing.do_bench`:

| arm | ms |
|---|---:|
| `F.scaled_dot_product_attention` (`fmha_cutlassF_f32_aligned_64x64_rf_sm75`) | 0.359 |
| `attention_flash` (this kernel) | 1.169 |
| `attention_torch` (unfused baseline) | 0.472 |

`attention_flash` is **~3.26x slower than SDPA** and **~2.48x slower
than the unfused torch baseline**. This is a clean, mature CUTLASS
kernel beating a first-pass Triton flash kernel decisively on latency --
expected and stated plainly, not spun as a win.

## DRAM traffic at batch 128

`_flash_kernel` is a single kernel per call (verified empirically with
`ncu --launch-count 15`: after 7 setup kernels from tensor construction,
`_flash_kernel` repeats with cycle length 1 -- `expected_kernels=1`
confirmed the capture was complete, not a partial cycle).

| kernel | read | write | total |
|---|---:|---:|---:|
| `_flash_kernel` | 21,183,232 | 26,755,840 | **47,939,072** |

Compare to Task 13's measured `triton_composed` total at batch 128:
**76,306,272 bytes** (4 kernels: `volta_sgemm_64x64_tn`, the standalone
`* scale` multiply, `_softmax_kernel`, `volta_sgemm_64x64_nn`).

**Reduction: 47,939,072 / 76,306,272 = 0.6282 -- a 37.2% traffic cut**
(28,367,200 bytes saved), removing the score-matrix materialization and,
folded into the same kernel, the standalone `* scale` round trip Task 13
measured at 12,256,416 bytes (~32% of that arm's excess over its own
theoretical prediction).

### Traffic did not fall as far as the naive array-pass model predicts

The theoretical minimum for this shape with no score matrix at all --
3 reads of q/k/v plus 1 write of the output, `[128, 3, 64, 64]` fp32 each
-- is `4 * (128*3*64*64) * 4 = 25,165,824` bytes. Measured
(47,939,072) is **1.90x** that minimum. The overshoot is explained
below: it is not extra *logical* traffic, it is register-spill traffic.

## Achieved occupancy: shared-memory-limited at the block level, but register-spill-limited in practice

Measured via `ncu`: `sm__warps_active.avg.pct_of_peak_sustained_active`
= **12.48%**, `launch__registers_per_thread` = **255** (the hardware
maximum for sm_75 -- the compiler hit the ceiling, not a value it chose
freely), and `l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum` /
`..._st.sum` = **39,714,816 / 20,840,448 bytes** -- nonzero, meaning the
kernel genuinely spills to local memory. Task 12's `_linear_gelu_kernel`
(168 regs, no spilling) never crossed this line; this kernel does.

**Shared memory arithmetic (the constraint the brief anticipated):**
Q + K + V per block = `3 * 64 * 64 * 4` = 49,152 bytes = exactly 48KB.
Even at Turing's full 64KB/SM, two resident blocks would need 98,304
bytes > 65,536 available, so shared memory permits **at most 1 block/SM**
regardless of registers.

**Register arithmetic (checked independently, per the brief's
instruction to work out which resource actually binds):** 255
regs/thread x 128 threads/block (`num_warps=4`) = 32,640 registers/block.
Turing's 65,536 registers/SM / 32,640 = 2.008 -> registers alone would
permit **2 blocks/SM**, one more than shared memory allows.

**Shared memory is therefore the binding constraint on block count**, as
the brief predicted -- 1 resident block/SM, exactly matching the
measured 12.48% warps-active (1 block x 4 warps / 32 max warps/SM =
12.5%, within rounding of the measured 12.48%).

But register pressure is not merely a secondary story here: the kernel
needed *more* than 255 registers/thread to hold Q, K, V, the full 64x64
score matrix, and the softmax intermediates simultaneously, hit the
hardware cap, and spilled the remainder to local memory. That spill
traffic (39.7M load + 20.8M store L1 bytes attributable to local memory
ops) is the direct cause of the 1.90x overshoot above the 25.17M-byte
theoretical minimum, and a likely major contributor to the latency loss
independent of the 1-block/SM occupancy ceiling.

**Spill traffic vs. DRAM traffic saved -- the document's punchline:**

- local-memory (spill) traffic: 39,714,816 + 20,840,448 = **60,555,264
  bytes**
- DRAM traffic saved vs `triton_composed`: 76,306,272 - 47,939,072 =
  **28,367,200 bytes**
- ratio: **2.13x** -- the fusion removed ~28.4 MB of DRAM traffic and
  introduced ~60.6 MB of L1 local-memory traffic in its place.

Caveat, so this isn't overstated: `l1tex__t_bytes_pipe_lsu_mem_local_op_ld/st.sum`
is measured at L1, not at DRAM -- some of that spill traffic may be
absorbed by L1/L2 and never actually reach DRAM, so it is **not**
directly comparable to `dram__bytes_*` as a like-for-like byte count.
Read the 2.13x figure as indicative of the mechanism (fusion traded
traffic it could avoid for traffic through a costlier part of the memory
pipeline), not as "60 MB of extra DRAM traffic." It is, however, the
single clearest explanation on hand for why a kernel that cuts measured
DRAM traffic by over a third still loses on latency: the traffic it
removed was cheap (sequential DRAM reads/writes); the traffic it
introduced replacing it runs through the local-memory spill path instead.

## Did occupancy of ~1 block/SM hurt on a 16-SM card?

Yes, unambiguously, and it compounds with the register spilling rather
than being the only cause. With `BLOCK_S = BLOCK_D = 64` the kernel
issues one block per `(batch, head)` -- 384 blocks at batch 128, plenty
of blocks to fill 16 SMs' worth of *serial* work, but each SM can only
run one of them **at a time** because of the 48KB shared-memory
requirement, and that resident block only occupies 4 of the SM's 32
warp slots (12.5%). There is no second resident block available to hide
the first one's stalls (memory latency, or here, spill-load/store
latency) the way occupancy-based latency hiding normally works. The
16-SM card has plenty of parallelism *across* blocks, but each
individual SM is running one severely under-occupied, register-starved
block, and that is consistent with losing on latency to both the
4-kernel unfused baseline and to SDPA's mature CUTLASS kernel, which
presumably manages its register/shared-memory budget far more
carefully (likely via a k-block loop even at this trivial size, or a
tighter register allocation that avoids spilling entirely).

## Untested masking path

`_flash_kernel`'s `-inf` masking (for `seq_len < BLOCK_S`) is exercised
nowhere in this codebase: `seq_len` is 64 everywhere `attention_flash` is
called, equal to `BLOCK_S = triton.next_power_of_2(64) = 64`, so `mask_s`
is always all-True and the masking branch is never taken. The masking
logic is structurally correct (verified by inspection: a masked row still
retains an unmasked lane for the max/softmax reduction to anchor on), but
it has zero test coverage today. Worth knowing if the sequence length
this project uses ever changes to something that isn't a power of 2, or
to a size where `BLOCK_S` must exceed `seq_len`.

## Summary: what this kernel actually establishes

- **Wins**: DRAM traffic against `triton_composed`, 37.2% lower
  (76,306,272 -> 47,939,072 bytes), genuinely avoiding score-matrix
  materialization (`test_flash_never_materializes_scores` is the
  intended discriminator for this -- see the test-suite note below) and
  removing the standalone `* scale` round trip.
- **Loses**: latency, at every batch, against `triton_composed`
  (1.69x slower at batch 128), the unfused `torch` baseline (2.16x-2.48x
  slower), and decisively against `F.scaled_dot_product_attention`'s
  fused CUTLASS kernel (~3.26x slower). The cause is not the fusion
  concept itself but this specific kernel's implementation: loading the
  entire 64x64 Q/K/V/scores working set into registers at once drives
  register demand past the 255-register hardware cap, forcing spills,
  and shared memory limits the SM to a single, severely under-occupied
  resident block (12.5% warps active) with no second block available to
  hide that spill latency.
- This is the project's clearest illustration so far that **reducing
  DRAM traffic and improving latency are not the same axis**: this rung
  cuts traffic by over a third and is still the slowest arm measured.

## Test-suite note: `test_flash_never_materializes_scores`

This test, specified verbatim in the task brief, compares
`peak_memory_after - peak_memory_before` against
`score_bytes = 256 * 3 * 64 * 64 * 4` using strict `<`. At S = head_dim =
64, `score_bytes` is numerically identical to the byte size of the
kernel's own output tensor (`torch.empty_like(q)`, same shape as `q`).
Any correct implementation must allocate that output tensor, so
`peak - baseline` can never be strictly less than `score_bytes` -- it is
*exactly* `score_bytes` for a kernel that allocates nothing else beyond
its output (confirmed: measured `peak - baseline` = 12,582,912 bytes =
`score_bytes` exactly, both in isolation and in the full suite run).
This is a structural flaw in the test's threshold, not a kernel defect:
the kernel demonstrably never allocates a separate `[B, H, S, S]` score
buffer (confirmed independently by the `ncu` DRAM traffic capture above,
which shows a single kernel touching only Q/K/V/O-sized traffic, no
intermediate score tensor). Per instruction, the test was not loosened
or modified -- it is left failing and flagged here as a finding for
review, rather than silently patched.
