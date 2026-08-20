# Tritonformer

This project reimplements the **forward pass** of a shallow vision
transformer using Triton kernels, and measures when kernel fusion helps
and when it hurts.

Primary hardware: GTX 1650 Ti (sm_75, 16 SMs, ~1 MB L2, 4 GB, no tensor
cores). Replicated on an NVIDIA L4 (sm_89, 58 SMs, 48 MB L2, 24 GB, with
tensor cores) via Modal.

Objectives:

a) Implement kernels: LayerNorm, Linear, GeLU, Softmax
b) Fuse them along a ladder from single-op to a fully fused transformer block
c) Measure the tradeoffs, backed by latency and hardware-counter data

## Findings

### 1. Exactly one fusion paid

| fusion | registers before/after | spill | result |
|---|---|---|---|
| layernorm + residual | 28 to 27 | 0 | **wins 22-31%** at every batch |
| linear + gelu (naive) | 128 to 168 | 0 | wins 7-14% |
| linear + gelu (autotuned) | 255 to 255 | 131 MB | **loses 9-13%** |
| composed to flash attention | to 255 (cap) | 473 KB | **loses 1.49-2.24x** |
| composed to mega-MLP | 128/168 to 226 | 0 | **loses 3.10-3.83x** |

`layernorm_residual` is the only winner. It uses one *fewer* register than
the kernel it fuses into and deletes a full DRAM round-trip. Every fusion
that bought traffic with register pressure or a serialized reduction lost.

Note that epilogue fusion reverses under tuning: `linear_gelu` wins 7-14%
naive but loses 9-13% once both arms are autotuned, because the fused
kernel hits the 255-register cap and spills 131 MB while the unfused one
hits 255 with zero spill.

### 2. The mechanism is loop serialization, not registers

The original claim was that the mega-MLP was register-limited. Two
designed interventions falsified it:

- Cutting registers from 226 to 128, which the project's own occupancy
  arithmetic says should restore occupancy to 50%, left measured occupancy
  pinned at **25.00%**. Registers were never binding.
- Halving shared memory via `BLOCK_H` produced the same null result and
  made latency *worse* (6.05x).

What survived: latency tracks H-loop iteration count (24 to 48). Setting
`num_stages` to 1, 2 or 3 changed nothing (0.003% spread, identical
compiled metadata) because the loop is compile-time unrolled, so there is
no pipelining to enable.

The occupancy arithmetic itself remained accurate at 128, 168, 226 and 255
registers per thread. Only the conclusion that registers were *binding*
for the mega-MLP was withdrawn.

### 3. `torch.compile` beat every hand-written arm

Given only the plain eager model, Inductor was faster than both Triton
arms at every batch on both cards. It won by **declining to fuse**:
`grep -c "tl.dot"` returns 0 in its generated code. It falls back to
cuBLAS for GEMMs, fuses layernorm and residual, and declines the MLP's
serial-loop shape, with no knowledge of this project's hypothesis. Against
eager torch it wins at small batch and loses at batch 512 (1.31x on the L4).

### 4. The conclusions survive a larger, tensor-core GPU

Cost relative to each card's own eager `torch` baseline, where above 1.0
means slower than torch:

| card | batch 1 | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| 1650 Ti composed | 2.26x | 2.15x | 1.95x | 2.27x | 1.91x |
| 1650 Ti fused | 5.01x | 5.27x | 5.07x | 5.93x | 4.74x |
| L4 composed | 2.05x | 1.80x | 1.67x | 1.38x | 1.43x |
| L4 fused | **1.80x** | 1.58x | 3.95x | 2.68x | 2.57x |

Over-fusion is not an artifact of a small, slow, tensor-core-less laptop
GPU. The penalty shrinks (5.93x to 2.68x at batch 128) but never inverts.
There is one rank flip, at batch 1, where fused (1.80x) beats composed
(2.05x); that flip is measured but not explained.

Fusion's DRAM-read advantage persists on every rung: 1.50x to 1.55x
(layernorm_residual), 2.47x to 2.02x (attention), 3.54x to 4.94x (mlp).
The pre-registered prediction that a 48 MB L2 would absorb the
intermediates and make fusion pointless is **broken**, by not happening
rather than by reversing.

### 5. The kernels are not TF32-safe, and one card could not reveal it

Triton's `tl.dot` defaults to TF32 on tensor-core hardware. sm_75 has
none, so every measurement in this project is IEEE fp32 *by accident of
hardware*, not by choice. On sm_89 that default binds: **70 of 153 tests
fail** at maximum absolute error 2.5e-3 against a 1e-4 tolerance, and no
kernel declares a precision policy. All 153 pass under
`TRITON_F32_DEFAULT=ieee`.

This was not one of the four pre-registered predictions and is the most
consequential result of the replication.

At matched precision cuBLAS still wins: Triton is 36-90% slower with both
at IEEE, and 3-45% slower with both at TF32. The apparent Triton win of
12-18% was a mismatched comparison (Triton on tensor cores against cuBLAS
on IEEE) and is withdrawn.

### 6. Four results are measured and unexplained

Stated plainly rather than given a mechanism the data does not support:
`mlp`'s 81.4% fused-arm traffic collapse, `layernorm_residual`'s win
eroding to -11% at batch 128 while its traffic ratio barely moves,
`attention_flash`'s L4 win despite traffic being flat at +0.9%, and the
batch-1 rank flip. Two carry named candidate hypotheses, both labelled
untested; two carry none.

## Method notes

- **Predictions were pre-registered and committed before measuring.** The
  committed version of `docs/findings/09-l4-replication.md` is a
  byte-identical prefix of the final one, so the predictions provably were
  not edited after the results arrived.
- **Throttling is disclosed, not hidden.** 60 of 405 sm_75 latency rows
  are flagged, with clocks spanning 300-1950 MHz. Six `linear_gelu`
  cross-card speedups are withdrawn because their sm_75 side ran at
  300 MHz, roughly 23% of nominal.
- **One instrumentation bug is documented as a result.** An ncu capture
  window landed inside benchmark tensor setup and profiled a torch
  elementwise multiply instead of the fused kernel under test. It was
  caught only because the two cards disagreed implausibly. The fix adds
  kernel *identity* validation to `bench/collect_counters.py`; a launch
  *count* check could not catch it, and did not.

## Reading order

1. [`docs/findings/06-synthesis.md`](docs/findings/06-synthesis.md):
   the original answer to the research question.
2. [`docs/findings/10-register-rule.md`](docs/findings/10-register-rule.md):
   two experiments that stress-tested and revised that synthesis.
3. [`docs/findings/08-inductor.md`](docs/findings/08-inductor.md):
   what `torch.compile` chose to fuse, and what it declined.
4. [`docs/findings/09-l4-replication.md`](docs/findings/09-l4-replication.md):
   the L4 replication, with all four predictions scored.

Read `06` before `10`: `06` states the original picture and `10` corrects it.

## Scope

Forward-pass inference only. There is no training loop in this codebase.
A Triton training loop and optimizer was an original stretch objective and
was not built; it remains future work, not a current objective.
