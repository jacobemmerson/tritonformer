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

![Fusion ladder: latency of each fused or tuned kernel relative to the unfused kernel(s) it replaces, on the GTX 1650 Ti](assets/fusion-ladder.svg)

## The fusion ladder, measured

Every row is naive (untuned) Triton against the matching PyTorch op,
unless marked otherwise. "Result" is the fused/combined kernel's latency
relative to the equivalent unfused kernel(s), not relative to PyTorch.

| rung | what changed | result |
|---|---|---|
| GeLU alone | Triton rewrite, no fusion | no change (0-5%, noise). Both are already one read + one write; there was nothing to remove |
| LayerNorm + residual add | one kernel instead of two | **22-31% faster**, DRAM traffic down 19.6% |
| Linear + GeLU epilogue (naive) | GeLU folded into the matmul's output tile | **7-14% faster**, DRAM traffic down 33% |
| Linear + GeLU epilogue (autotuned) | same fusion, both arms autotuned | **9-13% slower**. The fused kernel needs a smaller tile to hold the GeLU math alongside the matmul accumulator, so it loses out on the large-tile configs the unfused matmul can use |
| Flash-style attention | Q/K/V/softmax/output in one kernel, no score matrix in DRAM | **1.49-2.24x slower**, despite 37% less DRAM traffic. It spills to local memory (hits the 255-register hardware cap) and only one block fits per SM (12.5% occupancy) |
| Whole-MLP mega-kernel | linear + GeLU + linear in one kernel | **3.10-3.83x slower** at every batch, despite up to 73% less DRAM traffic. See "loop serialization" below |
| Full fused transformer block | LayerNorm + residual + attention + MLP, all fused | loses; inherits the attention and MLP kernels' losses |

Two rungs are not fusion, just autotuning a single kernel (`triton.autotune`
over tile sizes, plus the standard L2 pid-swizzle):

| kernel | naive vs. torch | tuned vs. torch |
|---|---|---|
| softmax | 1.29-3.74x slower | reaches parity at every batch, **beats torch below batch 32** |
| linear (plain matmul) | 1.9-4.3x slower | 1.13-4.25x slower. Tuning recovers roughly half the gap but never closes it; cuBLAS still wins on every clean measurement |

`layernorm_residual` is the only fusion that wins outright, and it is
also the only one that costs *fewer* registers than the kernel it folds
into (28 to 27). Every fusion that bought a smaller DRAM footprint with
more registers or a slower, serialized reduction lost, even when the
traffic drop was large.

## Six things worth knowing

### 1. Cutting DRAM traffic did not reliably cut latency

Across every fused kernel, the correlation between "less DRAM traffic"
and "less time" breaks down. The flash-attention kernel cuts traffic 37%
and is still 1.49-2.24x slower. The mega-MLP kernel cuts traffic up to
73% and is still 3.10-3.83x slower. Traffic and latency are separate
axes; a kernel can win on one and lose badly on the other.

### 2. The mega-MLP's slowdown is a serialized loop, not a register limit

The first explanation tried was "too many registers, too little
occupancy." Two tests ruled that out:

- Cutting registers from 226 to 128 (which should, by the project's own
  occupancy arithmetic, double occupancy from 25% toward 50%) left
  measured occupancy pinned at exactly 25.00%. Registers were not the
  binding constraint.
- Halving shared memory produced the same null result and made latency
  *worse* (6.05x slower).

What actually predicts latency is the number of iterations in the
kernel's inner loop over the hidden dimension (24 vs. 48). Changing
`num_stages` (1, 2, or 3) changed nothing (a 0.003% spread) because the
loop is unrolled at compile time, so there is no pipelining left to
enable. The occupancy math itself was correct at every register count
tested; only the claim that registers were the *bottleneck* was wrong.

### 3. `torch.compile` beat every hand-written Triton kernel here, by declining to fuse

Given only the plain eager PyTorch model, Inductor (torch.compile's
backend) ran faster than both hand-written Triton arms at every batch on
both cards. Its generated code contains zero `tl.dot` calls: it falls
back to cuBLAS for the matmuls, fuses LayerNorm with the residual add
(the one fusion this project also found to be a genuine win), and
declines to build the MLP's serial loop. It made those choices with no
knowledge of this project's results. Against eager PyTorch it wins at
small batch and loses at batch 512 (1.31x slower on the L4).

### 4. The results hold on a bigger, tensor-core GPU

Cost relative to each card's own eager-PyTorch baseline (above 1.0 means
slower than plain PyTorch):

| card | batch 1 | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| 1650 Ti, composed (unfused) kernels | 2.26x | 2.15x | 1.95x | 2.27x | 1.91x |
| 1650 Ti, fully fused | 5.01x | 5.27x | 5.07x | 5.93x | 4.74x |
| L4, composed (unfused) kernels | 2.05x | 1.80x | 1.67x | 1.38x | 1.43x |
| L4, fully fused | 1.80x | 1.58x | 3.95x | 2.68x | 2.57x |

Over-fusion is not something specific to a small, slow, no-tensor-core
laptop GPU. The penalty shrinks on the bigger card (5.93x down to 2.68x
at batch 128) but never flips to a win. There is one exception, at
batch 1, where fused (1.80x) beats composed (2.05x); it was measured but
the cause was not investigated.

Fusion's traffic advantage holds on both cards at every rung:
1.50-1.55x less traffic for LayerNorm+residual, 2.02-2.47x for attention,
3.54-4.94x for the MLP. The original guess was that the L4's much larger
L2 cache (48 MB vs. ~1 MB) would absorb the intermediate tensors and make
fusion's traffic advantage pointless. That did not happen.

### 5. The kernels give wrong-precision answers by default on tensor-core hardware, and the small GPU could not have caught it

Triton's `tl.dot` defaults to TF32 on hardware that has tensor cores. The
1650 Ti has none, so every number measured on it is full-precision fp32
by accident of the hardware, not by any choice made in the code. On the
L4, that default matters: **70 of 153 tests fail**, with a worst-case
error of 2.5e-3 against a 1e-4 tolerance, and no kernel in this project
declares what precision it expects. Setting `TRITON_F32_DEFAULT=ieee`
fixes all 153.

This was not something the project set out to check. It is the most
consequential result of putting the code on a second GPU.

Once both sides are compared at matched precision, cuBLAS still wins:
Triton is 36-90% slower at matched full precision (IEEE), and 3-45%
slower at matched reduced precision (TF32). An earlier read of the data
said Triton beat cuBLAS by 12-18%; that was comparing Triton on tensor
cores against cuBLAS forced to full precision, not a fair comparison, and
is withdrawn.

### 6. Four results are measured but not explained

Listed plainly rather than given a story the data does not support:

- The MLP kernel's DRAM traffic collapses 81.4% on the fused arm, more
  than the arithmetic predicts.
- LayerNorm+residual's win shrinks to -11% (a loss) by batch 128, even
  though its traffic ratio barely changes across the same range.
- The flash-attention kernel wins on the L4 despite its traffic ratio
  staying flat (+0.9%).
- The batch-1 rank flip in item 4 above.

Two of these have a candidate explanation that has not been tested yet;
two have none.

## How this was measured

- **Predictions were written down before measuring, not after.** The
  committed version of the L4 replication write-up is a byte-identical
  prefix of the final one, so the predictions provably were not edited
  once results were in.
- **Clock throttling is disclosed, not hidden.** 60 of 405 latency rows
  on the 1650 Ti are flagged for clock drift, with observed clocks
  ranging from 300 MHz to 1950 MHz under sustained load and thermal
  limits. Six `linear_gelu` cross-card comparisons are withdrawn because
  the 1650 Ti side of them ran at roughly 23% of its nominal clock.
  Getting this flag to work correctly took three separate bug fixes: the
  card exposes no queryable clock lock, so the target has to be supplied
  by the operator rather than auto-detected; the original code sampled
  clock speed *after* the benchmark loop finished, by which point the
  card had already recovered; and the first fix for that used a slow
  polling call that itself perturbed the measurement, and was replaced
  with a cheaper one that could sample every rep instead of every fifth.
- **One measurement bug was caught and is reported as a result, not
  quietly fixed.** A hardware-counter capture window landed inside a
  benchmark's tensor setup and ended up profiling a plain elementwise
  multiply instead of the fused kernel under test. It was only caught
  because the two GPUs disagreed in a way that made no sense. The fix
  adds a kernel-identity check to the counter-collection script; a
  simple launch-count check would not have caught this bug, and did not.

## Scope

Forward-pass inference only. There is no training loop in this codebase.
A Triton training loop and optimizer was an original stretch goal and was
not built; it remains future work, not something attempted here.

Detailed per-experiment logs (register sweeps, hardware-counter dumps,
full prediction/verdict writeups for each rung) are kept as local
research notes and are not part of this repository's tracked history.
The numbers above are the results from those notes; this file is the
complete, standalone record of what was found.
