# Finding 01: Elementwise GeLU shows no improvement (as expected)

Rung 3 of the fusion ladder. Task 8's stated expectation was that a Triton
GeLU would show **no improvement** over `F.gelu(approximate="tanh")`. It
did not. This document records the measurement and explains why that
result is correct rather than a failure to optimize.

## Setup

- GPU: GTX 1650 Ti, sm_75, 16 SMs, no tensor cores, 4 GB, peak DRAM
  bandwidth ~192 GB/s.
- Clocks unlocked (`clocks.applications.graphics` reports `[N/A]`;
  `scripts/lock_clocks.sh` was not run — it requires sudo). `sm_clock_mhz`
  and `temp_c` are recorded per row from live telemetry, and each row's
  `flagged` is trivially `False` because clock-drift flagging is disabled
  when no locked clock is set.
- Shape: `(batch, 64, 768)` fp32, `bytes_theoretical = 2 * batch * 64 * 768 * 4`
  (one read + one write of the full tensor — GeLU is purely elementwise,
  so this is the exact byte count, not an estimate).
- 30 interleaved reps per arm via `bench/harness.py::compare`.
- Commit at time of measurement: `05790e2` (pre-existing tree state; this
  task's own commit follows below).

## Measured latency (median of 30 interleaved reps, fp32)

| batch | torch ms | triton ms | ratio (triton/torch) | torch GB/s | triton GB/s | torch %peak | triton %peak |
|------:|---------:|----------:|----------------------:|-----------:|------------:|------------:|--------------:|
|     1 |   0.0061 |    0.0064 |                  1.05 |       64.0 |        61.0 |       33.3% |         31.8% |
|     8 |   0.0215 |    0.0221 |                  1.03 |      146.4 |       142.6 |       76.3% |         74.3% |
|    32 |   0.0768 |    0.0772 |                  1.01 |      163.7 |       163.0 |       85.3% |         84.9% |
|   128 |   0.2978 |    0.2979 |                  1.00 |      169.0 |       169.0 |       88.0% |         88.0% |
|   512 |   1.1806 |    1.1801 |                 1.00  |      170.5 |      170.6 |       88.8% |         88.9% |

At every batch size the two arms are within a few percent of each other,
well inside run-to-run noise on an unlocked-clock card (the small-batch
gap at `batch=1` is launch-overhead noise on a ~6 microsecond kernel, not
a bandwidth effect). At the largest batch — where launch overhead is
amortized and the measurement is most trustworthy — the two implementations
are statistically indistinguishable: triton is 0.04% faster, which is
noise, not a result.

Both arms plateau at ~170 GB/s, about 89% of the ~192 GB/s peak. Neither
implementation reaches full peak, which is expected: no realistic
"read+write, no reuse" kernel hits 100% of nameplate bandwidth, because
achieving it requires a large enough working set that ramp-up and
tail effects wash out, plus perfect coalescing and zero contention. At
128 and 512 the tensor is large enough that the two curves have fully
converged.

## Why there was no headroom to recover

`F.gelu(approximate="tanh")` is already the case Task 7 (LayerNorm) tells
us to watch for the opposite of: LayerNorm's PyTorch implementation is
slow (60–81 GB/s, well under the 169 GB/s a bare `.clone()` achieves)
because it is a generic, multi-pass, multi-kernel reduction op — mean,
then variance, then normalize, then affine — with intermediate tensors
and kernel-launch overhead compounding at every stage. A Triton rewrite
that fused those passes into one kernel had ~2x of real headroom to
recover, and did (1.89x measured).

GeLU has none of that structure. It is a single elementwise transcendental
applied to each element once — no reduction, no multi-pass, no
intermediate tensor materialized between stages. `F.gelu` on CUDA already
lowers to one fused elementwise kernel: one read of `x`, one write of
`out`, done. There is no second pass to eliminate, no intermediate buffer
to keep resident in registers instead of round-tripping through DRAM, and
no kernel-launch chain to collapse — there is already only one launch.
Both the torch and Triton kernels are bandwidth-bound doing the exact
same memory traffic, so they converge to the same latency because they
are, at the memory-traffic level, the same computation.

**If both implementations are bandwidth-bound and near peak, what could a
Triton rewrite possibly have improved?** Nothing, on its own. A rewrite of
an already-memory-bound, already-single-pass kernel cannot beat the
memory system by writing different instructions to do the same reads and
writes — bandwidth is bandwidth regardless of which compiler emitted the
load/store. The only way an elementwise op like GeLU becomes a win for
Triton is if it is **fused with a neighboring op** so that an intermediate
tensor that would otherwise round-trip through DRAM (e.g. `linear ->
gelu -> linear` in the MLP block) instead stays in registers/shared
memory between stages, cutting real memory traffic rather than just
re-expressing the same traffic in a different kernel. That is precisely
the motivation for the next rungs of the fusion ladder: fusing GeLU into
the MLP's matmul epilogue, where there is an actual second read/write to
eliminate, is where a win becomes possible. GeLU on its own, benchmarked
in isolation, is the honest baseline that shows there was no such
opportunity to begin with.

## Hardware counters: DEFERRED

Step 5 of the task brief calls for capturing DRAM counters via
`bench/profile.py::profile_kernel`, which shells out to `ncu`. On this
host:

```
$ command -v ncu
$ echo $?
1
```

`ncu` (Nsight Compute) is not installed, and this task does not install
it (no sudo, no package installation permitted). Counter collection for
GeLU is therefore **DEFERRED** until Nsight Compute is available on this
host. The latency sweep above is unaffected — `do_bench` measures wall
time directly and has no dependency on `ncu` — and is sufficient on its
own to support the finding, since the achieved-GB/s-vs-peak comparison
already demonstrates both arms are bandwidth-bound without needing
per-kernel DRAM-sector counters to confirm it.

## Bottom line

The negative result predicted by the task brief held: Triton GeLU is not
faster than `F.gelu(approximate="tanh")` because there was nothing to
recover — both are the same one-pass, bandwidth-bound elementwise kernel
running at ~89% of peak DRAM bandwidth. This is not a wasted rung; it is
the control measurement that makes the eventual MLP-fusion win (or loss)
interpretable.
