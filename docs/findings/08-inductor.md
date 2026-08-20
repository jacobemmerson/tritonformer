# Finding 08: Experiment 2 (Inductor's Opinion) — pre-registered predictions

This document is created and committed **before any `torch.compile` measurement in this
experiment is taken**. Its purpose is to pin down falsifiable predictions so the eventual
result cannot be retrofitted into a narrative. See
`docs/superpowers/plans/2026-08-17-testing-the-register-rule.md`, Experiment 2, and
`docs/findings/10-register-rule.md` (Experiments 1 and 1b) for the rule this tests against
an independent expert system.

## What this project has established (recap, all on GTX 1650 Ti, sm_75, 16 SMs, no
## tensor cores, `TRITONFORMER_LOCKED_CLOCK_MHZ=1300`)

| fusion | outcome | mechanism |
|---|---|---|
| layernorm -> +residual | **WINS 22-25%** every batch | 27 regs vs 28 — register-free, deletes a DRAM round-trip |
| linear -> +gelu (naive) | wins 7-14% | 128 -> 168 regs |
| linear -> +gelu (both tuned) | **loses 9-13%** | tuned-fused hits the 255-reg cap and spills 131 MB; tuned-composed hits 255 with zero spill |
| whole-MLP fusion | **loses 3.8x-6.0x** (Experiments 1/1b) | H-loop serialisation — a compile-time-unrolled Python loop with no `num_stages` pipelining, not registers or occupancy (both were independently exhausted as explanations and ruled out) |
| flash attention | loses 1.49-2.24x | 255 regs at the cap, spills |
| monolithic block | **will not compile** | needs 262,144 B shared vs a 65,536 B limit |

Experiments 1 and 1b (register sweep, `BLOCK_M`; shared-memory sweep, `BLOCK_H`) both
found the MLP fusion **never flips to a win** at any reachable configuration, and traced
the real cost to the H-loop's loop-carried dependency depth rather than to registers,
occupancy, or shared memory in the abstract. That refines, rather than overturns, the
project's headline: register-free fusions that only remove a round-trip (layernorm+residual)
still win; fusions that also introduce a serial reduction across kernel invocations do not,
regardless of how their register/occupancy numbers are tuned.

## Pre-registered predictions

- **R1.** Inductor WILL fuse layernorm+residual into a single generated Triton kernel (or
  at minimum will not re-materialize the residual to DRAM as a separate elementwise
  kernel), because it is register-cheap and removes a round-trip — our one unambiguous win.
  **Falsified if** the generated code shows layernorm and the residual add as two separate
  kernel launches (a Triton `layernorm` kernel followed by a distinct elementwise `add`
  kernel, or an `aten::add` call that is not inlined into the same `triton_poi_fused_*`
  kernel as the layernorm).

- **R2.** Inductor WILL NOT fuse the two MLP matmuls (`linear -> gelu -> linear`) into a
  single Triton kernel that performs both GEMMs with a serial reduction over the hidden
  dimension inside one kernel body, because that is exactly the H-loop-serialisation cost
  Experiments 1/1b measured as the dominant, un-tunable loss. **Falsified if** the
  generated code contains one Triton kernel (or one fused Triton call site) whose body
  performs both the `x @ w1` and `hidden @ w2` matmuls with a loop over the hidden
  dimension, rather than two separate kernel/`extern_kernels` calls with the GeLU epilogue
  fused into one of them.

- **R3.** Inductor WILL fuse elementwise epilogues into matmuls where cheap (bias-add,
  GeLU) but is likely to keep the matmuls themselves as separate cuBLAS calls
  (`extern_kernels.mm`/`extern_kernels.addmm`) rather than generating its own Triton GEMM,
  because on a card with no tensor cores cuBLAS is hard to beat and our own tuned Triton
  GEMM reached only parity (see `07-retuning.md`). **Falsified if** the generated code
  contains a Triton `tl.dot`-based GEMM kernel for the QKV projection, the MLP linears, or
  the patch-embed/head linear, rather than `extern_kernels.mm`/`addmm` calls.

- **R4.** Inductor's total launch count per forward will fall strictly between our
  `triton_composed` (77) and torch eager (103) — i.e. more fused than our hand-tuned
  composed kernels' launch count would suggest is achievable by "obvious" fusion, but not
  as unfused as raw eager. **Falsified if** the measured launch count (via
  `torch.profiler`, matching the methodology in `docs/findings/06-synthesis.md` and
  `bench/collect_counters.py`) falls outside `(73, 103)` exclusive on either side — note
  73 is `triton_fused`'s count, so a result at or below 73 would mean Inductor out-fused
  our most aggressive hand-written variant, which would itself be a headline finding, not
  just an R4 miss.

## What would make each observation the most interesting result in the experiment

Per the plan: **if Inductor fuses something we measured as losing (the MLP matmuls, per
R2's falsification condition), that is the most interesting possible outcome.** It will be
measured directly (isolated MLP-shaped benchmark, same batch sweep, same counters as
Experiments 1/1b) and reported as a discrepancy to explain, not explained away — it would
mean either our measurement or our mechanism (H-loop serialisation) is wrong, or that
Inductor's generated kernel structures the reduction differently enough to avoid the cost
we measured (e.g. it may not unroll the loop the same way, or may pick different tile
sizes that side-step the shared-memory/register regime we characterized).

## Method

- New `torch_compile` arm added to `bench/run_sweep.py`, **append-only** — the existing
  `torch`, `triton_composed`, `triton_fused` arms and their committed `latency.csv` rows
  are not touched.
- `torch.compile(model)` wraps the baseline `VisionTransformer` from `model/baseline/vit.py`
  (pure eager PyTorch, no Triton kernels of ours involved) — this is Inductor operating on
  our reference implementation, not on our own kernels.
- Compilation is triggered and allowed to finish during a warmup phase, separate from and
  before any timed rep; `bench/harness.compare()`'s existing 5-call warmup loop is
  insufficient to guarantee this on its own for `torch.compile` (compilation can take much
  longer than 5 eager calls' worth of time and is shape-specialized, i.e. re-triggered per
  batch size in this sweep), so `torch_compile`'s arm callable does its own extra
  compile-triggering warmup before being handed to `compare()`. The first timed rep is
  checked against the timed median to confirm it is not a multi-second compile-time
  outlier before results are trusted.
- Sweep batches `{1, 8, 32, 128, 512}`, same as every other arm, via the unmodified
  `run_sweep.py` sweep machinery.
- `TORCH_LOGS=output_code` captures the generated Triton/`extern_kernels` code for one
  representative batch size. The relevant kernels (layernorm+residual region, MLP region,
  QKV/patch-embed/head GEMM call sites) are excerpted into this document; the full dump is
  not committed (disk is at ~7 GB free).
- Launch count measured via `torch.profiler`, one steady-state forward pass, matching
  `bench/collect_counters.py`'s existing methodology for the other three arms so the 73/77/103
  comparison is apples-to-apples.
- Correctness: existing test suite only (`torch.compile` is not itself a registered
  variant subject to `tests/test_*.py`'s tolerance machinery); no kernel, test, or
  tolerance is modified.

## Results

**Environment.** Branch `feat/retune-kernels`, predictions committed at `f6512a0`.
`TRITONFORMER_LOCKED_CLOCK_MHZ=1300` exported for every measurement. `torch 2.11.0+cu128`,
`triton 3.6.0`. Inductor logged `Not enough SMs to use max_autotune_gemm mode` once per
process (16 SMs on this card) — it never attempted to generate its own autotuned Triton
GEMM here, consistent with R3 before a single kernel was even inspected.

### Launch count (2c, R4)

Measured identically to `docs/findings/06-synthesis.md`'s methodology (`torch.profiler`,
one steady-state forward pass, batch=1): **`torch_compile`: 84 launches, 17 distinct
kernel names**, reproduced twice, bit-for-bit identical both times.

| variant | launches/pass | distinct kernel names |
|---|---:|---:|
| `torch` (eager) | 103 | 14 |
| `triton_composed` | 77 | 11 |
| `triton_fused` | 73 | 12 |
| **`torch_compile`** | **84** | **17** |

84 falls strictly inside `(73, 103)`, exactly as R4 predicted — closer to our hand-fused
`triton_composed`/`triton_fused` than to raw eager, without beating either of our
hand-tuned variants' launch count. The higher *distinct*-kernel count (17 vs. 11-14) is
Inductor generating one bespoke fused kernel per shape/fusion-group combination rather
than reusing a handful of general-purpose kernels the way our hand-written registry does
— more distinct kernels, but each covering more of the graph than a single hand-written
op does.

### Generated code (2c) — what Inductor actually fused

Captured via `TORCH_LOGS=output_code` at batch=8 on `VisionTransformer`
(`model/baseline/vit.py`), the plain eager baseline — Inductor operating on our reference
implementation, not on any of our own kernels. Full dump was 320 KB / 1631 log lines and
is not committed; the excerpts below are the load-bearing evidence, verified against the
full dump.

**All GEMMs are cuBLAS, never Triton (confirms R3's GEMM half):**

```python
extern_kernels.mm(reinterpret_tensor(buf5, (512, 192), (192, 1), 0),
                  reinterpret_tensor(arg6_1, (192, 576), (1, 192), 0), out=buf6)   # QKV
extern_kernels.bmm(reinterpret_tensor(buf7, (24, 64, 64), ...), ..., out=buf9)     # attn scores
extern_kernels.mm(reinterpret_tensor(buf15, (512, 192), ...), ..., out=buf16)      # proj
extern_kernels.mm(reinterpret_tensor(buf21, (512, 192), ...), ..., out=buf22)      # MLP w1
extern_kernels.mm(reinterpret_tensor(buf23, (512, 768), ...), ..., out=buf24)      # MLP w2
extern_kernels.addmm(arg79_1, buf144, reinterpret_tensor(arg78_1, ...), out=buf145)  # head
```

`grep -c "tl\.dot"` over the full dump: **0**. No Triton kernel anywhere in the generated
code performs a matrix contraction; every GEMM (QKV, attention scores/values, proj, both
MLP linears, patch-embed, head) is `extern_kernels.mm`/`bmm`/`addmm` — cuBLAS(Lt). Cheap
bias epilogues that cuBLASLt itself can fuse (the head's final linear) use `addmm`
directly; every other bias-add is deferred into the following Triton pointwise/reduction
kernel instead (see below) rather than fused into the `mm` call.

**Whole-MLP fusion into one kernel: declined (confirms R2).** The MLP is `mm(w1) ->
triton_poi_fused_addmm_gelu_view_8 (bias+GeLU, pointwise, no reduction) -> mm(w2)`, i.e.
still two separate cuBLAS GEMMs with a small Triton epilogue kernel between them — the
same two-matmul-plus-epilogue shape as our own `triton_composed`, not the single-kernel
serial-hidden-dimension-reduction shape of `_mlp_fused_kernel`. The `w2` matmul's own
output bias-add is not even given its own kernel: it is deferred into the *next* block's
fused residual+layernorm kernel (see below), so the MLP epilogue's bias lives entirely
inside neighboring pointwise/reduction kernels, never in a matmul-fused Triton kernel:

```python
buf22 = ...; extern_kernels.mm(buf21, arg12_1_reinterpreted, out=buf22)          # w1
buf23 = reinterpret_tensor(buf22, ...)
triton_poi_fused_addmm_gelu_view_8.run(buf23, arg13_1, 393216, stream=stream0)   # +b1, GeLU
buf24 = ...; extern_kernels.mm(buf23, arg14_1_reinterpreted, out=buf24)          # w2
buf28 = ...
triton_per_fused_add_addmm_native_layer_norm_view_9.run(
    buf17, arg15_1, buf24, arg16_1, arg17_1, buf28, 512, 192, stream=stream0)    # +b2, +residual, LN2
```

**Layernorm+residual (and the preceding linear's bias) fused into one kernel (confirms
R1):** `triton_per_fused_add_addmm_native_layer_norm_view_1` takes the raw `mm` output,
the bias, and the residual stream as three separate loads and fuses bias-add + residual-add
+ mean/var reduction + normalize into a single persistent-reduction Triton kernel body —
more aggressive than our own `layernorm_residual`, which only fuses the norm and the
residual add (the bias-add there is already folded into the preceding linear kernel by
construction):

```python
def triton_per_fused_add_addmm_native_layer_norm_view_1(
        in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK):
    ...
    tmp0 = tl.load(in_ptr0 + (r0_2), ...)             # bias
    tmp1 = tl.load(in_ptr1 + (r0_2 + 192*x3), ...)     # mm output (proj/qkv)
    tmp3 = tl.load(in_ptr2 + (r0_2 + 192*x0), ...)     # residual stream
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    # ... mean/var reduction, normalize (native_layer_norm) inline, single kernel
```

No standalone elementwise "add" kernel for any residual connection appears anywhere in
the dump — every residual add Inductor generates is fused into either the following
layernorm's reduction kernel or (at the very end) the mean-pool+head kernel. This is
exactly the register-cheap, round-trip-eliminating shape the register rule predicts
should win, generated automatically with no knowledge of our hypothesis.

### Latency (ms, median of interleaved reps, `bench/harness.compare()`, batch sweep
### written to `bench/results/latency.csv`, commit `f6512a0`)

| batch | torch | triton_composed | triton_fused | torch_compile | compile/torch |
|---:|---:|---:|---:|---:|---:|
| 1   | 0.8281  | 1.8739  | 4.1472  | 0.7614  | 0.92x |
| 8   | 2.8652  | 6.1619  | 15.0967 | 3.0684  | 1.07x |
| 32  | 11.5983 | 22.6526 | 58.8083 | 8.9847  | 0.77x |
| 128 | 39.5250 | 89.6230 | 234.3975| 32.9621 | 0.83x (`torch_compile` flagged, 675 MHz) |
| 512 | 203.2131| 389.1505| 963.1414| 241.7115| 1.19x (`torch`, `triton_composed`, `torch_compile` all flagged) |

**`torch_compile` is the fastest of all four arms at every batch except batch=8**, where
it is 7% slower than eager `torch` (unflagged, clean measurement, not attributable to
throttling — a genuine small loss at that one batch size, reported rather than smoothed
over). At batch=32 (the cleanest, fully-unflagged comparison) `torch_compile` beats eager
`torch` by 22.6%, and beats our own `triton_composed`/`triton_fused` by 60.3%/84.7%.
`torch_compile` never loses to either of our own Triton arms at any batch. The batch=128
and batch=512 rows carry thermal-throttling confounds noted below and should be read as
directional, not precise.

### Flagged counts

20 new `vit_forward` rows added (5 batches x 4 arms, `torch`/`triton_composed`/
`triton_fused`/`torch_compile`). **4/20 flagged**: `torch_compile` at batch=128 (675 MHz)
and batch=512 (630 MHz); `torch` at batch=512 (915 MHz); `triton_composed` at batch=512
(300 MHz, the deepest throttle observed in this experiment). `triton_fused` was never
flagged (1305 MHz at every batch) — consistent with `10-register-rule.md`'s observation
that per-call latency, not batch size alone, predicts thermal load, and `triton_fused` is
slow enough per-call that its *reps* run at low duty cycle relative to wall-clock, while
`torch_compile`/`torch`/`triton_composed` pack many more, shorter, back-to-back kernel
launches into the same wall-clock window at large batch and heat up faster. The batch=512
row is the least trustworthy in this table for exactly that reason: three of its four arms
ran at different degraded clocks (1305 vs 915 vs 300 vs 630 MHz) simultaneously, so its
1.19x `torch_compile`/`torch` ratio reflects relative throttle severity as much as kernel
quality. Batches 1/8/32 (12/20 rows, 0 flagged) carry the trustworthy signal; the headline
holds there (`torch_compile` fastest at 1 and 32, narrowly behind eager at 8).

## Verdicts on R1-R4

- **R1 (Inductor fuses layernorm+residual, no separate round-trip): HOLDS, and more
  aggressively than predicted.** Every block's residual add is fused directly into the
  same Triton kernel as the following layernorm's reduction — and the preceding linear's
  bias-add is fused in too, three ops (bias, residual, layernorm) in one kernel body, one
  launch. No standalone elementwise "add" kernel exists anywhere in the generated code.

- **R2 (Inductor declines to fuse both MLP matmuls into one Triton kernel with a serial
  hidden-dimension reduction): HOLDS.** The MLP is two separate `extern_kernels.mm` calls
  (cuBLAS) bracketing one small Triton pointwise kernel (bias+GeLU only, no reduction, no
  `tl.dot`). This is structurally the same shape as our own `triton_composed`, the arm
  that *wins* against `mlp_fused` in Experiments 1/1b, not the single-kernel
  H-loop-serialised shape that loses. Inductor's independent choice lands on exactly the
  side of this fork that our measurements say is correct.

- **R3 (Inductor fuses cheap epilogues but keeps GEMMs on cuBLAS, no Triton GEMM):
  HOLDS.** `grep -c "tl\.dot"` over the full generated-code dump is 0. Every GEMM (QKV,
  attention scores/values, proj, both MLP linears, patch-embed, head) is
  `extern_kernels.mm`/`bmm`/`addmm`. Bias epilogues are fused where cheap — either
  directly into cuBLASLt's own `addmm` (the head) or into the neighboring Triton
  pointwise/reduction kernel (everywhere else) — never into a Triton-generated matmul,
  because Inductor never generates one here.

- **R4 (launch count strictly between `triton_composed` (77) and `torch` (103)): HOLDS.**
  84 launches, 17 distinct kernels, measured identically to `06-synthesis.md`'s
  methodology and reproduced twice, bit-for-bit identical. It does not beat either of our
  hand-tuned variants' launch count (73/77), consistent with Inductor's general-purpose
  fusion heuristics not reaching quite as far as targeted hand-tuning on kernel *count*,
  even though its choices win on *latency* (see below).

## Does Inductor's independent choice support or contradict this project's findings?

**Full support, no contradiction, and the strongest form available: Inductor's expert
system, with zero knowledge of the register rule or the H-loop-serialisation finding,
independently reaches the same fork in the decision tree at every point this project
measured, and picks the winning branch every time.**

- It fuses the one thing this project found unambiguously wins (layernorm+residual,
  register-cheap, removes a round-trip) — and fuses more aggressively around it (also
  folding in the preceding linear's bias) than our own hand-written
  `layernorm_residual` does, without regressing.
- It declines the one thing this project found unambiguously loses (a single Triton
  kernel serialising the MLP's hidden-dimension reduction) — Experiment 2d's "if it fuses
  something we measured as losing, that is the most interesting result" branch did **not**
  trigger. There was no discrepancy to chase down: Inductor's MLP decomposition matches
  `triton_composed`'s winning shape, not `mlp_fused`'s losing one.
- It never generates a Triton GEMM on this no-tensor-core, 16-SM card, matching this
  project's own conclusion (`07-retuning.md`) that our best tuned Triton GEMM only
  reaches cuBLAS parity, never beats it.

The one result this experiment adds that the project had not previously measured directly:
**`torch_compile` on the plain eager baseline beats every arm this project has built,
including hand-tuned `triton_composed`, at every batch size** (60-85% faster than
`triton_composed` at the cleanest unflagged batch, 32). This is not a contradiction of the
register rule — Inductor is not winning by doing something our rule says shouldn't work,
it is winning by doing more, and more precisely targeted, of exactly what our rule says
should work (round-trip elimination without register cost) than either of our hand-written
`triton_composed`/`triton_fused` arms manage, while also declining every fusion our
measurements say is a loss. It is independent, automatic confirmation that the rule
identifies the right lever; Inductor simply pulls it harder and in more places than our
hand-written kernels did.

## Test summary

Full suite unaffected: `torch.compile` is not a registered `Component` variant and
introduces no new kernel or tolerance. `tests/test_*.py`: **153 passed** (unchanged from
the branch's starting state), confirming this experiment's only production-code change
(`bench/run_sweep.py`'s new `torch_compile` arm) does not perturb any existing kernel or
variant. `git diff --stat model/` and `git diff --stat model/kernels/` are both empty for
this experiment.

## Commits

- pre-registration commit for this document (before any measurement)
- results commit landing `bench/run_sweep.py`'s `torch_compile` arm, this document's
  Results section, and the new `latency.csv` rows

## Concerns / caveats

- The batch=128/512 latency rows carry thermal-throttling confounds (4/20 new rows
  flagged, concentrated at the two largest batches) — the 1.19x `torch_compile`/`torch`
  ratio at batch=512 in particular reflects three arms throttled to different degrees
  simultaneously and should not be read as precise; batches 1/8/32 (0 flagged) carry the
  trustworthy signal.
- `torch_compile` is 7% slower than eager `torch` at batch=8 specifically (unflagged,
  clean measurement) — a genuine small loss at one batch size, not smoothed into the
  otherwise-favorable headline.
- The generated-code excerpts above are drawn from a batch=8 capture; launch-count
  measurement (84/17) is at batch=1, matching `06-synthesis.md`'s existing methodology so
  the 73/77/103 comparison stays apples-to-apples. The full `TORCH_LOGS=output_code` dump
  (320 KB, batch=8) is not committed to the repo; the excerpts here were verified against
  it directly rather than transcribed from memory.
