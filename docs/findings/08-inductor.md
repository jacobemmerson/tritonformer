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

(Filled in below after measurement — this section did not exist at commit time of the
predictions above.)
