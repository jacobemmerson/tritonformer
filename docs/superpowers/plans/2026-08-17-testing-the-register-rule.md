# Testing the Register Rule — Three Experiments

> Execute with superpowers:subagent-driven-development, one experiment at a time, review after each.

## Context

The 19-task study concluded fusion mostly does not pay on this GTX 1650 Ti. The retuning follow-up
(`docs/findings/07-retuning.md`) showed much of the per-kernel deficit was configuration, and
surfaced a sharper claim. Measured across every fusion in the project:

| fusion | regs before -> after | spill | result |
|---|---|---|---|
| layernorm -> +residual | 28 -> **27** | 0 | **WINS 22-25%** at every batch |
| linear -> +gelu (naive) | 128 -> 168 | 0 | wins 7-14% |
| linear -> +gelu (tuned) | 255 -> 255 | **131 MB** | loses 9-13% |
| composed -> flash attention | -> **255 (cap)** | 473 KB | loses 1.49-2.24x |
| composed -> mega-MLP | 128/168 -> 226 | 0 | loses 3.10-3.83x |

**THE RULE: fusion pays when it removes a memory round-trip without spending registers.**

`layernorm_residual` uses one FEWER register than the kernel it fuses into, deletes a full DRAM
round-trip, and wins everywhere. Every fusion that bought traffic with register pressure lost,
because on a 16-SM card the occupancy loss exceeds the bandwidth saving.

That rule is currently a pattern fitted to five observations. These three experiments try to break it.

## Global Constraints

- **Never modify an existing kernel.** New variants register alongside; the delta is the finding.
- Tolerances in `tests/conftest.py` are never loosened.
- Latency -> `latency.csv`; counters -> `counters.csv`. Separate, never mixed.
- `sm_clock_mhz`/`temp_c` now mean min-observed/max-observed during measurement. `flagged` is live:
  export `TRITONFORMER_LOCKED_CLOCK_MHZ` to the user's lock. **Report flagged counts; heavy kernels
  drift on ~60% of rows on this card and that must be visible, not hidden.**
- **Pre-register predictions in the findings doc BEFORE measuring.** A broken prediction is the most
  valuable outcome available; a retrofitted narrative is worthless.
- Never run `nsys` (broken importer, multi-GB dumps). Never use sudo. Never run `nvidia-smi -rgc`.

---

## Experiment 1: The Flip — can a losing fusion be made to win?

**The test.** `mlp_fused` cuts DRAM traffic 73% and still loses 3.10-3.83x, at 226 regs x 256 threads
= 57,856 regs/block -> **1 block/SM -> 25% occupancy**. The rule predicts that if register pressure
drops enough to restore occupancy, it should flip to winning, because the traffic saving is already
banked.

Concrete target: **<=128 regs/thread at num_warps=4** gives 128 x 128 = 16,384 regs/block ->
**4 blocks/SM -> 50% occupancy**, matching `_linear_kernel`.

**Where the registers are.** The `[BLOCK_M, BLOCK_D]` accumulator dominates: 16 x 256 = 4,096 floats
over 256 threads = 16 regs/thread for the accumulator alone, before the hidden tile.

- [ ] **1a. Pre-register predictions** in `docs/findings/10-register-rule.md`:
      - P1: reducing `BLOCK_M` 16 -> 8 -> 4 monotonically reduces regs/thread.
      - P2: at <=128 regs the fused MLP reaches >=50% occupancy.
      - P3: **at >=50% occupancy the fused MLP beats composed**, because its 73% DRAM saving is
        unchanged while the occupancy penalty is gone.
      - P4: there is a register threshold between 128 and 226 where it crosses from loss to win.
- [ ] **1b.** Add `mlp_fused_lowreg` as a NEW variant sweeping `BLOCK_M` in {8, 4, 2} and
      `num_warps` in {4, 8}. Do not modify `_mlp_fused_kernel`.
- [ ] **1c.** Measure regs/occupancy/spill/DRAM per configuration via `bench/collect_counters.py`.
      Plot regs against fused/composed latency ratio — that curve is the experiment's product.
- [ ] **1d.** Also try staging the hidden tile through **shared memory** rather than registers, as a
      second route to the same target. Shared memory is the resource the project's headline blames;
      trading registers for it tests both claims at once.
- [ ] **1e.** State the outcome plainly. **If it never flips, the rule is WRONG or incomplete — say
      so and propose what else explains it** (candidates: the H-loop's sequential iterations, absent
      `num_stages` double-buffering, or arithmetic intensity too low to amortise regardless).

## Experiment 2: Inductor's opinion — does an expert system agree?

**The test.** `torch.compile` generates Triton and makes fusion decisions automatically. If Inductor
fuses where our rule says fusion pays and declines where it says it doesn't, that is independent
corroboration from a system with no knowledge of our hypothesis.

- [ ] **2a. Pre-register:** Inductor SHOULD fuse layernorm+residual (register-cheap, removes a
      round-trip) and SHOULD NOT fuse the whole MLP into one kernel (register-expensive).
- [ ] **2b.** Add a `torch_compile` arm to `bench/run_sweep.py`, excluding compile time from
      measurement (warm up until compiled).
- [ ] **2c.** Capture the generated kernels (`TORCH_LOGS=output_code`) and count launches per forward
      versus our measured 103 (torch eager) / 77 (triton_composed) / 73 (triton_fused).
- [ ] **2d.** **The headline is anywhere Inductor's decision disagrees with ours.** Where it declines
      a fusion we performed, check whether its reasoning matches the register rule. Where it fuses
      something we did not, measure that fusion ourselves.
- [ ] **2e.** `docs/findings/08-inductor.md`: our Triton vs Inductor's Triton vs eager, and an
      explicit verdict on whether Inductor's choices support or contradict the rule.

## Experiment 3: L4 — is the rule card-specific?

**PREREQUISITE (human):** `modal setup` (interactive browser auth). `modal` is already installed.

**The test.** The rule was derived on a 16-SM, ~1 MB-L2, 50 W, no-tensor-core laptop card. The L4 is
sm_89: 58 SMs, ~48 MB L2, ~300 GB/s, tensor cores, datacenter cooling, **same 65,536 regs/SM**.

- [ ] **3a. Pre-register these four predictions verbatim before measuring:**
      1. **The rule's premise may collapse.** A `[128,3,64,64]` fp32 intermediate is 6.29 MB
         (`128 x 3 x 64 x 64 x 4 = 6,291,456 B`; corrected from 12.6 MB — the original figure appears
         to have costed a write+read round trip rather than the tensor's own size) and fits
         in 48 MB of L2, so the unfused arm may never reach DRAM. If so, **even register-free fusion
         stops paying** — which would make `layernorm_residual`'s 22-25% win an artifact of a
         small-L2 card. Testable: `dram__bytes_read.sum` for composed arms should collapse.
      2. **Register arithmetic transfers unchanged.** Same 65,536 regs/SM, so `mlp_fused` at 226 regs
         x 256 threads -> 1 block/SM -> 25% should reproduce exactly.
      3. **The matmul comparison needs controlling.** cuBLAS gets tensor cores; our fp32 `tl.dot`
         does not. Report strict-fp32 AND TF32-enabled numbers, or the comparison measures precision
         policy rather than kernel quality.
      4. **The monolithic block kernel still fails to compile.** It needs 262,144 B; Ada allows
         ~99 KB/block. The bound moves, it does not disappear.
- [ ] **3b.** Modal app pinned to the same torch/triton versions. **Run the test suite first** —
      correctness must hold on new hardware before any measurement counts.
- [ ] **3c.** Full sweep plus counters. Record which SDPA backend sm_89 selects (real FlashAttention
      is available there; sm_75's cutlass fallback already beat our flash kernel 3.26x).
- [ ] **3d.** `docs/findings/09-l4-replication.md`: which predictions held, which broke, and what the
      rule becomes on hardware where L2 may make fusion pointless rather than registers making it
      costly.

---

## Debt to clear alongside

`docs/findings/03-epilogue-fusion.md` and `06-synthesis.md` still present `linear_gelu` as a fusion
that pays. `07-retuning.md` showed that reverses under tuning (verified on unflagged batches 8/128/512
with the effect *growing* 4.7% -> 8.6% -> 13.8%, opposite to noise). A reader hitting `06` first gets
the superseded claim. Add a correction pointer to both — do not rewrite their measured numbers.

## Verification

After each experiment: `.venv/bin/python -m pytest -q` (currently 143 passed).
Confirm `git diff --stat model/kernels/` shows append-only changes.

---

## Experiment 4: Consistency pass — make the record agree with itself

Run LAST, once Experiments 1-3 have landed, so it reconciles against final results rather than
moving targets. This is editorial work on the findings corpus: **do not re-measure anything, and do
not rewrite any measured number.** Where a claim is superseded, annotate it and point forward.

- [ ] **4a. Audit every cross-document claim.** `docs/findings/00` through `10` plus `README.md`.
      Build a table of every quantitative claim that appears in more than one document, and flag any
      that disagree. The merged study, the retuning follow-up, and these experiments were written at
      different times against different measurement regimes.
- [ ] **4b. The known superseded claim.** `03-epilogue-fusion.md` and `06-synthesis.md` present
      `linear_gelu` epilogue fusion as one of two fusions that paid (+9-11%). `07-retuning.md`
      showed this REVERSES under autotuning — verified on unflagged batches with the effect growing
      4.7% -> 8.6% -> 13.8%, opposite to what noise would do, and mechanistically explained
      (`_linear_gelu_tuned_kernel` hits the 255-register cap and spills 131 MB while
      `_linear_tuned_kernel` hits 255 with zero spill). Add a forward pointer to both documents.
      **Keep their original numbers** — they are the correct record of what was measured under that
      configuration.
- [ ] **4c. The measurement-regime problem.** Three regimes now exist in `latency.csv`:
      (i) the merged study, unlocked, 300-1575 MHz, both power and thermal caps active, `flagged`
      dead by construction; (ii) the retune, locked 1300 MHz, `flagged` live, ~60% of heavy rows
      drifted; (iii) whatever these experiments add. Rows carry identical column names across all
      three, and `sm_clock_mhz`/`temp_c` CHANGED MEANING between (i) and (ii) from point-sample to
      min/max-observed. Document this prominently — anyone comparing rows across regimes without
      knowing it will draw false conclusions. Consider whether a `regime` marker is worth adding
      going forward (note: adding a CSV column breaks `record()`'s header logic for existing rows,
      so weigh that carefully rather than doing it reflexively).
- [ ] **4d. Update `README.md`** to state the current headline, the register rule, and its status
      after Experiments 1-3 — validated, refuted, or qualified.
- [ ] **4e. Reconcile the headline conclusion.** `05-over-fusion.md` and `06-synthesis.md` currently
      say *"Shared memory bounds fusion first, at compile time; registers collapse occupancy second,
      but fusion is already dead before registers spill."* The register rule says fusion pays when
      it is register-free. These are compatible but not obviously so to a reader. If Experiment 1
      flips the MLP by cutting registers alone, the shared-memory framing needs revisiting — state
      the reconciled version in one sentence, supported by both sets of measurements.
