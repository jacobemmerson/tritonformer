# Retune, Re-baseline, and Replicate — Follow-up Plan

> **For agentic workers:** execute with superpowers:subagent-driven-development, one task at a time,
> review after each.

**Context.** The original 19-task plan concluded that Triton kernels are 1.75–5.03× slower than
PyTorch end-to-end, and that "fusion stops paying once shared memory forces tiles too small to
amortize the fusion, not once registers run out." Two follow-up measurements taken after that plan
merged show a substantial part of the *per-kernel* deficit is a configuration artifact, not a Triton
limitation:

- **Softmax:** the committed `num_warps=4` at `BLOCK=64` gives 0.5 elements/thread. At
  `num_warps=1` (2.0 elem/thread) it goes 2.74× slower → 1.15× slower; with 8-row-per-program
  batching (4.0 elem/thread) it reaches **parity and beats torch below batch 32**.
- **Linear:** autotuning + L2 group-ordering recovers **1.67–2.36×**, taking the gap from
  2.05–2.76× slower to 1.02–1.65×, reaching **parity at K=192→N=192**. Winning configs use
  `BLOCK_M` 128–256; the committed 64 was far too small for M = 32,768.

Both were measured on a cool, unthrottled card. The committed sweep ran with `SW Power Cap` and
`SW Thermal Slowdown` both active and the SM clock swinging 300–1575 MHz, so its absolute numbers
are lower bounds.

**Goal.** Establish how much of the Triton deficit is configuration versus intrinsic; add a baseline
that isolates "did we write good kernels?" from "is Triton competitive with cuBLAS?"; and test which
conclusions survive on hardware with a different shared-memory, L2, and tensor-core profile.

## Global Constraints

- **Do NOT modify or delete the existing kernels.** Every published finding cites them as the rung
  baselines. Tuned kernels are registered as NEW variants alongside; the naive-vs-tuned delta is
  itself the finding.
- Latency → `bench/results/latency.csv`; counters → `bench/results/counters.csv`. Separate, never mixed.
- Tolerances in `tests/conftest.py` are never loosened. A tuned kernel that needs more slack is a bug.
- Canonical dtype `"float32"`. Forward pass only.
- New findings go in NEW documents. Existing `docs/findings/00-06` get a pointer line, not a rewrite —
  they are the record of what was measured under the original configuration.
- Disk is at ~1.9 GB free. Never run `nsys`. Delete large temporaries immediately.

---

### Task A: Retune the kernels and re-measure under controlled clocks

**Files:** modify `model/kernels/{softmax,linear,mlp,attention}.py` (append only), `tests/` (new tests
for tuned variants); create `docs/findings/07-retuning.md`.

**PREREQUISITE (human): DONE.** Clocks are locked at 1830 MHz via
`sudo nvidia-smi -pm 1; sudo nvidia-smi -lgc 1830,1830`. 1830 is the LOWEST SUPPORTED graphics clock
on this card (range 1830–2100); the original plan's 1200 is below range and `nvidia-smi` rejects it.
Verified: `clocks.sm` and `clocks.gr` both pinned at 1830, and all throttle reasons
(`SW Power Cap`, `SW Thermal Slowdown`, `HW Thermal Slowdown`) now read `Not Active`.
Reset afterwards with `sudo nvidia-smi -rgc`.

- [ ] **A0.** Fix `bench/clocks.py::locked_clock_mhz()` — it is why `flagged` was dead on all 240
      rows of the original sweep. It queries `clocks.applications.graphics`, which is `[N/A]` on
      GeForce: application clocks (`-ac`) are a different mechanism this card does not support, and
      **there is no queryable read-back of an `-lgc` lock** (confirmed: no lock field in
      `nvidia-smi -q`, and pynvml is unavailable). So auto-detection is impossible here.
      Make it EXPLICIT instead: read the locked target from an env var (e.g. `TRITONFORMER_LOCKED_CLOCK_MHZ`)
      or a CLI flag, defaulting to `None` when unset. `flagged` then compares each sampled
      `clocks.sm` against that declared target using the existing 5% tolerance. Do NOT infer the
      lock from clock stability — that is clever and fragile. Add a test covering: unset (returns
      `None`, `flagged` False), set-and-matching (False), set-and-drifting (True).

- [ ] **A1.** Add `softmax` tuned variant `(SOFTMAX, "triton_tuned")`: multiple rows per program
      (`ROWS` a `tl.constexpr`, 8 measured best), 2D masking over rows and columns. Keep the naive
      `"triton"` variant untouched. Reuse `TOLERANCES["softmax"]`.
- [ ] **A2.** Add `linear` tuned variant `(LINEAR, "triton_tuned")` using `triton.autotune` over a
      config grid (`BLOCK_M` 64/128/256, `BLOCK_N` 64/128, `BLOCK_K` 32/64, `num_warps` 4/8,
      `num_stages` 2/3/4) keyed on `(M, N, K)`, PLUS L2 group-ordering (`GROUP_M`, the standard
      Triton matmul pid swizzle). Keep `_linear_kernel` untouched. Same for `linear_gelu`.
- [ ] **A3.** Apply the same autotune treatment to `mlp_fused` and `_flash_kernel` as new variants.
      For `mlp_fused` specifically, let the grid explore larger `BLOCK_H` and `num_stages > 1` — its
      measured failure mode was `BLOCK_H=32` forcing 24 sequential iterations with no double
      buffering, so this directly tests whether that was the binding cost.
- [ ] **A4.** Correctness: every tuned variant must pass its component's existing test file
      unchanged, at unchanged tolerances. Add the tuned variants to the existing parametrized
      `VARIANTS` lists where they exist.
- [ ] **A5.** Re-run the full sweep with naive AND tuned arms. Report the naive-vs-tuned delta per
      kernel and the tuned-vs-torch ratio. Collect counters for the tuned variants (registers,
      occupancy, spill, DRAM) and compare against the naive ones.
- [ ] **A6.** `docs/findings/07-retuning.md`: how much of each deficit was configuration; which
      kernels reach parity; whether the fusion conclusion changes. **State plainly if it does not** —
      autotuning `mlp_fused` may sharpen the over-fusion finding rather than overturn it.
- [ ] **A7.** Add a pointer line to `docs/findings/06-synthesis.md` noting the retuned results exist
      and where. Do not rewrite its numbers.

### Task B: `torch.compile` baseline

**Files:** modify `bench/run_sweep.py` (add arm); create `docs/findings/08-inductor.md`.

Inductor generates Triton. Comparing our hand-written Triton against Inductor's generated Triton
separates "did we write good kernels?" from "is Triton competitive with cuBLAS?" — two questions the
current setup conflates.

- [ ] **B1.** Add a `torch_compile` arm to the sweep: `torch.compile(model)` on the baseline
      `VisionTransformer`, with compile time excluded from measurement (warm up until compiled).
- [ ] **B2.** Sweep it across `{1, 8, 32, 128, 512}` alongside `torch`, `triton_composed`,
      `triton_fused`, and the Task A tuned arms.
- [ ] **B3.** Capture which kernels Inductor generates (`TORCH_LOGS=output_code` or
      `torch._inductor.config.debug`) and how many launches per forward, versus our 103/77/73.
      Note specifically whether Inductor fuses where we fused, and where it declines to.
- [ ] **B4.** `docs/findings/08-inductor.md`: our Triton vs Inductor's Triton vs eager. The most
      interesting result is anywhere Inductor **chose not to fuse** something we did — that is an
      independent expert opinion on our fusion decisions.

### Task C: L4 replication on Modal, with pre-registered predictions

**Files:** create `modal_app.py`, `docs/findings/09-l4-replication.md`.

**PREREQUISITE (human):** `pip install modal && modal setup` (interactive browser auth).

**Write the predictions BEFORE measuring.** Record them in the findings doc first, then report which
broke. Hardware deltas: L4 is sm_89, 58 SMs (vs 16), ~24 GB (vs 4), ~300 GB/s (vs ~192),
**~48 MB L2 (vs ~1 MB)**, up to ~99 KB shared/block (vs 64 KB), **has tensor cores**, same 65,536
registers/SM, datacenter cooling (no throttle).

- [ ] **C1.** Pre-register these four predictions verbatim in the findings doc:
      1. **Traffic savings shrink or vanish.** A `[128,3,64,64]` fp32 score matrix is 12.6 MB and
         fits in 48 MB L2, so the unfused arm may never reach DRAM. Flash's −37.2% and
         `layernorm_residual`'s −20% should shrink. Testable: `dram__bytes_read.sum` for composed arms.
      2. **The matmul comparison becomes unfair unless controlled.** cuBLAS gets 4th-gen tensor
         cores; our fp32 `tl.dot` does not. Control explicitly — report both strict-fp32 and
         TF32-enabled numbers, or the comparison measures precision policy, not kernel quality.
      3. **Register-limited occupancy transfers unchanged.** Same register file, so `mlp_fused` at
         226 regs × 256 threads → 1 block/SM → 25% should reproduce.
      4. **The monolithic block kernel still fails to compile.** It needs 262,144 B; Ada allows
         ~99 KB/block. The bound moves, it does not disappear.
- [ ] **C2.** Modal app: L4 container, pinned to the same torch/triton versions, running the existing
      test suite first (correctness must hold on new hardware before any measurement counts).
- [ ] **C3.** Run the full sweep and counter grid on L4. Note `F.scaled_dot_product_attention` will
      select a different (real FlashAttention) backend on sm_89 — record which, as sm_75's cutlass
      fallback was already 3.26× faster than our flash kernel.
- [ ] **C4.** `docs/findings/09-l4-replication.md`: which predictions held, which broke, and what the
      headline conclusion becomes on hardware where L2 may make fusion pointless rather than shared
      memory making it impossible. A broken prediction is the most valuable outcome here.

---

## Verification

After each task: `.venv/bin/python -m pytest -q` (currently 110 passed; tuned variants add tests).
Sweep rows append to `bench/results/latency.csv`; counters to `bench/results/counters.csv`.
No existing kernel modified — verify with `git diff --stat` that changes to `model/kernels/*` are
append-only.
