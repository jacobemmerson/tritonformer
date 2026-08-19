# Tritonformer

This project reimplements the **forward pass** of a shallow vision
transformer using Triton kernels, and measures when kernel fusion helps
and when it hurts, on a GTX 1650 Ti (sm_75, no tensor cores, 4GB). The
objectives:

a) Implement kernels:
	1. LayerNorm
	2. Linear
	3. GeLU
	4. Softmax
b) Fuse kernels along a ladder from single-op to a fully fused
   transformer block
c) Measure performance tradeoffs of fused operations, backed by latency
   and hardware-counter measurements

The project's answer to its own research question is
[`docs/findings/06-synthesis.md`](docs/findings/06-synthesis.md) — which
fusions helped, which hurt and why, and where the fusion break-even point
sits on this hardware. [`docs/findings/10-register-rule.md`](docs/findings/10-register-rule.md)
documents two follow-up experiments that stress-tested and revised that
synthesis; read both, in that order — `06` states the original picture,
`10` corrects it.

**Headline, as it stands after all experiments:** on this card, fusion
pays only when it removes a DRAM round-trip without adding register or
loop-serialization cost (`layernorm_residual`, +22-31%); every fusion
that instead traded traffic for register pressure or a serialized
reduction lost, up to 3.8x (`mlp_fused`) and 2.24x (`attention_flash`).
`torch.compile`, given only the plain eager model, beat every arm this
project hand-built at every batch except 8 — and it won by declining to
attempt a hand-written GEMM, a fused MLP, or flash attention, not by
out-fusing them (`docs/findings/08-inductor.md`).

**The register rule, and its status:**
*"Fusion pays when it removes a memory round-trip without spending
registers."* Originally fitted to five observations in `06-synthesis.md`.
Two follow-up experiments (`docs/findings/10-register-rule.md`) tried to
break it:

- **Refuted as a general predictor for `mlp_fused`.** Cutting registers
  226 -> 128 (which should restore occupancy to 50% by the project's own
  arithmetic) left measured occupancy pinned at 25.00% — registers were
  never what was binding. The rule's original "no round-trip, no
  registers, so it should win" reasoning does not transfer to a fusion
  whose reduction is a compile-time-unrolled loop with no pipelining;
  the real cost there is the loop's serial dependency depth, a mechanism
  the rule never accounted for.
- **Not wrong about `layernorm_residual`**, where register count and DRAM
  traffic are genuinely independent levers — the rule's core claim holds
  for the case it was built on.
- **Corroborated independently by `torch.compile`/Inductor**, which fuses
  layernorm+residual (register-cheap) and declines the MLP's serial-loop
  shape, with zero knowledge of this project's hypothesis
  (`docs/findings/08-inductor.md`).
- **The occupancy-arithmetic method used throughout this project remains
  valid** — it matched measurement exactly at 128, 168, 226, and 255
  regs/thread — only the conclusion that registers were *binding* for
  the mega-MLP specifically was withdrawn.
- **Replicated on an L4 (sm_89) via Modal**, predictions pre-registered
  before measuring (`docs/findings/09-l4-replication.md`). The headline
  survives the card change: `triton_fused` costs more than
  `triton_composed`, which costs more than eager `torch`, at every batch
  but one — over-fusion is not an artifact of a small, slow,
  tensor-core-less laptop GPU. The penalty shrinks substantially
  (5.93x -> 2.68x torch at batch 128) without inverting, and fusion's
  DRAM-read advantage persists on every rung (1.50x->1.55x, 2.47x->2.02x,
  3.54x->4.94x), so the prediction that a 48 MB L2 would make fusion
  pointless is broken — by not happening, not by reversing.
- **A precision caveat the project could not see on one card.** Triton's
  `tl.dot` defaults to TF32 on tensor-core hardware; sm_75 has none, so
  every measurement here is IEEE fp32 *by accident of hardware*, not by
  choice. Under that default on sm_89, 70 of 153 tests fail and no kernel
  declares a precision policy — arguably the most consequential thing
  Experiment 3 found, and none of its four predictions
  (`docs/findings/09-l4-replication.md`).

**Scope note:** this is forward-pass inference only; there is no training
loop in this codebase. A Triton training loop and optimizer was an
original stretch objective and was not built — it remains future work,
not a current objective of this project.
