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
sits on this hardware.

**Scope note:** this is forward-pass inference only; there is no training
loop in this codebase. A Triton training loop and optimizer was an
original stretch objective and was not built — it remains future work,
not a current objective of this project.
