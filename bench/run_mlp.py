"""Latency benchmark for the whole-MLP mega-kernel (Task 16, rung 12).

Three arms: `torch` (F.linear/F.gelu/F.linear, cuBLAS + a fused GeLU
kernel), `triton_composed` (Task 12's `linear_gelu` then Task 10's
`linear` -- two kernel launches, the 768-wide hidden activation round-
trips through DRAM between them), and `triton_fused` (`model/kernels/mlp.py`,
one kernel launch, hidden activation never leaves the SM).

This is the deliberate over-fusion rung: the brief predicted the fused
kernel would win at batch 1 (fewer launches dominate) and lose at batch
512 (occupancy/spill dominates). See docs/findings/05-over-fusion.md for
the full analysis, including why `_mlp_fused_kernel` had to be restructured
away from the brief's reference design (a single BLOCK_M=8 program holding
the whole H=768 width) -- that design cannot compile at any BLOCK_M on
this card's 64KB shared memory; the w1/w2 tiles alone (BLOCK_D=256 x
BLOCK_H=1024 x 4 bytes = 1,048,576 bytes) are ~16x the budget regardless
of batch tiling. The registered kernel instead loops over H in
BLOCK_H=32-sized tiles (found empirically to be the largest H-tile that
compiles, at BLOCK_M=16), the same K-loop structure `_linear_kernel` uses.

Shape matches the model's MLP block: (batch, 64, 192) -> (batch, 64, 768)
-> (batch, 64, 192).

See bench/runner.py for the sweep / single-shot profile contract shared
by every bench/run_*.py module.
"""
import torch

from bench.runner import RunnerSpec, main
from model.baseline.layers import mlp as mlp_torch
from model.kernels.mlp import mlp_composed, mlp_fused

SEQ, DIM, HIDDEN = 64, 192, 768


def _arms_for_batch(batch: int, dtype: torch.dtype):
    x = torch.randn(batch, SEQ, DIM, device="cuda", dtype=dtype)
    w1 = torch.randn(HIDDEN, DIM, device="cuda", dtype=dtype) * 0.05
    b1 = torch.randn(HIDDEN, device="cuda", dtype=dtype)
    w2 = torch.randn(DIM, HIDDEN, device="cuda", dtype=dtype) * 0.05
    b2 = torch.randn(DIM, device="cuda", dtype=dtype)
    return {
        "torch": lambda: mlp_torch(x, w1, b1, w2, b2),
        "triton_composed": lambda: mlp_composed(x, w1, b1, w2, b2),
        "triton_fused": lambda: mlp_fused(x, w1, b1, w2, b2),
    }


def _bytes_theoretical(batch: int) -> int:
    # read x, w1, b1, w2, b2; write the [batch, 64, 192] output. The
    # composed arm additionally round-trips the [batch, 64, 768] hidden
    # activation through DRAM (write + read back) between its two kernel
    # launches -- that round trip, and its absence in the fused arm, is
    # the entire point of this rung, so it is deliberately NOT included
    # here: this formula is the fused arm's true theoretical minimum, and
    # the composed arm's measured traffic should exceed it by roughly
    # that round trip (see docs/findings/05-over-fusion.md).
    x = batch * SEQ * DIM
    w1 = HIDDEN * DIM
    w2 = DIM * HIDDEN
    out = batch * SEQ * DIM
    return (x + w1 + HIDDEN + w2 + DIM + out) * 4


SPEC = RunnerSpec(kernel="mlp", arms_for_batch=_arms_for_batch,
                  bytes_theoretical=_bytes_theoretical)


if __name__ == "__main__":
    main(SPEC)
