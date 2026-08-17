"""Latency benchmark for the fully fused transformer block (Task 17, rung 13
-- the deliberate far end of the fusion ladder, predicted to hurt more than
Task 16's mega-MLP).

Three arms: `torch` (`model.baseline.layers.block`, the plain reference,
six separate F.* calls), `triton_composed` (`model/kernels/block.py`'s
`block_composed`: `layernorm`, `qkv_project` (one [D -> 3D] GEMM),
`attention_flash`, `linear`, `layernorm_residual`, `mlp_composed` -- six
kernel launches, the best individual Triton variant at every step), and
`triton_fused` (`block_fused`: identical except the last step is
`mlp_fused`, Task 16's over-fused single-kernel MLP -- five launches).

A `triton_fused_monolithic` variant that holds a [BLOCK_M, 768] MLP hidden
tile and a [64, 64] attention tile in one kernel was attempted separately
(not as a registered arm here) and failed to compile: combining even a
single head's Q/K/V tiles with the MLP's weight tiles required 262,144
bytes of shared memory against the 65,536-byte/SM budget, a 4x overflow
independent of the MLP's own BLOCK_H tiling. See
docs/findings/05-over-fusion.md for the exact error text and the
configurations tried. `triton_composed`/`triton_fused` above are therefore
the only two Triton arms on this rung.

Shape matches the model's per-block computation: (batch, 64, 192) in,
(batch, 64, 192) out, heads=3, mlp_hidden=768.

See bench/runner.py for the sweep / single-shot profile contract shared
by every bench/run_*.py module.
"""
import torch

from bench.runner import RunnerSpec, main
from model.baseline.layers import block as block_torch
from model.config import ViTConfig
from model.kernels.block import block_composed, block_fused

CFG = ViTConfig()
SEQ, DIM, HIDDEN, HEADS = 64, CFG.dim, CFG.mlp_hidden, CFG.heads
SCALE, EPS = CFG.scale, CFG.eps


def _params(dtype: torch.dtype):
    scaled = lambda *shape: torch.randn(*shape, device="cuda", dtype=dtype) * 0.05
    return dict(
        ln1_w=torch.ones(DIM, device="cuda", dtype=dtype),
        ln1_b=torch.zeros(DIM, device="cuda", dtype=dtype),
        qkv_w=scaled(3 * DIM, DIM), qkv_b=torch.zeros(3 * DIM, device="cuda", dtype=dtype),
        proj_w=scaled(DIM, DIM), proj_b=torch.zeros(DIM, device="cuda", dtype=dtype),
        ln2_w=torch.ones(DIM, device="cuda", dtype=dtype),
        ln2_b=torch.zeros(DIM, device="cuda", dtype=dtype),
        w1=scaled(HIDDEN, DIM), b1=torch.zeros(HIDDEN, device="cuda", dtype=dtype),
        w2=scaled(DIM, HIDDEN), b2=torch.zeros(DIM, device="cuda", dtype=dtype),
        heads=HEADS, scale=SCALE, eps=EPS)


def _arms_for_batch(batch: int, dtype: torch.dtype):
    x = torch.randn(batch, SEQ, DIM, device="cuda", dtype=dtype)
    p = _params(dtype)
    return {
        "torch": lambda: block_torch(x, **p),
        "triton_composed": lambda: block_composed(x, **p),
        "triton_fused": lambda: block_fused(x, **p),
    }


def _bytes_theoretical(batch: int) -> int:
    # One theoretical minimum shared by all three arms: x read, all
    # parameter tensors read once, output written. Does not account for
    # any intermediate round trip (attended, normed, hidden activation,
    # etc.) -- those round trips, and the composed/fused launch-count
    # difference in how many of them hit DRAM, are exactly what this
    # rung measures; baking them into the formula would hide the effect
    # instead of exposing it (see bench/run_mlp.py for the same choice).
    x = batch * SEQ * DIM
    qkv_w, qkv_b = 3 * DIM * DIM, 3 * DIM
    proj_w, proj_b = DIM * DIM, DIM
    w1, b1 = HIDDEN * DIM, HIDDEN
    w2, b2 = DIM * HIDDEN, DIM
    ln = 4 * DIM  # ln1_w, ln1_b, ln2_w, ln2_b
    out = batch * SEQ * DIM
    params = qkv_w + qkv_b + proj_w + proj_b + w1 + b1 + w2 + b2 + ln
    return (x + params + out) * 4


SPEC = RunnerSpec(kernel="block", arms_for_batch=_arms_for_batch,
                  bytes_theoretical=_bytes_theoretical)


if __name__ == "__main__":
    main(SPEC)
