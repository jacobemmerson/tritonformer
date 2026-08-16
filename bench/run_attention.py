"""Latency benchmark for the composed (unfused) Triton attention rung.

`triton_composed` runs five kernel launches per call -- q@k.T (matmul),
the scale multiply, softmax, then scores@v (matmul) -- and materializes
the [B, H, 64, 64] score matrix in DRAM between the two matmuls. That
round trip is the whole point of this rung: it is the baseline that
rung 10's flash-style fusion (Task 15) must beat. bytes_theoretical
below includes that round trip explicitly (see the brief), unlike a
normal single-kernel rung's bytes_theoretical.

Task 10 (linear) and Task 9 (softmax) both lose to their torch/cuBLAS
counterparts on this GPU (no tensor cores, occupancy-starved softmax);
triton_composed inherits both deficits, so it is expected to lose to the
torch baseline here, sometimes badly. That is the correct, honest result
for an unfused baseline -- see bench/runner.py for the sweep / single-shot
profile contract shared by every bench/run_*.py module.

`triton_qkv_fused` (Task 14, rung 9) replaces three separate [D -> D]
projection GEMMs with one [D -> 3D] GEMM ahead of `triton_composed`'s
same four-kernel attention cycle. It calls Task 10's Triton `linear`,
which already loses to cuBLAS 2.26-3.42x on this card, so this arm can
be slower than three cuBLAS GEMMs even though it launches fewer kernels
and does the matmul at higher arithmetic intensity. `bytes_theoretical`
below is the q/k/v-shaped formula shared with the other two arms; it
does not account for the fused GEMM's own x/qkv_w/qkv_b traffic, so its
achieved_gbps for this arm is an approximation, not a tight bound --
consistent with Task 13's finding that this formula already undershoots
GEMM-containing arms.

`triton_qkv_unfused` is the honest baseline for the fusion claim: the
same qkv_w/qkv_b split into three [D, D] slices and projected with three
separate Triton `linear` calls (mirroring what an unfused QKV rung would
launch), feeding the identical `triton_composed` attention cycle. It is
NOT the same comparison as `triton_composed` above -- that arm receives
pre-split q/k/v tensors and never launches a projection kernel at all.
Comparing `triton_qkv_fused` against `triton_qkv_unfused` isolates
exactly the "one GEMM vs three GEMMs" launch-count and latency delta
this rung is about.
"""
import torch

from bench.runner import RunnerSpec, main
from model.baseline.layers import attention as attention_torch
from model.kernels.attention import attention_composed as attention_triton
from model.kernels.attention import attention_qkv_fused
from model.kernels.linear import linear

HEADS, SEQ, HEAD_DIM = 3, 64, 64
DIM = HEADS * HEAD_DIM
SCALE = HEAD_DIM ** -0.5


def _qkv_unfused(x: torch.Tensor, qkv_w: torch.Tensor, qkv_b: torch.Tensor) -> torch.Tensor:
    batch, seq, dim = x.shape
    head_dim = dim // HEADS
    wq, wk, wv = qkv_w[:dim], qkv_w[dim:2 * dim], qkv_w[2 * dim:]
    bq, bk, bv = qkv_b[:dim], qkv_b[dim:2 * dim], qkv_b[2 * dim:]
    q = linear(x, wq, bq).reshape(batch, seq, HEADS, head_dim).transpose(1, 2)
    k = linear(x, wk, bk).reshape(batch, seq, HEADS, head_dim).transpose(1, 2)
    v = linear(x, wv, bv).reshape(batch, seq, HEADS, head_dim).transpose(1, 2)
    out = attention_triton(q, k, v, SCALE)
    return out.transpose(1, 2).reshape(batch, seq, -1)


def _arms_for_batch(batch: int, dtype: torch.dtype):
    q, k, v = (torch.randn(batch, HEADS, SEQ, HEAD_DIM, device="cuda", dtype=dtype)
               for _ in range(3))
    x = torch.randn(batch, SEQ, DIM, device="cuda", dtype=dtype)
    qkv_w = torch.randn(3 * DIM, DIM, device="cuda", dtype=dtype) * 0.05
    qkv_b = torch.randn(3 * DIM, device="cuda", dtype=dtype)
    return {
        "torch": lambda: attention_torch(q, k, v, SCALE),
        "triton_composed": lambda: attention_triton(q, k, v, SCALE),
        "triton_qkv_fused": lambda: attention_qkv_fused(x, qkv_w, qkv_b, HEADS, SCALE),
        "triton_qkv_unfused": lambda: _qkv_unfused(x, qkv_w, qkv_b),
    }


def _bytes_theoretical(batch: int) -> int:
    # q, k, v reads (3x) + score-matrix write and read-back (2x, the
    # materialized [B, H, 64, 64] round trip this rung deliberately pays
    # for) + output write (1x), all batch*heads*seq*seq elements at fp32.
    qkv = batch * HEADS * SEQ * HEAD_DIM
    scores = batch * HEADS * SEQ * SEQ
    return (3 * qkv + 2 * scores + qkv) * 4


SPEC = RunnerSpec(kernel="attention", arms_for_batch=_arms_for_batch,
                  bytes_theoretical=_bytes_theoretical)


if __name__ == "__main__":
    main(SPEC)
