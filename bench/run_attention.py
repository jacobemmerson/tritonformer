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
"""
import torch

from bench.runner import RunnerSpec, main
from model.baseline.layers import attention as attention_torch
from model.kernels.attention import attention_composed as attention_triton

HEADS, SEQ, HEAD_DIM = 3, 64, 64
SCALE = HEAD_DIM ** -0.5


def _arms_for_batch(batch: int, dtype: torch.dtype):
    q, k, v = (torch.randn(batch, HEADS, SEQ, HEAD_DIM, device="cuda", dtype=dtype)
               for _ in range(3))
    return {
        "torch": lambda: attention_torch(q, k, v, SCALE),
        "triton_composed": lambda: attention_triton(q, k, v, SCALE),
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
