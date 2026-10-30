# Tritonformer

A small vision transformer forward pass (CIFAR-10 size: 32x32 input, patch 4, dim 192, depth 6,
3 heads, MLP 768) written in Triton, used to measure when kernel fusion helps and when it hurts.

- GTX 1650 Ti (Turing, sm_75, 16 SMs, ~1 MB L2, no tensor cores): the original card. Gone since
  2026-10; its rows are kept, not regenerated.
- NVIDIA L4 (Ada, sm_89, 58 SMs, 48 MB L2, tensor cores) on Modal: replication and all later work.
- Baselines: the matching eager PyTorch op, the unfused Triton kernels, and `torch.compile`.

## Summary

- **A tiled flash attention kernel beats eager PyTorch at every batch size on the L4**, 2.3x at
  batch 1 and 1.15x at batch 512. It is the first Triton kernel in this project to do so, and the
  first fusion other than LayerNorm + residual to beat its own unfused arm.
- **The earlier attention loss was mostly cuBLAS versus Triton GEMM, not fusion.** On these 64x64
  problems a plain Triton batched matmul beats cuBLAS's 128x128 sgemm by 12-33%, and the original
  single-tile flash kernel spent 146 MB per call on register spills.
- **The MLP mega-kernel loss is real.** It sits on the shared-memory roofline (93% of L1TEX peak,
  0.9% of DRAM), with 8x the bank conflicts of the composed kernels and one block per SM.
- **ncu's default cache flush overstated the DRAM savings.** Warm, neither LayerNorm + residual arm
  touches DRAM on the L4, which is why that fusion stopped winning there.
- **End to end, the Triton model still loses to eager PyTorch** on both GPUs, and `torch.compile`
  beats everything.

![Attention on the L4: latency of four Triton attention arms relative to eager PyTorch](assets/attention-l4.svg)

## Attention

Two new arms, both on the L4 (IEEE fp32, batch sweep, median of 30 interleaved reps):

- `triton_composed_bmm`: the composed arm (q@k^T, softmax, p@v) with cuBLAS replaced by a Triton
  batched matmul using `_linear_kernel`'s 64x64x32 tiling. Three launches; the scale is the GEMM
  epilogue.
- `triton_flash_tiled`: 16 query rows per program, a loop over 32-wide key/value tiles, online
  softmax. 60 registers and no spills, against 255 registers and 94 MB of local-memory loads for the
  single-tile `triton_flash`.

Median latency in ms:

| batch | torch | composed (cuBLAS) | composed (Triton GEMM) | flash, single tile | flash, tiled |
|---|---|---|---|---|---|
| 1 | 0.031 | 0.031 | 0.020 | 0.028 | **0.013** |
| 8 | 0.039 | 0.040 | 0.030 | 0.034 | **0.023** |
| 32 | 0.067 | 0.074 | 0.065 | 0.103 | **0.061** |
| 128 | 0.236 | 0.266 | 0.195 | 0.279 | **0.190** |
| 512 | 0.931 | 1.090 | 0.845 | **0.783** | 0.809 |

- Tiled flash vs the composed arm on the same GEMM: 3-35% faster at every batch. Below batch 32 the
  margin is launch count; above it the fusion itself is worth 3-5%.
- Single-tile flash vs the same arm: 1.14-1.60x slower at batch 1-128, 7% faster at 512. Why it
  wins at 512 is not explained.
- Triton GEMM vs cuBLAS in the composed arm: 0.67-0.88x. cuBLAS picks a 128x128 tile for a 64x64
  problem.
- The torch baseline is the unfused eager path, not `F.scaled_dot_product_attention`.

## MLP

Extended Nsight Compute counters at batch 128 on the L4, cold cache:

| kernel | DRAM % peak | L2 % peak | L1TEX % peak | L1TEX bytes | shared bank conflicts | top stall | blocks/SM |
|---|---|---|---|---|---|---|---|
| `_mlp_fused_kernel` | 0.9 | 7.2 | 93.3 | 630 MB | 264M | `mio_throttle` 11.2 | 1 |
| `_linear_gelu_kernel` | 9.4 | 14.0 | 79.7 | 178 MB | 33M | `short_scoreboard` 3.7 | 2 |
| `_linear_kernel` | 24.9 | 13.4 | 86.5 | 158 MB | 33M | `mio_throttle` 4.3 | 3 |

- The fused kernel is bound by shared memory, not L2 or DRAM: 1.9x the L1TEX bytes of both composed
  kernels together, from one block per SM (51,200 B shared, 210 registers x 256 threads).
- `num_stages` never applied on Turing. The H loop is a runtime `scf.for`; two stages need 83,968 B
  of shared memory against Turing's 65,536 B. On the L4 two stages compile (38 `cp.async`), three
  do not (149,504 B against 101,376 B).

## Counters

Total DRAM bytes per arm at batch 128 on the L4, with ncu's default cache flush (`counters.csv`)
and with `--cache-control none` (`counters_cache_control_none.csv`):

| arm | cold | warm |
|---|---|---|
| attention, composed (cuBLAS) | 43.2 MB | 29.2 MB |
| attention, composed (Triton GEMM) | 21.4 MB | 13.3 MB |
| attention, flash single tile | 24.9 MB | 8.1 MB |
| attention, flash tiled | 21.0 MB | 0.04 MB |
| layernorm_residual, two kernels | 21.1 MB | 0.02 MB |
| layernorm_residual, fused | 13.6 MB | 0.08 MB |
| mlp, composed | 45.4 MB | 37.8 MB |
| mlp, fused | 8.3 MB | 0.8 MB |

- Every cold ratio reported before this follow-up is a flushed-cache number.
- Warm numbers are a lower bound on the flush's effect, not the `do_bench` loop: ncu still restores
  device memory between replay passes. The composed MLP's second kernel still reads 32 MB warm.

## Ladder

Each fused kernel against the unfused Triton kernels it replaces, on the GTX 1650 Ti unless noted:

![Fusion ladder: latency of each fused or tuned kernel relative to the unfused kernel(s) it replaces, on the GTX 1650 Ti](assets/fusion-ladder.svg)

| step | result | why |
|---|---|---|
| GeLU rewrite (no fusion) | no change (0-5%, noise) | already one read and one write |
| LayerNorm + residual add | 22-31% faster on 1650 Ti; 29% faster to 11% slower on L4 | DRAM traffic down 19.6%; warm, no traffic to save on the L4 |
| Linear + GeLU epilogue, naive | 7-14% faster | DRAM traffic down 33% |
| Linear + GeLU epilogue, both autotuned | 9-13% slower | fused kernel needs a smaller tile |
| Flash attention, single tile | 1.49-1.70x slower | 255 registers, spills, one block per SM |
| Flash attention, tiled (L4 only) | 3-35% faster than the same-GEMM composed arm | 60 registers, no spills, 4x the grid |
| Whole-MLP kernel | 3.1-3.8x slower | shared-memory roofline, one block per SM |
| Full fused block | slower | inherits the attention and MLP losses |

Autotuning one kernel against PyTorch: softmax goes from 1.29-3.74x slower to parity, and faster
below batch 32. Plain matmul at the model's shapes goes from 1.9-4.3x slower to 1.13-4.25x slower;
cuBLAS wins there, and loses only on the 64x64 attention shapes.

## Model

End-to-end latency relative to eager PyTorch (above 1.0 is slower), `vit_forward` in `latency.csv`:

| GPU, variant | batch 1 | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| 1650 Ti, unfused Triton kernels | 2.26x | 2.15x | 1.95x | 2.27x | 1.91x |
| 1650 Ti, fully fused | 5.01x | 5.27x | 5.07x | 5.93x | 4.74x |
| L4, unfused Triton kernels | 2.05x | 1.80x | 1.67x | 1.38x | 1.43x |
| L4, fully fused | 1.80x | 1.58x | 3.95x | 2.68x | 2.57x |
| L4, `torch.compile` | 0.63x | 0.71x | 0.83x | 1.01x | 1.31x |

`torch.compile` calls cuBLAS for the matmuls, fuses LayerNorm with the residual add, and does not
fuse the MLP. The fused model still uses the single-tile attention kernel; the tiled one has not
been wired into the block.

## Findings

1. **Less DRAM traffic did not mean less time.** Attention and MLP cut cold-cache traffic 37% and up
   to 73% and were still 1.5-3.8x slower on the 1650 Ti.
2. **The MLP loss is a shared-memory roofline, not registers and not DRAM.** Cutting registers to
   128 left occupancy at 25%; the kernel runs at 93% of L1TEX peak with one block per SM, and no
   second pipeline stage fits on Turing.
3. **The attention loss was mostly the baseline.** Against the same GEMM implementation, and with a
   kernel that does not spill, fusion wins at every batch on the L4.
4. **ncu's default cache flush inflates DRAM savings on a 48 MB L2.** The LayerNorm + residual
   intermediate never reaches DRAM warm, so the L4 loss at batch 128 is expected.
5. **`tl.dot` defaults to TF32 on tensor-core GPUs.** 70 of 153 tests fail on the L4 until
   `TRITON_F32_DEFAULT=ieee` is set. At matched precision cuBLAS beats Triton 36-90% at the model's
   GEMM shapes.
6. **Not explained:** the single-tile flash kernel winning at batch 512; the MLP's 81.4% cold traffic
   drop exceeding the arithmetic (consistent with the flush); the fully fused model beating unfused
   at batch 1 and 8 on the L4.

## Method

- Latency: median of 30 interleaved `triton.testing.do_bench` reps per (kernel, variant, batch),
  fp32, in `bench/results/latency.csv`.
- Counters: Nsight Compute, in `bench/results/counters.csv` (cold) and
  `bench/results/counters_cache_control_none.csv` (warm). The extended metric set adds L2, L1TEX,
  stall and occupancy-limit counters. Every capture checks the captured kernel's name.
- Clocks: flagged per row on the 1650 Ti (60 of 405 rows, 300-1950 MHz under thermal limits). The
  L4 runs with `--clock-control none` and unlocked clocks; arms within a batch are interleaved and
  comparable, batches against each other are not.
- Predictions for the first L4 run were committed before it ran.

## Run

Needs an NVIDIA GPU, Python 3.10+, PyTorch, torchvision and Triton (the Modal image pins
`torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `triton==3.6.0`).

```
uv sync
uv pip install torch==2.11.0 torchvision==0.26.0 triton==3.6.0 pytest numpy --extra-index-url https://download.pytorch.org/whl/cu128

uv run pytest                                   # set TRITON_F32_DEFAULT=ieee on tensor-core GPUs
scripts/lock_clocks.sh 1200                     # optional, needs root; reset with: sudo nvidia-smi -rgc
uv run python -m bench.run_attention            # one kernel's sweep, appends to bench/results/latency.csv
uv run python -m bench.run_sweep                # end-to-end ViT forward, every block variant
uv run python -m bench.collect_counters         # Nsight Compute counters, needs ncu
uv run python -m bench.collect_counters --metrics extended --arm bench.run_mlp:mlp:triton_fused:128
TRITONFORMER_NCU_CACHE_CONTROL=none uv run python -m bench.collect_counters --out bench/results/counters_cache_control_none.csv --arm ...
```

Each `bench/run_*.py` runs a batch sweep by default. With `--kernel --variant --batch --dtype` it
runs one arm for profiling and writes no CSV.

On Modal, one step per call (`build_check`, `unit_tests`, `smoke`, `tests`, `sweep_kernels`,
`sweep_vit`, `counter_grid`, `followup_tests`, `followup_counters`, ...):

```
modal run modal_app.py --step smoke --out-dir out/
modal run modal_app.py --step tests --tests-args "tests/test_attention.py" --out-dir out/
modal run modal_app.py --step sweep_kernels --modules bench.run_attention --out-dir out/
modal run modal_app.py --step counter_grid --metrics extended --cache-control none --clock-control none --out-dir out/
python modal_app.py merge --csv out/<file>.csv   # append returned rows to bench/results/latency.csv
python modal_app.py merge --csv out/counter_grid_warm.csv --into bench/results/counters_cache_control_none.csv
```

`scripts/train_reference.py` trains the PyTorch reference model on CIFAR-10 and writes
`data/checkpoint.pt`, which the end-to-end test loads.

## Scope

Forward-pass inference only; no training loop in Triton. Per-experiment logs, the external review
and the follow-up plan are local notes under `docs/`, which is not tracked.
