# Tritonformer

Forward pass of a small vision transformer (CIFAR-10 size: 32x32 input, patch 4, dim 192, depth 6,
3 heads, MLP 768) written in Triton, used to measure when kernel fusion helps and when it hurts.

- Primary GPU: GTX 1650 Ti (Turing, sm_75, 16 SMs, ~1 MB L2, no tensor cores).
- Replication: NVIDIA L4 (Ada, sm_89, 58 SMs, 48 MB L2, tensor cores) on Modal.
- Baselines: the matching eager PyTorch op, the unfused Triton kernels, and `torch.compile`.

![Fusion ladder: latency of each fused or tuned kernel relative to the unfused kernel(s) it replaces, on the GTX 1650 Ti](assets/fusion-ladder.svg)

## Results

**End to end, the Triton model is slower than eager PyTorch on both GPUs.** Latency relative to
eager PyTorch (above 1.0 is slower), from `bench/results/latency.csv` (`vit_forward`):

| GPU, variant | batch 1 | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| 1650 Ti, unfused Triton kernels | 2.26x | 2.15x | 1.95x | 2.27x | 1.91x |
| 1650 Ti, fully fused | 5.01x | 5.27x | 5.07x | 5.93x | 4.74x |
| L4, unfused Triton kernels | 2.05x | 1.80x | 1.67x | 1.38x | 1.43x |
| L4, fully fused | 1.80x | 1.58x | 3.95x | 2.68x | 2.57x |
| L4, `torch.compile` | 0.63x | 0.71x | 0.83x | 1.01x | 1.31x |

`torch.compile` beats every hand-written variant. Its generated code calls cuBLAS for the matmuls,
fuses LayerNorm with the residual add, and does not fuse the MLP.

**Per fusion step**, each fused kernel against the unfused Triton kernels it replaces (1650 Ti unless
noted):

| step | result | why |
|---|---|---|
| GeLU rewrite (no fusion) | no change (0-5%, noise) | already one read and one write |
| LayerNorm + residual add | 22-31% faster on 1650 Ti; 29% faster to 11% slower on L4 | DRAM traffic down 19.6%, registers 28 to 27 |
| Linear + GeLU epilogue, naive | 7-14% faster | DRAM traffic down 33% |
| Linear + GeLU epilogue, both autotuned | 9-13% slower | fused kernel needs a smaller tile, loses the large-tile configs |
| Flash-style attention | 1.49-1.70x slower (1.60-2.24x vs PyTorch) | 255-register cap, local-memory spills, one block per SM; DRAM traffic down 37% |
| Whole-MLP kernel (linear + GeLU + linear) | 3.1-3.8x slower | serialized inner loop over the hidden dim; DRAM traffic down up to 73% |
| Full fused block | slower | inherits the attention and MLP losses |

Autotuning a single kernel (`triton.autotune` over tile sizes plus L2 pid swizzle), against PyTorch:
softmax goes from 1.29-3.74x slower to parity, and faster below batch 32. Plain matmul goes from
1.9-4.3x slower to 1.13-4.25x slower; cuBLAS still wins.

## Findings

1. **Less DRAM traffic did not mean less time.** The attention and MLP kernels cut traffic 37% and
   up to 73% and were still 1.5-3.8x slower. LayerNorm + residual is the only fusion that won, and
   the only one that did not add registers.
2. **The MLP kernel's slowdown is not register pressure.** Nsight Compute showed 226 registers and
   25% occupancy. Cutting registers to 128 left occupancy at exactly 25.00%; halving shared memory
   changed nothing and made latency worse. Latency tracks the inner-loop iteration count over the
   hidden dim (24 vs 48). `num_stages` 1-3 changed latency by 0.003%, because the loop is unrolled
   at compile time.
3. **The results hold on the L4.** The fused penalty shrinks (5.93x to 2.68x at batch 128) but does
   not become a win. The exception is the L4 at batch 1 and 8, where fully fused beats unfused. Fusion's traffic reduction also holds:
   1.50-1.55x less for LayerNorm + residual, 2.02-2.47x for attention, 3.54-4.94x for the MLP. The
   48 MB L2 did not absorb the intermediates.
4. **`tl.dot` defaults to TF32 on tensor-core GPUs.** The 1650 Ti has none, so it always ran fp32.
   On the L4, 70 of 153 tests fail (worst error 2.5e-3 against a 1e-4 tolerance) until
   `TRITON_F32_DEFAULT=ieee` is set. At matched precision cuBLAS still wins: Triton is 36-90% slower
   at IEEE fp32 and 3-45% slower at TF32. An earlier reading that Triton beat cuBLAS by 12-18%
   compared Triton on TF32 with cuBLAS on fp32 and is withdrawn.
5. **Measured but not explained:** the MLP's 81.4% traffic drop on the fused arm (more than the
   arithmetic predicts); LayerNorm + residual losing on the L4 at batch 128 while its traffic ratio
   barely moves; flash attention winning on the L4 at batch 1, 8 and 512 with a flat traffic ratio;
   the batch 1-8 rank flip in finding 3.

## How it was measured

- Latency: median of 30 interleaved `triton.testing.do_bench` reps per (kernel, variant, batch), fp32, in
  `bench/results/latency.csv`. Hardware counters (DRAM bytes, registers, warps active, hit rates)
  from Nsight Compute, in `bench/results/counters.csv`.
- Predictions for the L4 run were committed before it ran; the final write-up is a byte-identical
  extension of that commit.
- Clock drift is flagged per row (`flagged` column). 60 of 405 rows on the 1650 Ti are flagged,
  with clocks from 300 to 1950 MHz under thermal limits. Six `linear_gelu` cross-card comparisons
  are withdrawn because the 1650 Ti ran at about 23% of nominal clock.
- One counter capture landed on an elementwise multiply in the benchmark's setup instead of the
  kernel under test. `bench/collect_counters.py` now checks the captured kernel's name.

## Run

Needs an NVIDIA GPU, Python 3.10+, and PyTorch, torchvision and Triton (the Modal image pins
`torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `triton==3.6.0`).

```
uv sync
uv pip install torch==2.11.0 torchvision==0.26.0 triton==3.6.0 pytest numpy --extra-index-url https://download.pytorch.org/whl/cu128

uv run pytest                                   # kernel and end-to-end tests; set TRITON_F32_DEFAULT=ieee on tensor-core GPUs
scripts/lock_clocks.sh 1200                     # optional, needs root; reset with: sudo nvidia-smi -rgc
uv run python -m bench.run_layernorm_residual   # one kernel's sweep, appends to bench/results/latency.csv
uv run python -m bench.run_sweep                # end-to-end ViT forward, every block variant
uv run python -m bench.collect_counters         # Nsight Compute counters, needs ncu
```

Each `bench/run_*.py` runs a batch sweep by default. With `--kernel --variant --batch --dtype` it
runs one arm for profiling and writes no CSV.

L4 replication on Modal, one step per call (`build_check`, `smoke`, `tests`, `sweep_kernels`,
`sweep_vit`, `counter_grid`, ...):

```
modal run modal_app.py --step smoke --out-dir out/
python modal_app.py merge --csv out/<file>.csv   # append returned rows to bench/results/latency.csv
```

`scripts/train_reference.py` trains the PyTorch reference model on CIFAR-10 and writes
`data/checkpoint.pt`, which the end-to-end test loads.

## Scope

Forward-pass inference only; there is no training loop in Triton. Per-experiment logs (register
sweeps, counter dumps, prediction write-ups) are local notes and not in this repository.
