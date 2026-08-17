# Finding 05: The whole-MLP mega-kernel (over-fusion)

Rung 12, the project's central document. The task brief predicted this
kernel would hurt: hold the whole `H=768`-wide hidden activation across
both matmuls in one program, expect it to spill to local memory and
occupancy to collapse, mirroring Task 15's flash-attention story. What
actually happened is a different, earlier failure than predicted, and the
kernel that had to be built to get any numbers at all shows a different
mechanism than register spilling.

## Setup

- GPU: GTX 1650 Ti, sm_75, 16 SMs, no tensor cores, 4 GB, ~192 GB/s peak
  DRAM bandwidth, **64KB shared memory/SM, 65,536 registers/SM, 255
  registers/thread max**.
- Shape: `x` is `[batch, 64, 192]`, `w1` is `[768, 192]`, `w2` is
  `[192, 768]`, fp32. `BATCHES = [1, 8, 32, 128, 512]`.
- `mlp_composed`: Task 12's `linear_gelu` (one kernel) then Task 10's
  `linear` (one kernel) -- two launches, the `[batch, 64, 768]` hidden
  activation round-trips through DRAM between them.
- `mlp_fused`: one kernel, `_mlp_fused_kernel` -- see below for why its
  final form differs from the brief's reference implementation.
- Commit at time of measurement: this task's commit (see report).

## Step 1: the brief's reference kernel does not compile, at any `BLOCK_M`

The brief's reference `_mlp_fused_kernel` holds the entire `H=768` width
per program (`BLOCK_H = next_power_of_2(768) = 1024`) and the entire
`D=192` width (`BLOCK_D = next_power_of_2(192) = 256`), tiling only the
batch dimension via `BLOCK_M`. Its documented remediation for a shared-
memory overflow is "reduce `BLOCK_M` to 16, then 8." Every value tried
failed, including values well past what the brief asked for:

| `BLOCK_M` | Required shared memory | Hardware limit | Result |
|---:|---:|---:|---|
| 32 | 1,180,672 | 65,536 | FAILED |
| 16 | 1,115,136 | 65,536 | FAILED |
| 8  | 1,082,368 | 65,536 | FAILED |
| 4  | 1,065,984 | 65,536 | FAILED |
| 2  | 1,056,768 | 65,536 | FAILED |
| 1  | 1,053,696 | 65,536 | FAILED |

Exact error text (identical shape, only the byte count changes):
```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: <N>, Hardware limit: 65536. Reducing block sizes or `num_stages`
may help.
```

**No `BLOCK_M` compiles, down to `BLOCK_M=1`.** The reason is visible in
the numbers: reducing `BLOCK_M` by 24 (32 -> 8) only reduces the
requirement by 98,304 bytes (~4,096 bytes per unit of `BLOCK_M`, matching
the `[BLOCK_M, BLOCK_H]` hidden tile's own size), while the *floor* stays
just above 1,048,576 bytes regardless of `BLOCK_M`. That floor is
`BLOCK_D * BLOCK_H * 4 = 256 * 1024 * 4 = 1,048,576` bytes -- one `w1` (or
`w2`) tile, held whole because the reference design never tiles `H`. That
term is **~16x the 65,536-byte/SM budget** and does not shrink no matter
how small a batch tile a single program owns. **The over-fusion ceiling
this rung actually finds is a compile-time wall, not a runtime spill: the
literal "hold everything" design is infeasible on this card at any batch
tile size**, which is a stronger result than the brief anticipated (it
expected the spill/occupancy failure mode from Task 15 to repeat, not a
kernel that cannot be launched at all).

## Step 2: what it took to get a running kernel

To measure anything, `H` had to be tiled with a reduction loop -- the
same K-loop structure `_linear_kernel` already uses over its own
reduction dimension, just applied to `H` instead of `K`. Each iteration
computes a `[BLOCK_M, BLOCK_H]` hidden tile, applies GeLU to it, multiplies
it into the output accumulator, and discards it -- the hidden activation
still never reaches DRAM (each tile lives only in registers between the
two `tl.dot` calls), so this still satisfies "one kernel, hidden
dimension never leaves the SM." It just never holds all 768 columns at
once, which the reference design's fixed-`BLOCK_H=1024` cannot avoid on
64KB shared memory.

The largest working `BLOCK_H` tile was found empirically (`BLOCK_D=256`
fixed throughout, correctness checked against `F.linear`/`F.gelu` at
`rtol=atol=1e-4`):

| `BLOCK_M` | `BLOCK_H` | Result |
|---:|---:|---|
| 64 | 32 | FAILED: shared memory, Required 106,496 |
| 32 | 32 | FAILED: shared memory, Required 69,632 |
| **16** | **32** | **COMPILED, correct** |
| 64 | 16 | FAILED: shared memory, Required 86,016 |
| 32 | 16 | COMPILED, correct |
| 16 | 16 | COMPILED, correct |
| 64 | 8  | FAILED: `tl.dot` minimum-dimension error |
| 8  | 64 | FAILED: shared memory, Required 75,776 |
| 8  | 32 | COMPILED, correct |
| 8  | 128 | FAILED: shared memory, Required 143,360 |
| 4  | 128 | FAILED: shared memory, Required 137,216 |
| 4  | 256 | FAILED: shared memory, Required 270,336 |

**`BLOCK_H = 32` is the ceiling** -- it fails even at the smallest
`BLOCK_M` tested (8), so it is not a batch-tiling artifact, it is fixed
by `BLOCK_D=256` alone (`256 * 32 * 4 = 32,768` bytes, already half the
65,536-byte budget once `x`, the hidden tile, and bias loads are added).
`BLOCK_M=16, BLOCK_H=32` was selected: the largest `BLOCK_M` at the
largest working `BLOCK_H`, `num_warps=8, num_stages=1`. This means the
registered `triton_fused` kernel loops **24 times** (`768 / 32`) per
program instance -- nothing like the single-shot "hold it all" design the
brief described, because that design cannot exist on this hardware.

## Correctness

All 9 tests pass (`pytest tests/test_mlp.py -v`), including the
batch-512 case, which exercises the `M`-mask on the store path the brief
flagged as a second plausible failure mode; no masking bug was found.
Full suite: 99/99 green.

## Latency across the batch sweep (median of 30 interleaved reps, fp32)

| batch | torch ms | triton_composed ms | triton_fused ms | fused/composed | fused/torch |
|------:|---------:|--------------------:|------------------:|---:|---:|
|     1 |   0.0416 |               0.1249 |             0.3871 | 3.10x | 9.32x |
|     8 |   0.1759 |               0.3977 |             1.4400 | 3.62x | 8.19x |
|    32 |   0.7909 |               1.5872 |             5.7907 | 3.65x | 7.32x |
|   128 |   2.5515 |               6.5539 |            23.0468 | 3.52x | 9.03x |
|   512 |  12.2599 |              26.0158 |            99.6272 | 3.83x | 8.13x |

**`triton_fused` loses at every batch, including batch 1 -- there is no
latency crossover.** The brief's hypothesis was that fusion wins at small
batch (fewer launches dominate) and loses at large batch (occupancy/spill
dominate). Neither half of that held: `triton_fused` is 3.1-3.8x slower
than `triton_composed` uniformly across the whole sweep, with a mild
*upward* trend (3.10x at batch 1 -> 3.83x at batch 512) rather than a
crossing.

## The four counters, both variants, both batch extremes

`triton_composed` launches two distinct kernels per call
(`_linear_gelu_kernel`, `_linear_kernel`); `triton_fused` launches one
(`_mlp_fused_kernel`). Cycle length confirmed empirically with
`ncu --launch-count`: 7 setup kernels from tensor construction precede a
steady repeating cycle of length 2 for `triton_composed` and length 1 for
`triton_fused` -- `launch_skip=7` (not the module default of 5) was
needed to land exactly on cycle start; `profile_kernel`'s
`expected_kernels` (2 and 1 respectively) confirmed each capture was
complete.

### DRAM traffic (`dram__bytes_read.sum` + `dram__bytes_write.sum`)

| batch | kernel | read | write | total |
|---:|---|---:|---:|---:|
| 1 | `_linear_gelu_kernel` | 686,880 | 138,304 | 825,184 |
| 1 | `_linear_kernel` | 823,904 | 87,520 | 911,424 |
| 1 | **composed total** | | | **1,736,608** |
| 1 | `_mlp_fused_kernel` | 1,305,536 | 233,760 | **1,539,296** |
| 512 | `_linear_gelu_kernel` | 303,484,576 | 100,459,072 | 403,943,648 |
| 512 | `_linear_kernel` | 321,221,120 | 25,135,936 | 346,357,056 |
| 512 | **composed total** | | | **750,300,704** |
| 512 | `_mlp_fused_kernel` | 176,932,096 | 25,198,400 | **202,130,496** |

DRAM traffic is **DOWN** for fused at both extremes, matching the brief's
predicted direction: 11.4% lower at batch 1 (1,736,608 -> 1,539,296, a
197,312-byte reduction) and **73.1% lower at batch 512** (750,300,704 ->
202,130,496, a 548,170,208-byte reduction) -- consistent with avoiding
the `[batch, 64, 768]` hidden-activation round trip, which is largest
relative to everything else precisely at large batch.

### `launch__registers_per_thread`

| batch | `_linear_gelu_kernel` | `_linear_kernel` | `_mlp_fused_kernel` |
|---:|---:|---:|---:|
| 1   | 168 | 128 | **226** |
| 512 | 168 | 128 | **226** |

**UP**, matching the predicted direction, and batch-independent as
expected (register allocation is a compile-time property). 226 is between
Task 12's `_linear_gelu_kernel` (168) and Task 15's `_flash_kernel` (255,
the hardware max) -- elevated, but with headroom below the cap.

### `sm__warps_active.avg.pct_of_peak_sustained_active`

| batch | `_linear_gelu_kernel` | `_linear_kernel` | `_mlp_fused_kernel` |
|---:|---:|---:|---:|
| 1   | 12.49% | 12.50% | **25.00%** |
| 512 | 37.35% | 49.44% | **25.00%** |

At batch 512 this is **DOWN** for fused relative to both composed
kernels, matching the predicted direction. At batch 1 it is the
*opposite*: fused's 25.00% is **higher** than either composed kernel's
12.49-12.50%. The explanation is occupancy is bound by two different
constraints at the two extremes. `_mlp_fused_kernel`'s occupancy is
register-limited at every batch: 226 regs/thread x 256 threads/block
(`num_warps=8`) = 57,856 registers/block; 65,536 / 57,856 = 1.13 -> 1
block/SM; 1 block x 8 warps / 32 max warps/SM = **25.0%**, matching the
measurement exactly at both batches. `_linear_gelu_kernel`/`_linear_kernel`
are register-limited to 3-4 blocks/SM (37.5%/50%, matching Task 12's
calibration and this task's own batch-512 measurement), but at batch 1
their grids are tiny (12 and 3 blocks respectively, for only 64 total
output rows) -- far fewer blocks than 16 SMs x their register ceiling, so
occupancy is *grid-size-limited*, not register-limited, and lands below
`_mlp_fused_kernel`'s fixed register-limited floor. `_mlp_fused_kernel`'s
own grid at batch 1 is 4 blocks (`cdiv(64, 16)`) -- also tiny, but its
register ceiling was already exactly 1 block/SM, so grid-limiting and
register-limiting coincide and its occupancy stays flat.

### `l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum` / `..._st.sum`

Caveat, so these counters aren't overstated anywhere in this document:
`l1tex__t_bytes_pipe_lsu_mem_local_op_ld/st.sum` is measured at L1, not
at DRAM -- some of that traffic may be absorbed by L1/L2 and never
actually reach DRAM, so it is **not** directly comparable byte-for-byte
to `dram__bytes_*`. Read every local-memory figure in this document as
indicative of the mechanism (a kernel spilling registers to local
memory, or not), not as an exact DRAM-traffic accounting. (Same wording
as `docs/findings/04-flash-attention.md`, which measured this first, on
`_flash_kernel`.)

| batch | `_linear_gelu_kernel` | `_linear_kernel` | `_mlp_fused_kernel` |
|---:|---:|---:|---:|
| 1   | 0 / 0 | 0 / 0 | **0 / 0** |
| 512 | 0 / 0 | 0 / 0 | **0 / 0** |

**Zero everywhere, at both batches, for all three kernels.**
`_mlp_fused_kernel` does not spill. This directly contradicts the
predicted signature (non-zero, "direct evidence of spilling"). 226
registers/thread fits with room to spare under the 255-register hardware
cap; forcing `BLOCK_H` down to 32 to make the kernel compile at all also
kept its per-iteration working set well inside the register file. The
spill mechanism that dominated Task 15's flash-attention story does not
repeat here.

## Register-pressure curve across rungs

| kernel | regs/thread | warps_active | spill (ld+st bytes) |
|---|---:|---:|---:|
| `_linear_kernel` (Task 12) | 128 | 49.37% | 0 |
| `_linear_gelu_kernel` (Task 12) | 168 (+31%) | 37.18% | 0 |
| **`_mlp_fused_kernel` (this task)** | **226 (+34% vs linear_gelu)** | **25.00%** | **0** |
| `_flash_kernel` (Task 15) | 255 (hardware max) | 12.48% | 60,555,264 (39,714,816 ld + 20,840,448 st) |

This kernel is the missing middle point on the curve the project has been
building: register pressure climbs steadily (128 -> 168 -> 226 -> 255)
and occupancy falls steadily (49.37% -> 37.18% -> 25.00% -> 12.48%) as
each rung asks a single program to hold more live state, right up until
Task 15's flash kernel hits the 255-register hardware ceiling and starts
spilling. `_mlp_fused_kernel` sits one step before that cliff: elevated
registers, halved-again occupancy, but the compiler still found a
spill-free allocation. Whether the *next* rung up spills or not looks
like it depends on how close to 255 registers a kernel's design pushes,
not on register pressure alone -- but note this kernel only reached 226
registers because it was forced into small `BLOCK_H=32` tiles by a
*shared-memory* ceiling that bound first, before registers had the chance
to become the binding constraint the way they did for flash attention.

## At what batch size does fusion stop paying?

**It never starts.** `triton_fused` is slower than `triton_composed` at
every measured batch (1, 8, 32, 128, 512) -- there is no crossover point,
and the gap widens slightly rather than narrowing as batch grows (3.10x
at batch 1 -> 3.83x at batch 512). If forced to name a single counter,
none of the four brief-predicted counters (DRAM traffic, registers,
occupancy, spill) directly explains the loss at *every* batch: DRAM
traffic favors fused throughout (it is never the bottleneck being
measured), registers and occupancy move in the predicted direction only
at batch 512, and spill is zero everywhere. **The counter that actually
explains this rung's result is not one of the four -- it is the forced
`BLOCK_H=32` tile size itself**, a consequence of the compile-time
shared-memory wall in Step 1. `_linear_gelu_kernel` and `_linear_kernel`
use `BLOCK_N=64`/`BLOCK_K=32`-scale tiles reused across the fewest
possible kernel launches; `_mlp_fused_kernel` is forced to re-derive the
same computation through 24 sequential `BLOCK_H=32`-wide iterations per
program, each with its own global-memory load of a `w1`/`w2` slice, GeLU
epilogue, and mask arithmetic, with no double buffering (`num_stages=1`)
to hide any of it. The fusion this rung set out to build was never
reachable on this hardware; the fusion it was forced to build instead
pays a fixed, batch-independent per-iteration tax that the two decomposed
kernels, run back-to-back, do not.

## Did the fused kernel ever win?

**No, not once, at any measured batch.** This is worth stating plainly
per the brief's own instruction: this is not a case where over-fusion
"stops paying" past some threshold -- it never paid. What it implies
about this card's register file versus its memory system is that,
**for this particular MLP shape, the register file was never actually the
scarce resource that mattered** -- shared memory was, and it bound so
early (at the compile stage, for any batch tile) that the kernel never
got the chance to run the experiment the brief designed: "does the
register file survive holding the whole hidden width." It could not
survive holding even a modest slice of it (`BLOCK_H=64` also fails to
compile) without the compiler running out of the *other* on-chip resource
first. Flash attention (Task 15) got far enough into the register-spill
regime to show that trade-off directly; this kernel's honest result is
one step earlier in the same story: **on a 64KB-shared-memory, 16-SM,
no-tensor-core card, "one kernel, hold everything" over-fusion for a
768-wide MLP hits a hard architectural wall (shared memory for `tl.dot`
operand staging) well before the register file becomes the limiting
factor**, and the workaround needed to get the kernel running at all
introduces its own, larger latency cost that has nothing to do with
spilling.

## Summary: what this kernel actually establishes

- **The brief's reference kernel is infeasible on this hardware**, not
  merely disadvantaged: it fails to compile at every `BLOCK_M` from 32
  down to 1, because holding the whole `H=768` width requires a `w1`/`w2`
  tile (`BLOCK_D=256 x BLOCK_H=1024 x 4` bytes = 1,048,576 bytes) that is
  ~16x the 65,536-byte/SM shared-memory budget, independent of batch
  tiling.
- **The working replacement (`H`-tiled, `BLOCK_H=32`) loses on latency at
  every batch**, 3.10x-3.83x slower than the unfused `triton_composed`
  baseline, with no crossover.
- **It does not spill** (all four `l1tex__t_bytes_pipe_lsu_mem_local_op_*`
  captures are zero) -- the predicted spill mechanism from Task 15 does
  not repeat, because the shared-memory ceiling forced small enough tiles
  (226 registers/thread) to stay under the 255-register hardware cap with
  room to spare.
- **DRAM traffic drops as predicted** (11.4% at batch 1, 73.1% at batch
  512) and **registers rise as predicted** (226 vs 128/168), but neither
  translates into a latency win, reinforcing Task 15's lesson from a
  different angle: DRAM traffic, register pressure, and latency are three
  separate axes, and moving two of them in the "good" direction does not
  guarantee the third follows.
- **This is the project's clearest illustration that over-fusion can fail
  before it even gets the chance to spill.** The interesting failure mode
  this rung actually measures is a shared-memory compile-time wall
  forcing a fine-grained, high-launch-overhead tiling strategy -- a
  different, earlier mechanism than the register-spill story the brief
  anticipated, but no less honest a demonstration that "one kernel, hold
  everything" does not scale to a 768-wide hidden dimension on this card.

---

# Task 17 addendum: the fully fused transformer block (rung 13)

The deliberate far end of the ladder, predicted to hurt *more* than the
mega-MLP above. `block_composed` assembles the composition the fusion
ladder's plan specified for this rung (`layernorm`, `qkv_project`,
`attention_flash`, `linear`, `layernorm_residual`, `mlp_composed`) --
using `attention_flash` because rung 10 is the ladder's designated
attention rung, not because it measured fastest (it did not, see
`04-flash-attention.md`); `block_fused` is
identical except its last step is `mlp_fused` -- it inherits Task 16's
mega-MLP by construction, since that kernel is one of its six sub-calls.

## Step 1 (addendum): the monolithic variant does not compile either

A `triton_fused_monolithic` variant -- one kernel holding a `[BLOCK_M,
768]` MLP hidden tile *and* a `[64, 64]` attention tile simultaneously --
was attempted as a standalone scratch prototype (not committed; the
prototype is deliberately not numerically exact, since the point was
only to test joint shared-memory residency). It combined per-head Q/K/V
projection tiles, the attention score tile, the output-projection weight
tile, and the MLP's `w1`/`w2` tiles, using the same `BLOCK_H`-tiled MLP
loop structure Task 16 already found necessary. Every `BLOCK_H` tried
(32, 16) failed identically:

```
OutOfResources: out of resource: shared memory, Required: 262144,
Hardware limit: 65536. Reducing block sizes or `num_stages` may help.
```

**262,144 bytes required against a 65,536-byte budget -- 4x over --
and, as with Task 16's MLP wall, unaffected by `BLOCK_H`.** This is a
strictly larger version of Task 16's own wall: that rung's `w1`/`w2`
tile alone was already ~16x the budget; this rung additionally needs
attention's Q/K/V and score tiles live in the same program. A design
that was already infeasible cannot become feasible by adding more
simultaneous residents. Per the brief, this is treated as a legitimate,
informative upper bound, not a failure to work around: `block_composed`
stands as the maximum rung this ladder *reached* on this hardware, not
a proven maximum achievable rung. `block_composed` uses `attention_flash`
because rung 10 is the ladder's designated attention rung, not because
it measured fastest -- `attention_flash` was measured 1.49-2.24x
*slower* than `attention_composed` (above), so this composition is not
latency-optimal, and a `block_composed` variant substituting
`attention_composed` was never built or measured; that substitution is
left as future work.

## Step 5: launch counts (torch.profiler, not nsys)

**Substitution note:** the brief's Step 5 calls for `nsys profile
--stats=true`. `nsys` is present (`command -v nsys` -> `/usr/bin/nsys`,
exit 0), but is unusable in this environment: its importer is broken on
this host and a capture writes a ~3.3 GB `.qdstrm` file against a disk
that was at 98% utilization (~2.1 GB free) at measurement time. `nsys`
was not run and nothing was installed. `torch.profiler`
(`ProfilerActivity.CUDA`, `key_averages()`) was used instead -- Task 14
made the identical substitution for the same reason, and it answers the
same question (distinct CUDA kernel launches per call) without writing
any capture file to disk.

Per-arm launch counts, one call each, steady state (3 warm-up calls
first), batch 1 and batch 512 identical:

| arm | total CUDA launches/call | distinct kernel names |
|---|---:|---:|
| `torch` | 16 (b=1) / 17 (b=512) | 11 (b=1) / 9 (b=512) |
| `triton_composed` | 12 | 7 |
| `triton_fused` | **11** | 7 |

`triton_fused` has the fewest total launches, as predicted: it replaces
`mlp_composed`'s two kernels (`_linear_gelu_kernel`, `_linear_kernel`)
with `mlp_fused`'s one (`_mlp_fused_kernel`), a net reduction of one
launch per call. Distinct-kernel-name counts are equal (7 vs 7) because
`triton_composed`'s `_linear_kernel` name is shared across three
differently-shaped calls (QKV projection, output projection, and MLP's
second matmul) that collapse to one name; the *total launch count* is
the number that actually reflects the launch-overhead this rung is
about, and it favors `triton_fused` as expected.

## `profile_kernel` counters, both Triton arms, batch 1 and batch 512

Cycle length confirmed empirically with `profile_kernel`'s
`expected_kernels=7`: `launch_skip=20` (past tensor-construction setup
and one cold warm-up call) with `launch_count` set to each arm's true
per-call launch count (12 for `triton_composed`, 11 for `triton_fused`)
landed cleanly on one full steady-state cycle at both batches.

### DRAM traffic, summed across every launch in one full call cycle

| batch | arm | read | write | total |
|---:|---|---:|---:|---:|
| 1 | `triton_composed` | 3,001,184 | 452,736 | 3,453,920 |
| 1 | `triton_fused` | 2,804,960 | 417,344 | **3,222,304** |
| 512 | `triton_composed` | 1,240,748,224 | 532,858,208 | 1,773,606,432 |
| 512 | `triton_fused` | 792,514,656 | 432,452,672 | **1,224,967,328** |

DRAM traffic is down for fused at both extremes -- 6.7% at batch 1,
30.9% at batch 512 -- smaller reductions than Task 16's standalone MLP
(11.4% / 73.1%) because the block's other five sub-kernels (identical
between both arms) dilute the MLP-specific saving across a much larger
total.

### Per-kernel registers, occupancy, and spill (representative single
launch per distinct kernel name; `_linear_kernel` is invoked with
different shapes inside one call -- QKV projection, output projection,
and, in `triton_composed` only, the MLP's second matmul -- so the value
below is one captured instance, not a per-name sum)

| kernel | regs/thread | warps_active% (b=1) | warps_active% (b=512) | spill ld+st (b=1) | spill ld+st (b=512) |
|---|---:|---:|---:|---:|---:|
| `_layernorm_kernel` | 28 | 48.3 | 92.3 | 0 | 0 |
| `_layernorm_residual_kernel` | 27 | 48.3 | 94.1 | 0 | 0 |
| `_linear_kernel` | 128 | 12.5 | 49.4 | 0 | 0 |
| `_linear_gelu_kernel` (composed only) | 168 | 12.5 | 37.4 | 0 | 0 |
| `_flash_kernel` | **255 (hw max)** | 12.5 | 12.5 | **473,088** | **242,221,056** |
| `_mlp_fused_kernel` (fused only) | 226 | 25.0 | 25.0 | 0 | 0 |

Every one of these values matches its calibration from the rung that
introduced it (Task 12: 128/168 regs; Task 15: 255 regs, spilling; Task
16: 226 regs, zero spill, 25.0% register-limited occupancy) -- the block
kernel does not change any sub-kernel's individual behavior, only how
many of them run per call and in what sequence. `_flash_kernel` is the
only spilling kernel in the block, in both arms, at both batches; that
spill is a property of Task 15's attention kernel alone and has nothing
to do with block-level fusion.

## Latency crossover

| batch | torch ms | triton_composed ms | triton_fused ms | fused/composed | fused/torch |
|------:|---------:|--------------------:|------------------:|---:|---:|
|     1 |   0.1004 |               0.2247 |             0.4850 | 2.16x | 4.83x |
|     8 |   0.3465 |               0.7279 |             1.7585 | 2.42x | 5.07x |
|    32 |   1.4608 |               2.8155 |             6.9310 | 2.46x | 4.74x |
|   128 |   5.5702 |              11.1456 |            27.6488 | 2.48x | 4.96x |
|   512 |  22.1550 |              45.1341 |           112.7328 | 2.50x | 5.09x |

**No crossover exists.** `triton_fused` loses to `triton_composed` at
every measured batch, including batch 1, where the brief predicted
fewer launches would win. The predicted headline result -- fusion wins
small, loses large -- does not appear at the block level either,
consistent with Task 16's finding one rung down. The fused/composed
ratio drifts mildly upward with batch (2.16x -> 2.50x) rather than
crossing 1.0 in either direction.

One notable, unpredicted detail: the block-level fused/composed ratio
(2.16x-2.50x) is *smaller* than the standalone MLP's own fused/composed
ratio (3.10x-3.83x, from Step-5 above), even though `block_fused`
contains that exact MLP kernel unmodified. This is pure dilution: five
of the block's six sub-kernels are byte-identical between the two arms,
so the MLP's 3-4x local penalty gets averaged against 1x-ratio work
everywhere else, shrinking the *composite* ratio without the underlying
kernel improving at all. A rung's fusion penalty, measured in isolation,
overstates the damage it does once it is embedded in a larger pipeline
where it is only one of several launches.

## Conclusion: the condition under which fusion stops paying on this hardware

**Shared memory bounds fusion first, at compile time; registers collapse
occupancy second, but fusion is already dead before registers spill.**

Supporting evidence, each claim numbered and sourced:

- `layernorm_residual`: **+22-31% faster** than the unfused pair
  (`docs/findings/02-layernorm-fusion.md`).
- `linear_gelu`: **+9-11% faster** than the unfused pair
  (`docs/findings/03-epilogue-fusion.md`).
- `triton_qkv_fused` wins only at batch 1 (0.0548ms vs 0.0868ms for
  `triton_qkv_unfused`), then is a wash or a loss at every larger batch
  (13.1648ms vs 13.1332ms at batch 512) (`04-flash-attention.md`).
- `attention_flash`: **1.49x-2.24x slower** than `triton_composed` and
  `torch` across the full batch sweep 1-512 (1.69x/2.16x at batch 128
  specifically) (`04-flash-attention.md`) -- attention itself never
  paid.
- mega-MLP (`mlp_fused`): **3.10x-3.83x slower** (this document, Task 16).
- fused block (`block_fused`): **2.16x-2.50x slower** (this document,
  Task 17).
- Shared memory binds first, at compile time: the mega-MLP's `w1`/`w2`
  tile is 1,048,576 bytes against a 65,536-byte/SM budget (~16x over,
  invariant in `BLOCK_M`, Step 1 above), and the monolithic block
  kernel's 262,144-byte requirement is the same wall, one rung larger.
- Registers collapse occupancy second, and this is the *measured* cause
  of the mega-MLP's latency loss: 226 regs/thread x 256 threads/block =
  57,856 -> 1 block/SM -> 25.00% occupancy, register-limited at every
  batch (above).
- The chain "shared memory forces `BLOCK_H=32` -> tiles too small to
  amortize the fusion" is inferred from the compile-time arithmetic
  above, not isolated by any experiment that varies tile size
  independently of register pressure; the only pure shared-memory
  datum is the monolithic kernel, which never compiled and so was
  never measured directly.

The boundary between fusion that pays and fusion that doesn't sits at
the QKV-projection rung, several rungs before attention -- and
attention was already on the losing side, not the boundary itself.
