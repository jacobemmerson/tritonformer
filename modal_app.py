"""Experiment 3 (docs/findings/09-l4-replication.md): re-run this project's
measurements on an NVIDIA L4 (sm_89) to test whether the sm_75 conclusions are
card-specific.

Two structural decisions here exist to protect a hard GPU-time budget, and both
cost some elegance:

1. Every measurement body is a plain module-level function taking no Modal
   machinery, and the ``@app.function`` wrappers are one-line calls into them.
   The local host has a GTX 1650 Ti, so every body can be exercised on real
   hardware before an L4 container is ever started. Debugging an import error
   on a running L4 costs money; debugging it locally does not.
2. The entrypoints are split finer than the work naturally divides (``smoke``
   before ``tests`` before ``sweep``) so a broken image is discovered by the
   cheapest possible invocation and the run can be abandoned before the
   expensive ones.

Nothing here writes to bench/results/*.csv on the L4's behalf. The runners write
a container-local CSV, this module returns its text, and ``merge_csv`` appends
the data rows to the repo's CSV locally, after checking the header came back
byte-identical -- ``bench/harness.record()`` writes a header only when the file
is new, so a schema drift would corrupt 400+ existing rows silently.
"""
import argparse
import os
import subprocess
import sys
import time

REMOTE_ROOT = "/root/tritonformer"
LATENCY_CSV = "bench/results/latency.csv"
COUNTERS_CSV = "bench/results/counters.csv"

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

SWEEP_MODULES = [
    "bench.run_layernorm",
    "bench.run_layernorm_residual",
    "bench.run_gelu",
    "bench.run_softmax",
    "bench.run_linear",
    "bench.run_linear_gelu",
    "bench.run_attention",
    "bench.run_mlp",
    "bench.run_block",
    "bench.run_sweep",
]

# Ada allows 48 resident warps/SM against Turing's 32, while the register file
# is 65,536 32-bit registers/SM on both. Occupancy percentages therefore do NOT
# transfer between the two cards even when the block count does; see
# docs/findings/09-l4-replication.md. These are the values the experiment was
# pre-registered against; the code reads the live device properties instead and
# prints both, so a wrong assumption here shows up as a mismatch rather than
# silently propagating into the occupancy arithmetic.
EXPECTED_MAX_WARPS_PER_SM = {(7, 5): 32, (8, 9): 48}


def _describe_device() -> str:
    import torch
    import triton
    from bench.harness import commit_sha
    props = torch.cuda.get_device_properties(0)
    return "\n".join([
        f"device_name        {torch.cuda.get_device_name(0)}",
        f"compute_capability {torch.cuda.get_device_capability(0)}",
        f"sm_count           {props.multi_processor_count}",
        f"total_memory_GB    {props.total_memory / 1e9:.2f}",
        f"shared_mem_per_block_B {props.shared_memory_per_block}",
        f"l2_cache_B         {getattr(props, 'L2_cache_size', 'n/a')}",
        f"torch              {torch.__version__}",
        f"torch.version.cuda {torch.version.cuda}",
        f"triton             {triton.__version__}",
        f"python             {sys.version.split()[0]}",
        f"commit_sha         {commit_sha()}",
        f"matmul_allow_tf32  {torch.backends.cuda.matmul.allow_tf32}",
        f"cudnn_allow_tf32   {torch.backends.cudnn.allow_tf32}",
    ])


def _run(command: list[str]) -> str:
    """Never raises. A missing binary is a result to report here (the ncu probe
    exists precisely to find out whether one is usable), not a crash that loses
    every other line of output the caller had already gathered."""
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                cwd=os.getcwd())
    except OSError as exc:
        return f"$ {' '.join(command)}\n[not runnable] {exc}\n"
    return (f"$ {' '.join(command)}\n[exit {result.returncode}]\n"
            f"{result.stdout}\n{result.stderr}\n")


def _ncu_candidates() -> list[str]:
    import glob
    return ["ncu"] + sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu")
                            + glob.glob("/usr/local/cuda*/bin/ncu"))


def smoke_body() -> str:
    return (_describe_device() + "\n\n"
            + _run([sys.executable, "-m", "pytest", "-q", "tests/test_gelu.py"]))


def tests_body(extra_args: tuple[str, ...] = ()) -> str:
    """Streams pytest's output to the container's stdout instead of capturing
    it. On a metered GPU, a suite that has produced no output for twenty
    minutes is indistinguishable from a hung one unless you can see which test
    it is sitting in -- and by the time a captured run returns, the money is
    already spent."""
    command = [sys.executable, "-m", "pytest", "-q", "--durations=15",
               *extra_args]
    result = subprocess.run(command, text=True)
    return f"$ {' '.join(command)}\n[exit {result.returncode}]\n"


def _compiled_mlp_fused_kernel(block_m: int, block_h: int, num_warps: int):
    """Compile `_mlp_fused_kernel` at the committed `mlp_fused` configuration
    and hand back Triton's compiled-kernel object, whose `n_regs` / `n_spills` /
    `metadata.shared` are what prediction 2 is scored on.

    Calls the kernel directly rather than through `mlp_fused`, because the
    registry wrapper returns the output tensor and discards the kernel handle.
    """
    import torch
    import triton
    from model.kernels.mlp import _mlp_fused_kernel

    seq, dim, hidden = 64, 192, 768
    x = torch.randn(8 * seq, dim, device="cuda")
    w1 = torch.randn(hidden, dim, device="cuda") * 0.05
    b1 = torch.randn(hidden, device="cuda")
    w2 = torch.randn(dim, hidden, device="cuda") * 0.05
    b2 = torch.randn(dim, device="cuda")
    out = torch.empty_like(x)
    m = x.shape[0]
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),)
    return _mlp_fused_kernel[grid](
        x, w1, b1, w2, b2, out, m, dim, hidden,
        x.stride(0), x.stride(1), w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_D=triton.next_power_of_2(dim), BLOCK_H=block_h,
        num_warps=num_warps, num_stages=1)


def registers_body() -> str:
    """Prediction 2: the register arithmetic behind `mlp_fused`'s 1 block/SM.

    Reports blocks/SM and occupancy% as two separate lines on purpose. The
    block count depends only on the register file (65,536 regs/SM on both
    cards) and is expected to transfer; the percentage divides by max resident
    warps/SM, which is 32 on sm_75 and 48 on sm_89, and is not expected to.
    """
    import torch
    props = torch.cuda.get_device_properties(0)
    capability = (props.major, props.minor)
    max_warps = props.max_threads_per_multi_processor // props.warp_size
    regs_per_sm = props.regs_per_multiprocessor
    shared_per_sm = props.shared_memory_per_multiprocessor
    lines = [f"compute_capability     {capability}",
             f"max_warps_per_sm       {max_warps} "
             f"(pre-registered expectation "
             f"{EXPECTED_MAX_WARPS_PER_SM.get(capability)})",
             f"regs_per_sm            {regs_per_sm}",
             f"shared_per_sm_B        {shared_per_sm}",
             f"shared_per_block_optin_B {props.shared_memory_per_block_optin}"]

    for block_m, block_h, num_warps in [(16, 32, 8), (2, 32, 8), (16, 16, 8)]:
        kernel = _compiled_mlp_fused_kernel(block_m, block_h, num_warps)
        threads = num_warps * props.warp_size
        regs_per_block = kernel.n_regs * threads
        blocks_by_regs = regs_per_sm // regs_per_block
        shared = kernel.metadata.shared
        blocks_by_shared = shared_per_sm // shared if shared else blocks_by_regs
        blocks = min(blocks_by_regs, blocks_by_shared,
                     max_warps // num_warps)
        lines.append(
            f"BLOCK_M={block_m} BLOCK_H={block_h} num_warps={num_warps}: "
            f"n_regs={kernel.n_regs} n_spills={kernel.n_spills} "
            f"shared_B={shared} threads={threads} "
            f"regs_per_block={regs_per_block} "
            f"blocks_per_sm_by_regs={blocks_by_regs} "
            f"blocks_per_sm_by_shared={blocks_by_shared} "
            f"blocks_per_sm={blocks} "
            f"occupancy_pct={blocks * num_warps / max_warps * 100:.2f}")
    return "\n".join(lines)


def _matmul_arms(batch: int):
    import torch
    from model.baseline.layers import linear as linear_torch
    from model.kernels.linear import linear as linear_triton
    from model.kernels.linear import linear_tuned

    seq = 64
    shapes = [(192, 576), (192, 192), (192, 768), (768, 192)]
    arms = {}
    for k, n in shapes:
        x = torch.randn(batch, seq, k, device="cuda")
        w = torch.randn(n, k, device="cuda") * 0.05
        b = torch.randn(n, device="cuda")
        arms[(k, n)] = {
            "torch": lambda x=x, w=w, b=b: linear_torch(x, w, b),
            "triton": lambda x=x, w=w, b=b: linear_triton(x, w, b),
            "triton_tuned": lambda x=x, w=w, b=b: linear_tuned(x, w, b),
        }
    return arms


def tf32_body(batch: int = 128) -> str:
    """Prediction 3: the cuBLAS-vs-`tl.dot` matmul comparison, measured under
    both precision policies.

    Deliberately does NOT write to latency.csv. The CSV's `dtype` column would
    read `float32` for both policies with nothing to tell them apart, and
    adding a column to disambiguate would break `record()`'s header contract
    for every existing row. These numbers live in the task report instead.

    Each flag is read back after being set: torch 2.11 routes `allow_tf32`
    through the newer fp32-precision API, so a write that silently does not
    take effect would otherwise be invisible.
    """
    import torch
    import triton
    lines = []
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        readback = (torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32)
        if readback != (tf32, tf32):
            return (f"ABORT: tf32 flags did not take effect; "
                    f"requested {tf32}, read back {readback}")
        lines.append(f"--- allow_tf32={tf32} (readback matmul={readback[0]} "
                     f"cudnn={readback[1]}) batch={batch}")
        for (k, n), arms in _matmul_arms(batch).items():
            timings = {name: triton.testing.do_bench(fn)
                       for name, fn in arms.items()}
            flops = 2 * batch * 64 * n * k
            detail = "  ".join(
                f"{name}={ms:.4f}ms/{flops / (ms * 1e-3) / 1e12:.3f}TF"
                for name, ms in timings.items())
            lines.append(f"k={k:>3} n={n:>3}: {detail}  "
                         f"torch/triton={timings['triton'] / timings['torch']:.2f}x")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return "\n".join(lines)


def sdpa_body(batch: int = 128) -> str:
    """The SDPA-backend observation (pre-registered as an observation, not a
    scored prediction). Reads the kernel name out of torch.profiler rather than
    inferring it from the backend-selection flags, matching how
    docs/findings/04-flash-attention.md identified
    `fmha_cutlassF_f32_aligned_64x64_rf_sm75` on the 1650 Ti.
    """
    import torch
    import torch.nn.functional as F
    import triton
    from model.kernels.attention import attention_composed, attention_flash

    heads, seq, head_dim = 3, 64, 64
    scale = head_dim ** -0.5
    q, k, v = (torch.randn(batch, heads, seq, head_dim, device="cuda")
               for _ in range(3))

    sdpa = lambda: F.scaled_dot_product_attention(q, k, v, scale=scale)
    for _ in range(5):
        sdpa()
    torch.cuda.synchronize()

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        sdpa()
        torch.cuda.synchronize()
    launched = sorted({event.key for event in prof.key_averages()
                       if event.device_type == torch.autograd.DeviceType.CUDA
                       and event.device_time_total > 0})

    timings = {
        "sdpa": triton.testing.do_bench(sdpa),
        "attention_flash": triton.testing.do_bench(
            lambda: attention_flash(q, k, v, scale)),
        "attention_composed": triton.testing.do_bench(
            lambda: attention_composed(q, k, v, scale)),
    }
    lines = [f"batch={batch} heads={heads} seq={seq} head_dim={head_dim} fp32",
             "sdpa_cuda_kernels: " + ", ".join(launched)]
    lines += [f"{name:>19} {ms:.4f} ms" for name, ms in timings.items()]
    lines.append(f"flash/sdpa = {timings['attention_flash'] / timings['sdpa']:.2f}x")
    return "\n".join(lines)


def monolithic_body() -> str:
    """Prediction 4: the monolithic single-kernel block still fails to compile.

    Reconstruction of the uncommitted scratch prototype described in
    docs/findings/05-over-fusion.md (one program holding attention's Q/K/V and
    score tiles, the output-projection weight tile, and the MLP's w1/w2 tiles
    at once). It is numerically meaningless -- the prototype was too, since the
    only question either answers is whether those tiles can be simultaneously
    resident.

    Not a byte-exact restoration, and the difference is stated rather than
    buried: the original prototype was never committed, and this reconstruction
    requires **278,528 B** on sm_75 where the finding recorded 262,144 B. The
    dominant term is identical and is what drives both figures -- the
    [BLOCK_D, BLOCK_D] = 256 x 256 x 4 = 262,144 B output-projection weight
    tile -- with this version additionally holding a 16,384 B attention tile
    live alongside it. Both are ~4x Turing's 65,536 B budget and invariant in
    BLOCK_H, so this probes the same wall; the L4 number should be read as
    "this reconstruction's requirement", not as the original prototype's.

    The interesting datum is the hardware limit the toolchain reports, not the
    failure: prediction 4 says the bound moves (65,536 B -> ~99 KB) without
    disappearing.
    """
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _monolithic_block_kernel(x_ptr, qkv_w_ptr, proj_w_ptr, w1_ptr, w2_ptr,
                                 out_ptr, M, D, H,
                                 BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
                                 BLOCK_H: tl.constexpr, SEQ: tl.constexpr):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_s = tl.arange(0, SEQ)

        x = tl.load(x_ptr + offs_m[:, None] * D + offs_d[None, :],
                    mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
                    other=0.0)
        qkv_w = tl.load(qkv_w_ptr + offs_d[:, None] * D + offs_d[None, :],
                        mask=(offs_d[:, None] < D) & (offs_d[None, :] < D),
                        other=0.0)
        q = tl.dot(x, qkv_w)

        k_tile = tl.load(x_ptr + offs_s[:, None] * D + offs_d[None, :],
                         mask=offs_d[None, :] < D, other=0.0)
        scores = tl.dot(q, tl.trans(k_tile))
        scores = scores - tl.max(scores, axis=1)[:, None]
        probs = tl.exp(scores)
        probs = probs / tl.sum(probs, axis=1)[:, None]

        v_tile = tl.load(x_ptr + offs_s[:, None] * D + offs_d[None, :],
                         mask=offs_d[None, :] < D, other=0.0)
        attended = tl.dot(probs.to(v_tile.dtype), v_tile)

        proj_w = tl.load(proj_w_ptr + offs_d[:, None] * D + offs_d[None, :],
                         mask=(offs_d[:, None] < D) & (offs_d[None, :] < D),
                         other=0.0)
        hidden_in = tl.dot(attended, proj_w)

        out = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for h_start in range(0, H, BLOCK_H):
            offs_h = h_start + tl.arange(0, BLOCK_H)
            w1 = tl.load(w1_ptr + offs_h[None, :] * D + offs_d[:, None],
                         mask=(offs_h[None, :] < H) & (offs_d[:, None] < D),
                         other=0.0)
            act = tl.dot(hidden_in, w1)
            act = tl.where(act > 0, act, 0.0)
            w2 = tl.load(w2_ptr + offs_d[None, :] * H + offs_h[:, None],
                         mask=(offs_d[None, :] < D) & (offs_h[:, None] < H),
                         other=0.0)
            out += tl.dot(act, w2)

        tl.store(out_ptr + offs_m[:, None] * D + offs_d[None, :], out,
                 mask=(offs_m[:, None] < M) & (offs_d[None, :] < D))

    seq, dim, hidden = 64, 192, 768
    m = 8 * seq
    x = torch.randn(m, dim, device="cuda")
    qkv_w = torch.randn(dim, 3 * dim, device="cuda")
    proj_w = torch.randn(dim, dim, device="cuda")
    w1 = torch.randn(hidden, dim, device="cuda")
    w2 = torch.randn(dim, hidden, device="cuda")
    out = torch.empty_like(x)

    props = torch.cuda.get_device_properties(0)
    lines = [f"device {torch.cuda.get_device_name(0)} "
             f"cc={torch.cuda.get_device_capability(0)}",
             f"props.shared_memory_per_block={props.shared_memory_per_block} B",
             f"props.shared_memory_per_block_optin="
             f"{getattr(props, 'shared_memory_per_block_optin', 'n/a')} B"]
    for block_h in (32, 16):
        try:
            _monolithic_block_kernel[(triton.cdiv(m, 16),)](
                x, qkv_w, proj_w, w1, w2, out, m, dim, hidden,
                BLOCK_M=16, BLOCK_D=triton.next_power_of_2(dim),
                BLOCK_H=block_h, SEQ=seq,
                num_warps=8, num_stages=1)
            torch.cuda.synchronize()
            lines.append(f"BLOCK_H={block_h}: COMPILED AND RAN")
        except Exception as exc:
            lines.append(f"BLOCK_H={block_h}: {type(exc).__name__}: {exc}")
    return "\n".join(lines)


def counters_body() -> str:
    """The ncu permission probe (prediction 1's only admissible instrument).

    Runs once, early, and on the smallest grid, because the expected outcome is
    a permission refusal (`ERR_NVGPUCTRPERM`) that costs seconds to discover
    and would cost minutes if discovered inside the counter grid. The exact
    error text is the deliverable either way: per the controller's ruling,
    prediction 1 is recorded UNRESOLVED if counters are unavailable rather than
    re-scored against weaker evidence.
    """
    lines = []
    candidates = _ncu_candidates()
    lines.append(f"ncu candidates: {candidates}")
    ncu = None
    for candidate in candidates:
        probe = _run([candidate, "--version"])
        lines.append(probe)
        if "[exit 0]" in probe:
            ncu = candidate
            break
    if ncu is None:
        lines.append("VERDICT: ncu not present in image; counters unavailable.")
        return "\n".join(lines)

    if os.path.dirname(ncu):
        os.environ["PATH"] = f"{os.path.dirname(ncu)}:{os.environ['PATH']}"
    # ncu's default clock-locking is refused on Modal's shared GPU hosts (the
    # first L4 probe failed on exactly that, not on counter permission), so the
    # probe retries under the waiver documented in bench/profile.py.
    os.environ.setdefault("TRITONFORMER_NCU_CLOCK_CONTROL", "none")
    from bench.profile import profile_kernel
    try:
        rows = profile_kernel("bench.run_mlp", "mlp", "triton_fused", 1,
                              "float32")
        lines.append(f"VERDICT: counters AVAILABLE; {len(rows)} metric rows")
        lines += [f"  {row['kernel_name']} {row['metric']} = {row['value']}"
                  for row in rows]
    except Exception as exc:
        lines.append(f"VERDICT: counters UNAVAILABLE\n{type(exc).__name__}: {exc}")
    return "\n".join(lines)


def sweep_body() -> tuple[str, str]:
    """Runs every latency runner, then hands back the container-local
    latency.csv verbatim for the caller to merge. Returns (log, csv_text)."""
    log = []
    for module in SWEEP_MODULES:
        log.append(_run([sys.executable, "-m", module]))
    return "\n".join(log), _read(LATENCY_CSV)


def counter_grid_body() -> tuple[str, str]:
    """The full counter grid, run only if the probe says counters work.

    Reuses bench/collect_counters.py rather than defining a new grid here, so
    the L4 rows are the same measurement as the sm_75 rows already in
    counters.csv (same batch, same variants, same launch-window logic) and are
    therefore comparable. Returns (log, csv_text).
    """
    os.environ.setdefault("TRITONFORMER_NCU_CLOCK_CONTROL", "none")
    ncu = next((c for c in _ncu_candidates() if os.path.dirname(c)
                and os.path.exists(c)), None)
    if ncu:
        os.environ["PATH"] = f"{os.path.dirname(ncu)}:{os.environ['PATH']}"
    log = _run([sys.executable, "-m", "bench.collect_counters"])
    return log, _read(COUNTERS_CSV)


def _read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path) as handle:
        return handle.read()


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    base_image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("git", "build-essential")
        .pip_install("torch==2.11.0+cu128", "torchvision==0.26.0+cu128",
                     extra_index_url=TORCH_INDEX)
        .pip_install("triton==3.6.0", "nvidia-ml-py", "pytest", "numpy"))

    # ncu is not on PyPI and ships in no torch wheel, so the counter probe needs
    # its own image. Installed BELOW the source layer, not above it: layers are
    # invalidated from the first change downward, and putting a ~1 GB Nsight
    # Compute install on top of the source would rebuild it on every edit.
    ncu_base_image = base_image.run_commands(
        "apt-get update && apt-get install -y --no-install-recommends "
        "wget ca-certificates",
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/"
        "debian12/x86_64/cuda-keyring_1.1-1_all.deb "
        "&& dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb",
        "apt-get update && apt-get install -y --no-install-recommends "
        "cuda-nsight-compute-12-8",
    )

    def _with_source(img):
        """copy=True bakes the source into an image layer instead of mounting
        it at runtime: the benchmark runners write bench/results/latency.csv and
        the end-to-end test downloads CIFAR-10 into data/, both inside the
        source tree, which a read-only runtime mount would not allow."""
        return img.add_local_dir(
            ".", remote_path=REMOTE_ROOT, copy=True,
            ignore=["**/.venv/**", "**/.venv", "**/__pycache__/**",
                    "**/*.pyc", "bench/results/**",
                    # data/cifar-10-python.tar.gz IS included, deliberately:
                    # tests/test_end_to_end.py builds CIFAR10(download=True),
                    # and fetching 170 MB from the upstream mirror inside the
                    # container stalled for minutes of billed GPU time. Shipping
                    # the archive (already in the repo) makes the dataset a
                    # build-time cost on a CPU builder instead. The extracted
                    # directory stays excluded -- torchvision unpacks the
                    # archive itself, so shipping both would just be duplication.
                    "data/cifar-10-batches-py/**",
                    "**/.pytest_cache/**", "**/.ruff_cache/**"]).workdir(REMOTE_ROOT)

    image = _with_source(base_image)
    ncu_image = _with_source(ncu_base_image)

    app = modal.App("tritonformer-l4")

    def _echo(text: str) -> str:
        """Container stdout streams to the CLI live, where a returned string
        only appears once the function finishes -- which is no use for watching
        a multi-minute sweep, or for seeing how far a failed one got."""
        print(text, flush=True)
        return text

    @app.function(image=image, timeout=900)
    def build_check() -> str:
        """CPU-only. Forces the image to build and proves the source tree,
        imports, and pinned versions are right without starting an L4 at all --
        every error caught here is an error not paid for in GPU seconds."""
        return _echo(_run([sys.executable, "-c",
                           "import torch, triton, torchvision, pytest, os; "
                           "print(torch.__version__, triton.__version__, "
                           "torchvision.__version__); "
                           "print(sorted(os.listdir('.')))"]))

    @app.function(image=ncu_image, timeout=900)
    def build_check_ncu() -> str:
        """CPU-only. `ncu --version` needs no GPU, so whether Nsight Compute
        installed at all is answerable before the permission probe spends L4
        time finding out."""
        candidates = _ncu_candidates()
        return _echo(f"candidates: {candidates}\n"
                     + "".join(_run([c, "--version"]) for c in candidates))

    @app.function(image=image, gpu="L4", timeout=600)
    def smoke() -> str:
        return _echo(smoke_body())

    @app.function(image=image, gpu="L4", timeout=2400)
    def tests() -> str:
        return _echo(tests_body())

    @app.function(image=ncu_image, gpu="L4", timeout=900)
    def probe_counters() -> str:
        return _echo(counters_body())

    @app.function(image=ncu_image, gpu="L4", timeout=2400)
    def counter_grid() -> tuple[str, str]:
        log, csv_text = counter_grid_body()
        return _echo(log), csv_text

    @app.function(image=image, gpu="L4", timeout=1800)
    def predictions() -> str:
        sections = [("device", _describe_device),
                    ("prediction2_registers", registers_body),
                    ("prediction4_monolithic", monolithic_body),
                    ("prediction3_tf32", tf32_body),
                    ("sdpa_backend", sdpa_body)]
        out = []
        for name, body in sections:
            try:
                out.append(f"===== {name}\n{body()}")
            except Exception as exc:
                out.append(f"===== {name}\nFAILED {type(exc).__name__}: {exc}")
        return _echo("\n\n".join(out))

    @app.function(image=image, gpu="L4", timeout=3600)
    def sweep() -> tuple[str, str]:
        log, csv_text = sweep_body()
        return _echo(log), csv_text

    @app.local_entrypoint()
    def run(step: str, out_dir: str) -> None:
        """One step per invocation, results written to out_dir.

        `modal run` discards a function's return value, so the CSVs a sweep
        produces would be lost without somewhere local to put them. Taking a
        single `step` rather than running everything is the budget discipline
        the module docstring describes: each L4 entrypoint is launched
        deliberately, after the cheaper one before it came back clean.
        """
        steps = {"build_check": build_check, "build_check_ncu": build_check_ncu,
                 "smoke": smoke, "tests": tests,
                 "probe_counters": probe_counters, "predictions": predictions,
                 "sweep": sweep, "counter_grid": counter_grid}
        os.makedirs(out_dir, exist_ok=True)
        started = time.monotonic()
        result = steps[step].remote()
        elapsed = time.monotonic() - started
        log, csv_text = result if isinstance(result, tuple) else (result, None)
        with open(os.path.join(out_dir, f"{step}.log"), "w") as handle:
            handle.write(log)
        if csv_text is not None:
            with open(os.path.join(out_dir, f"{step}.csv"), "w") as handle:
                handle.write(csv_text)
        print(f"[{step}] {elapsed:.1f}s wall (upper bound on billed GPU "
              f"seconds: includes queueing, container start and image pull)")
        with open(os.path.join(out_dir, "elapsed.txt"), "a") as handle:
            handle.write(f"{step} {elapsed:.1f}\n")
        print(f"[{step}] wrote {out_dir}/{step}.log"
              + (f" and {step}.csv ({len(csv_text.splitlines())} lines)"
                 if csv_text is not None else ""))


def merge_csv(remote_text: str, path: str) -> str:
    """Append the container's rows to the repo's CSV, refusing on any schema
    drift.

    `bench/harness.record()` writes a header only when the file is new, so the
    container always produces one and the repo file must never gain a second.
    The header equality check is byte-for-byte and the field count is checked
    per row: a silently reordered or renamed column would make every prior row
    unreadable against the new ones, which is a worse outcome than this
    function refusing to merge.
    """
    remote_lines = remote_text.strip().splitlines()
    if not remote_lines:
        return f"{path}: nothing returned, nothing merged"
    with open(path) as handle:
        local_lines = handle.read().splitlines()
    if remote_lines[0] != local_lines[0]:
        raise SystemExit(f"{path}: header mismatch, refusing to merge\n"
                         f"  local:  {local_lines[0]}\n"
                         f"  remote: {remote_lines[0]}")
    expected = len(local_lines[0].split(","))
    rows = remote_lines[1:]
    bad = [row for row in rows if len(row.split(",")) != expected]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} row(s) have the wrong field "
                         f"count; first: {bad[0]}")
    with open(path, "a") as handle:
        handle.write("\n".join(rows) + "\n")
    return (f"{path}: {len(local_lines)} lines -> "
            f"{len(local_lines) + len(rows)} lines ({len(rows)} appended)")


def _local_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a measurement body against the local GPU (for "
                    "debugging before spending L4 seconds), or merge a CSV "
                    "returned by an L4 run into bench/results/.")
    parser.add_argument("body", choices=["device", "smoke", "registers",
                                         "tf32", "sdpa", "monolithic",
                                         "counters", "merge"])
    parser.add_argument("--csv", help="file holding the returned CSV text")
    parser.add_argument("--into", default=LATENCY_CSV)
    args = parser.parse_args()
    if args.body == "merge":
        with open(args.csv) as handle:
            print(merge_csv(handle.read(), args.into))
        return
    bodies = {"device": _describe_device, "smoke": smoke_body,
              "registers": registers_body, "tf32": tf32_body,
              "sdpa": sdpa_body, "monolithic": monolithic_body,
              "counters": counters_body}
    print(bodies[args.body]())


if __name__ == "__main__":
    _local_main()
