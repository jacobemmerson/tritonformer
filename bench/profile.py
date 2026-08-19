"""Hardware counter collection via Nsight Compute.

ncu serializes execution and replays each kernel to gather counter sets,
so its reported durations are inflated and meaningless as performance
numbers. Latency lives in bench/harness.py and a separate CSV; nothing
here produces a timing.

By default ncu fixes the GPU's clocks for the duration of a capture, so that
counters gathered across its replay passes describe one consistent operating
point. A host that forbids clock control fails the whole capture with
"Failed to lock GPU clock frequencies!" -- observed on Modal's shared L4, where
counter *permission* was granted but clock control was not. Setting
TRITONFORMER_NCU_CLOCK_CONTROL=none passes `--clock-control none`, which lets
such a host produce counters at whatever frequency it happens to run.

Unset means unchanged behaviour, which is what every measurement before this
was taken with, and unset is the right choice wherever clock control works: the
waiver is not free. Counts of bytes, sectors, and warps are frequency-invariant
and stay comparable; anything rate- or duration-shaped (and any comparison
against a locked-clock run) is not. Rows gathered under the waiver must say so.

Setup, once, on the profiling host:
    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \\
        | sudo tee /etc/modprobe.d/nvidia-profiling.conf
    sudo update-initramfs -u && sudo reboot
"""
import csv
import io
import os
import subprocess
import sys
from datetime import datetime, timezone

from bench.harness import commit_sha, gpu_name

METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
]

COUNTER_COLUMNS = [
    "timestamp", "commit_sha", "gpu", "kernel", "variant", "batch", "dtype",
    "kernel_name", "metric", "unit", "value",
]


def parse_ncu_csv(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith('"ID"'):
            reader = csv.DictReader(io.StringIO("\n".join(lines[index:])))
            rows = [row for row in reader if row.get("Metric Name")]
            for row in rows:
                if "Metric Value" in row and row["Metric Value"] is not None:
                    row["Metric Value"] = row["Metric Value"].replace(",", "")
            return rows
    return []


def base_kernel_name(name: str) -> str:
    """The bare kernel identifier, with return type, namespaces, template
    arguments and parameter list stripped.

    Exists so a capture can be checked against what torch.profiler predicted:
    the two spell the same kernel differently (torch says
    `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<...>>(...)`,
    ncu says `void vectorized_elementwise_kernel<4, AUnaryFunctor<...>>(...)`),
    so only the identifier is comparable.
    """
    name = name.split("(")[0].split("<")[0].strip()
    if name.startswith("void "):
        name = name[len("void "):]
    return name.strip().split("::")[-1]


def profile_kernel(module: str, kernel: str, variant: str, batch: int,
                   dtype: str, launch_skip: int = 5,
                   launch_count: int = 1,
                   expected_kernels: int | None = None,
                   expected_kernel_names: set[str] | None = None) -> list[dict]:
    """Profile a single steady-state launch.

    launch_skip avoids the cold first launch, whose counters reflect
    autotuning and cache-cold behaviour rather than steady state.

    expected_kernel_names checks kernel IDENTITY, and is the check that matters.
    expected_kernels counts launches, which cannot detect a window that landed
    on the wrong kernels entirely: a fused arm launching one kernel and a
    benchmark's own setup `randn`/`* 0.05` elementwise launch both satisfy
    "exactly 1 distinct kernel". That is not hypothetical -- it is precisely how
    an earlier L4 counter grid recorded a scalar-multiply setup kernel as if it
    were `_mlp_fused_kernel`. Pass the base names the arm is expected to launch
    and a misaimed capture window fails loudly instead of returning plausible
    numbers for the wrong kernel.

    An arm can launch more than one distinct kernel per call (e.g. a
    composed "separate op then op" arm), and ncu's --launch-count only
    captures however many launches you ask for -- there is no built-in
    signal that a capture window fell short and silently missed one of
    them. expected_kernels lets the caller declare how many distinct
    kernel_name values this arm should produce; a mismatch raises rather
    than returning a partial, undercounted capture. Declaring a count is
    simpler for a caller to get right than enumerating name substrings --
    it doesn't require the caller to predict ncu's kernel-name mangling
    (e.g. torch's templated elementwise kernel names).
    """
    clock_control = os.environ.get("TRITONFORMER_NCU_CLOCK_CONTROL")
    command = [
        "ncu", "--csv", "--target-processes", "all",
        *(["--clock-control", clock_control] if clock_control else []),
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
        "--metrics", ",".join(METRICS),
        sys.executable, "-m", module,
        "--kernel", kernel, "--variant", variant,
        "--batch", str(batch), "--dtype", dtype,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ncu failed ({result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    stamp = datetime.now(timezone.utc).isoformat()
    sha, gpu = commit_sha(), gpu_name()
    rows = [{
        "timestamp": stamp, "commit_sha": sha, "gpu": gpu,
        "kernel": kernel, "variant": variant, "batch": batch, "dtype": dtype,
        "kernel_name": row["Kernel Name"], "metric": row["Metric Name"],
        "unit": row.get("Metric Unit", ""), "value": row["Metric Value"],
    } for row in parse_ncu_csv(result.stdout)]

    if expected_kernel_names is not None:
        captured = {base_kernel_name(row["kernel_name"]) for row in rows}
        expected = {base_kernel_name(name) for name in expected_kernel_names}
        if captured != expected:
            raise RuntimeError(
                f"profile_kernel({kernel!r}, {variant!r}) captured the wrong "
                f"kernel(s). expected {sorted(expected)}, captured "
                f"{sorted(captured)}. The launch window did not land on the "
                f"arm under test -- check launch_skip against the benchmark's "
                f"own setup launches.")

    if expected_kernels is not None:
        captured = sorted(set(row["kernel_name"] for row in rows))
        if len(captured) != expected_kernels:
            raise RuntimeError(
                f"profile_kernel({kernel!r}, {variant!r}) expected "
                f"{expected_kernels} distinct kernel launch(es) but "
                f"captured {len(captured)}: {captured}. A silent partial "
                f"capture would undercount this arm's traffic; raise "
                f"launch_count or fix the arm.")

    return rows


def record_counters(rows: list[dict], path: str) -> None:
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COUNTER_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
