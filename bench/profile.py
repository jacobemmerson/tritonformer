"""Hardware counter collection via Nsight Compute.

ncu serializes execution and replays each kernel to gather counter sets,
so its reported durations are inflated and meaningless as performance
numbers. Latency lives in bench/harness.py and a separate CSV; nothing
here produces a timing.

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
            return [row for row in reader if row.get("Metric Name")]
    return []


def profile_kernel(module: str, kernel: str, variant: str, batch: int,
                   dtype: str, launch_skip: int = 5,
                   launch_count: int = 1) -> list[dict]:
    """Profile a single steady-state launch.

    launch_skip avoids the cold first launch, whose counters reflect
    autotuning and cache-cold behaviour rather than steady state.
    """
    command = [
        "ncu", "--csv", "--target-processes", "all",
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
    return [{
        "timestamp": stamp, "commit_sha": sha, "gpu": gpu,
        "kernel": kernel, "variant": variant, "batch": batch, "dtype": dtype,
        "kernel_name": row["Kernel Name"], "metric": row["Metric Name"],
        "unit": row.get("Metric Unit", ""), "value": row["Metric Value"],
    } for row in parse_ncu_csv(result.stdout)]


def record_counters(rows: list[dict], path: str) -> None:
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COUNTER_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
