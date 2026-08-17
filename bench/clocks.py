import subprocess


def _query(field: str) -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return int(out.stdout.strip().splitlines()[0])
    except ValueError:
        # Unlocked/unsupported readings come back as "[N/A]".
        return None


def telemetry() -> tuple[int, int]:
    return _query("clocks.sm") or -1, _query("temperature.gpu") or -1


def locked_clock_mhz() -> int | None:
    """Applications clock, set by `nvidia-smi -lgc`. None if unlocked."""
    return _query("clocks.applications.graphics")
