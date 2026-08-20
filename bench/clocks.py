import os
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
    """The operator-declared `nvidia-smi -lgc` target, in MHz.

    GeForce cards have no queryable read-back of an `-lgc` lock: the only
    clock-lock field nvidia-smi exposes is `clocks.applications.graphics`,
    which is a different mechanism (`-ac`, application clocks) this card
    does not support and reports as "[N/A]". There is no way to
    auto-detect the lock, so the target must be declared explicitly via
    TRITONFORMER_LOCKED_CLOCK_MHZ; unset means "no lock declared" (None),
    not "unlocked" -- the two are indistinguishable from here.
    """
    value = os.environ.get("TRITONFORMER_LOCKED_CLOCK_MHZ")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
