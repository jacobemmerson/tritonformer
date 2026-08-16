import pytest

from bench.profile import METRICS, parse_ncu_csv, profile_kernel

TWO_KERNEL_SAMPLE = (
    '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    '"0","torch_add_kernel","dram__bytes_read.sum","byte","1000"\n'
    '"0","torch_add_kernel","launch__registers_per_thread","register/thread","32"\n'
    '"1","_layernorm_kernel","dram__bytes_read.sum","byte","2000"\n'
    '"1","_layernorm_kernel","launch__registers_per_thread","register/thread","28"\n'
)

ONE_KERNEL_SAMPLE = (
    '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    '"0","_layernorm_kernel","dram__bytes_read.sum","byte","2000"\n'
    '"0","_layernorm_kernel","launch__registers_per_thread","register/thread","28"\n'
)


class FakeCompletedProcess:
    def __init__(self, stdout: str):
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


def _patch_subprocess(monkeypatch, stdout: str):
    monkeypatch.setattr(
        "bench.profile.subprocess.run",
        lambda *args, **kwargs: FakeCompletedProcess(stdout))

SAMPLE = '''==PROF== Connected to process 1234
==PROF== Profiling "layernorm_kernel" - 0: 0%....50%....100%
"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","layernorm_kernel","dram__bytes_read.sum","byte","98304"
"0","layernorm_kernel","launch__registers_per_thread","register/thread","32"
==PROF== Disconnected from process 1234
'''


def test_parse_skips_preamble():
    rows = parse_ncu_csv(SAMPLE)
    assert len(rows) == 2
    assert rows[0]["Metric Name"] == "dram__bytes_read.sum"
    assert rows[0]["Metric Value"] == "98304"


def test_parse_returns_empty_for_no_header():
    assert parse_ncu_csv("==PROF== nothing here\n") == []


def test_parse_strips_thousands_separators():
    sample = (
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","layernorm_kernel","dram__bytes_read.sum","byte","25,184,832"\n'
    )
    rows = parse_ncu_csv(sample)
    assert rows[0]["Metric Value"] == "25184832"


def test_metric_set_includes_spill_counters():
    assert "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum" in METRICS
    assert "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum" in METRICS
    assert "sm__warps_active.avg.pct_of_peak_sustained_active" in METRICS
    assert "launch__registers_per_thread" in METRICS


def test_profile_kernel_accepts_satisfied_expected_kernels(monkeypatch):
    _patch_subprocess(monkeypatch, TWO_KERNEL_SAMPLE)
    rows = profile_kernel("bench.run_x", "x", "triton", 128, "float32",
                          launch_count=2, expected_kernels=2)
    assert {row["kernel_name"] for row in rows} == {
        "torch_add_kernel", "_layernorm_kernel"}


def test_profile_kernel_raises_on_undercounted_capture(monkeypatch):
    _patch_subprocess(monkeypatch, ONE_KERNEL_SAMPLE)
    with pytest.raises(RuntimeError, match="expected 2.*captured 1"):
        profile_kernel("bench.run_x", "x", "triton", 128, "float32",
                       launch_count=1, expected_kernels=2)


def test_profile_kernel_without_expected_kernels_is_unchanged(monkeypatch):
    _patch_subprocess(monkeypatch, ONE_KERNEL_SAMPLE)
    rows = profile_kernel("bench.run_x", "x", "triton", 128, "float32")
    assert len(rows) == 2
