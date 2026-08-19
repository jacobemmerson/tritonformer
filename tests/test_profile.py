import pytest

from bench.profile import (METRICS, base_kernel_name, parse_ncu_csv,
                           profile_kernel)

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


# Real kernel names captured from actual runs and recorded in
# bench/results/counters.csv -- ncu's spelling and torch.profiler's spelling of
# the same kernels, plus the two functors that must not be confused. Invented
# names would not exercise the differences that matter here: ncu writes non-type
# template arguments as `(int)4` and drops namespaces that torch keeps.
NCU_ADD = (
    "void at::native::vectorized_elementwise_kernel<(int)4, "
    "at::native::CUDAFunctor_add<float>, "
    "std::array<char *, (unsigned long)3>>(int, T2, T3)")
TORCH_ADD = (
    "void at::native::vectorized_elementwise_kernel<4, "
    "at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >"
    "(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)")
NCU_MUL = (
    "void at::native::vectorized_elementwise_kernel<(int)4, "
    "at::native::AUnaryFunctor<float, float, float, "
    "at::native::binary_internal::MulFunctor<float>>, "
    "std::array<char *, (unsigned long)2>>(int, T2, T3)")
NCU_MUL_NO_NAMESPACES = (
    "void vectorized_elementwise_kernel<4, "
    "AUnaryFunctor<float, float, float, MulFunctor<float>>, "
    "array<char *, 2>>(int, T2, T3)")
NCU_GELU_LAMBDA = (
    "void at::native::vectorized_elementwise_kernel<(int)4, "
    "at::native::GeluCUDAKernelImpl(at::TensorIteratorBase &, "
    "at::native::GeluType)::[lambda() (instance 1)]::operator ()() const::"
    "[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], "
    "std::array<char *, (unsigned long)2>>(int, T2, T3)")


def _sample(*kernel_names: str) -> str:
    """An ncu --csv capture of one row per named kernel. Names are quoted
    because real ones are full of commas."""
    header = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    return header + "".join(
        f'"{index}","{name}","dram__bytes_read.sum","byte","1000"\n'
        for index, name in enumerate(kernel_names))


def test_base_kernel_name_survives_ncu_non_type_template_arguments():
    """ncu writes `<(int)4, ...>`, so parsing that splits on the first "("
    truncates the name before its functor arguments while leaving torch's
    spelling intact -- silently erasing the discriminator. Both spellings of
    the same kernel must reduce to the same identity."""
    assert base_kernel_name(NCU_ADD) == base_kernel_name(TORCH_ADD)
    assert base_kernel_name(NCU_ADD) == (
        "vectorized_elementwise_kernel<CUDAFunctor_add>")


def test_base_kernel_name_normalises_namespaces():
    assert base_kernel_name(NCU_MUL) == base_kernel_name(NCU_MUL_NO_NAMESPACES)


def test_base_kernel_name_distinguishes_functors():
    """The whole point of keeping functor names: two instantiations of the same
    templated elementwise kernel are different kernels."""
    assert base_kernel_name(NCU_ADD) != base_kernel_name(NCU_MUL)


def test_base_kernel_name_handles_nested_templates_and_commas():
    assert base_kernel_name(NCU_MUL) == (
        "vectorized_elementwise_kernel<AUnaryFunctor,MulFunctor>")


def test_base_kernel_name_leaves_untemplated_names_unchanged():
    assert base_kernel_name("_mlp_fused_kernel") == "_mlp_fused_kernel"
    assert base_kernel_name("ampere_sgemm_128x128_tn") == "ampere_sgemm_128x128_tn"


def test_base_kernel_name_falls_open_on_kernels_without_a_functor():
    """Documented limitation, asserted so it is a known quantity rather than a
    surprise: torch templates some elementwise kernels on a lambda instead of a
    named functor, and those carry no discriminator to keep. Such a name reduces
    to the bare kernel identity, so the guard cannot tell two of them apart. It
    fails OPEN (accepts) rather than closed, which is the safer asymmetry: a
    false rejection would cost a GPU re-run to diagnose."""
    assert base_kernel_name(NCU_GELU_LAMBDA) == "vectorized_elementwise_kernel"


def test_profile_kernel_rejects_setup_kernel_captured_as_fused(monkeypatch):
    """Regression test for the defect that motivated this guard: a launch window
    that skipped too few launches landed in the benchmark's own tensor setup, so
    a `* 0.05` elementwise multiply was recorded as `_mlp_fused_kernel`. Both are
    exactly one kernel, so the launch-count check passed it."""
    _patch_subprocess(monkeypatch, _sample(NCU_MUL))
    with pytest.raises(RuntimeError, match="captured the wrong kernel"):
        profile_kernel("bench.run_mlp", "mlp", "triton_fused", 128, "float32",
                       launch_count=1, expected_kernels=1,
                       expected_kernel_names={"_mlp_fused_kernel"})


def test_profile_kernel_rejects_wrong_functor(monkeypatch):
    """The count check cannot see this: one elementwise kernel was expected and
    one was captured, but not the same one."""
    _patch_subprocess(monkeypatch, _sample(NCU_MUL))
    with pytest.raises(RuntimeError, match="captured the wrong kernel"):
        profile_kernel("bench.run_x", "x", "triton", 128, "float32",
                       launch_count=1, expected_kernel_names={TORCH_ADD})


def test_profile_kernel_accepts_matching_names_across_spellings(monkeypatch):
    """The expected set comes from torch.profiler and the capture from ncu, so
    acceptance has to survive the two spellings differing."""
    _patch_subprocess(monkeypatch, _sample(NCU_ADD, "_layernorm_kernel"))
    rows = profile_kernel("bench.run_layernorm_residual", "layernorm_residual",
                          "triton", 128, "float32", launch_count=2,
                          expected_kernels=2,
                          expected_kernel_names={TORCH_ADD, "_layernorm_kernel"})
    assert len(rows) == 2


def test_profile_kernel_with_only_expected_kernels_skips_identity_check(monkeypatch):
    """The sm_75 vit_forward path in bench/collect_counters.py passes
    expected_kernels and no names; it must keep behaving exactly as before."""
    _patch_subprocess(monkeypatch, _sample(NCU_MUL))
    rows = profile_kernel("bench.run_x", "x", "triton", 128, "float32",
                          launch_count=1, expected_kernels=1)
    assert len(rows) == 1
