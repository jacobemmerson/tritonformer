from bench.profile import METRICS, parse_ncu_csv

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
