"""CPU checks of the GPU probe's resource policy and paired analysis."""
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).with_name("bench_pipeline_gpu.py")


def load_runner():
    assert SCRIPT.is_file(), "GPU pipeline runner is missing"
    spec = importlib.util.spec_from_file_location("pipeline_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_leaves_thirty_percent_plus_foreign_allocations():
    runner = load_runner()
    for total, free in ((32000, 31000), (32000, 16000), (32000, 11000)):
        fraction = runner.memory_fraction(total, free)
        assert 0 < fraction <= .55
        assert free - fraction * total >= .30 * total - 1e-8
    with pytest.raises(RuntimeError, match="headroom"):
        runner.memory_fraction(32000, 9600)


def test_summary_requires_identical_initial_weights_and_inputs():
    runner = load_runner()
    records = [dict(task="coeff", system="w90", variant=name, round=0,
                    initial_sha256="same", inputs_sha256="data", unit="step",
                    wall_ms=[value], peak_allocated_mib=10, peak_reserved_mib=12)
               for name, value in (("mse_dense", 10), ("mse_coeff", 8))]
    summary = runner.summarize(records)
    assert summary["coeff/w90"]["mse_coeff"]["paired_wall_ratio"] == [.8]
    records[1]["inputs_sha256"] = "different"
    with pytest.raises(AssertionError, match="inputs"):
        runner.summarize(records)


def test_sparse_pair_requires_same_prediction_reconstruction_oracle():
    runner = load_runner()
    records = [dict(task="sparse", system="w90", variant=name, round=0,
                    initial_sha256="same", inputs_sha256="data", unit="step",
                    wall_ms=[10], peak_allocated_mib=10, peak_reserved_mib=12,
                    exact_sparse_equivalence=True, matrix_sha256=name)
               for name in ("legacy", "vectorized")]
    assert runner.summarize(records)["sparse/w90"]["vectorized"]["paired_wall_ratio"] == [1.]
    records[1]["exact_sparse_equivalence"] = False
    with pytest.raises(AssertionError, match="oracle"):
        runner.summarize(records)


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_four_rounds_balance_every_variant_pair(size):
    runner = load_runner()
    variants = tuple(range(size))
    orders = [runner.variant_order(variants, index) for index in range(4)]
    for a in variants:
        for b in variants:
            if a < b:
                assert sum(order.index(a) < order.index(b) for order in orders) == 2


def test_loader_factorial_changes_workers_and_pinning_independently():
    runner = load_runner()
    options = [runner.loader_options("loader", name) for name in ("w0_p0", "w0_p1", "w4_p0", "w4_p1")]
    assert options == [(0, False), (0, True), (4, False), (4, True)]
    assert runner.loader_options("cache", "safe") == (4, True)
    assert runner.loader_options("cache", "safe0") == (0, False)
    with pytest.raises(ValueError):
        runner.loader_options("loader", "w4_p2")


def test_memory_tree_excludes_foreign_processes_and_handles_disappearing_pid(tmp_path):
    runner = load_runner()
    for pid, children, rss, pss in ((10, "11 12", 100, 80), (11, "", 50, 30), (99, "", 900, 900)):
        root = tmp_path / str(pid)
        (root / "task" / str(pid)).mkdir(parents=True)
        (root / "task" / str(pid) / "children").write_text(children)
        (root / "status").write_text(f"Name:\tfixture\nVmRSS:\t{rss} kB\n")
        (root / "smaps_rollup").write_text(f"Rss: {rss} kB\nPss: {pss} kB\n")
    result = runner.process_tree_memory(10, tmp_path)
    assert result["pids"] == [10, 11]
    assert result["workers"] == 1
    assert result["main_rss_mib"] == 100 / 1024
    assert result["children_rss_mib"] == 50 / 1024
    assert result["rss_sum_mib"] == 150 / 1024
    assert result["pss_sum_mib"] == 110 / 1024
    (tmp_path / "11" / "smaps_rollup").unlink()
    assert runner.process_tree_memory(10, tmp_path)["pss_sum_mib"] is None


def test_summary_loader_pairs_against_unpinned_zero_worker_reference():
    runner = load_runner()
    records = [dict(task="loader", system="w90", variant=name, round=0,
                    initial_sha256="same", inputs_sha256="data", unit="step",
                    wall_ms=[value], peak_allocated_mib=10, peak_reserved_mib=12)
               for name, value in (("w0_p0", 10), ("w4_p0", 12))]
    assert runner.summarize(records)["loader/w90"]["w4_p0"]["paired_wall_ratio"] == [1.2]


def test_summary_coefficient_optimization_uses_original_projector_reference():
    runner = load_runner()
    records = [dict(task="coeff_opt", system="6x6", variant=name, round=0,
                    initial_sha256="same", inputs_sha256="data", unit="step",
                    wall_ms=[value], peak_allocated_mib=10, peak_reserved_mib=12)
               for name, value in (("mse_legacycoeff", 10), ("mse_coeff", 8))]
    assert runner.summarize(records)["coeff_opt/6x6"]["mse_coeff"]["paired_wall_ratio"] == [.8]
    records.extend([dict(task="coeff_opt", system="6x6", variant=name, round=0,
        initial_sha256="gauge-model", inputs_sha256="h-plus-overlap", unit="step",
        wall_ms=[value], peak_allocated_mib=10, peak_reserved_mib=12)
        for name, value in (("gauge_dense", 10), ("gauge_coeff", 10))])
    assert runner.summarize(records)["coeff_opt/6x6"]["gauge_coeff"]["paired_wall_ratio"] == [1.]


def test_guard_terminates_only_own_child_and_keeps_telemetry(tmp_path, monkeypatch):
    import sys
    runner = load_runner()
    monkeypatch.setattr(runner, "gpu_state", lambda: {"memory.total": 100, "memory.free": 24,
                                                    "temperature.gpu": 50})
    with pytest.raises(RuntimeError, match="guard"):
        runner.guarded_command([sys.executable, "-c", "import time; time.sleep(30)"],
                               tmp_path / "child.log", tmp_path / "telemetry.json", {})
    import json
    assert json.loads((tmp_path / "telemetry.json").read_text())[0]["memory.free"] == 24


@pytest.mark.parametrize("abort", [False, True])
def test_guard_cleans_descendants_even_if_leader_exits_or_worker_ignores_term(tmp_path, monkeypatch, abort):
    import os
    import signal
    import sys
    import time
    runner = load_runner()
    pid_file = tmp_path / "worker.pid"
    ready = tmp_path / "ready"
    worker_code = ("import signal,time;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"Path({str(ready)!r}).touch();time.sleep(30)")
    leader_code = ("import subprocess,sys,time;from pathlib import Path;"
        f"p=subprocess.Popen([sys.executable,'-c',{worker_code!r}]);"
        f"Path({str(pid_file)!r}).write_text(str(p.pid));"
        + ("time.sleep(30)" if abort else "time.sleep(.1);sys.exit(7)"))
    def state():
        return {"memory.total": 100, "memory.free": 24 if abort and ready.exists() else 90,
                "temperature.gpu": 50}
    monkeypatch.setattr(runner, "gpu_state", state)
    try:
        command = [sys.executable, "-c", leader_code]
        if abort:
            with pytest.raises(RuntimeError, match="guard"):
                runner.guarded_command(command, tmp_path / "log", tmp_path / "telemetry", os.environ.copy())
        else:
            assert runner.guarded_command(command, tmp_path / "log", tmp_path / "telemetry", os.environ.copy()) == 7
        pid = int(pid_file.read_text())
        def alive():
            try:
                return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
            except FileNotFoundError:
                return False
        deadline = time.monotonic() + 2
        while alive() and time.monotonic() < deadline:
            time.sleep(.02)
        assert not alive(), "supervisor leaked a live worker after its leader exited"
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
