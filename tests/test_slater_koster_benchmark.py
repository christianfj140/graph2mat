"""Real subprocess smoke; timings are evidence, never performance thresholds."""

import json
import importlib
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import pytest


def test_rss_workers_start_before_parent_allocates_timing_models(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "auditorias_mejoras"))
    bench = importlib.import_module("bench_slater_koster_cpu")
    calls = []
    monkeypatch.setattr(bench, "host_memory", lambda *args: calls.append("memory") or {})
    def stop(*args):
        calls.append("model")
        raise RuntimeError("stop before model allocation")
    monkeypatch.setattr(bench, "model_for", stop)
    with pytest.raises(RuntimeError, match="stop before"):
        bench.probe(4, 1, 1, 1)
    assert calls == ["memory", "model"]


def test_three_variant_cpu_benchmark_contract():
    root = Path(__file__).resolve().parents[1]
    script = root / "auditorias_mejoras/bench_slater_koster_cpu.py"
    assert script.exists(), "7.4 paired benchmark missing"
    # The installed graph2mat (compiled extensions), not the uncompiled src tree.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    run = subprocess.run([sys.executable, str(script), "--points", "4", "--rank", "1",
                          "--rounds", "1", "--warmup", "1"],
                         capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    names = {"baseline", "low_rank", "sk_residual"}
    assert result["device"] == "cpu" and result["dtype"] == "float32"
    assert "vram_bytes" not in result
    for scope in ("full_model", "operation"):
        assert set(result["measure"][scope]) == names
        for phases in result["measure"][scope].values():
            for record in phases.values():
                raw = record["raw_ms"]
                assert len(raw) == result["rounds"] and all(math.isfinite(x) and x >= 0 for x in raw)
                median = statistics.median(raw)
                assert record["median"] == median
                assert record["mad"] == statistics.median(abs(x-median) for x in raw)
    assert len(result["batch_tensor_sha256"]) == 64 and result["input_hashes_unchanged"]
    assert result["branches_effective"] and result["shared_weights_equal"]
    assert set(result["host_memory"]) == names
    for memory in result["host_memory"].values():
        assert memory["process_peak_rss_bytes"] > 0
        assert "whole process" in memory["scope"]
    assert "architectures differ" in result["limits"]
