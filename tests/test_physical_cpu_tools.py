"""Contracts of the standalone, read-only physical CPU evidence tools."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def cpu_tools(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "auditorias_mejoras"))
    return (importlib.import_module("smoke_objetivos_fisicos"),
        importlib.import_module("bench_objetivos_fisicos"))


def test_timing_summary_and_pair_order(cpu_tools):
    _, bench = cpu_tools
    assert bench.summarize_seconds([1., 2., 3.]) == {
        "median_s": 2., "mad_s": 1., "samples_s": [1., 2., 3.]}
    calls = []
    result = bench.paired_times(lambda: calls.append("A"), lambda: calls.append("B"), warmup=1, rounds=4)
    assert calls == list("ABABBAABBA")
    assert len(result["baseline"]["samples_s"]) == len(result["physical"]["samples_s"]) == 4
    for warmup, rounds in ((-1, 1), (0, 0)):
        with pytest.raises(ValueError):
            bench.paired_times(lambda: None, lambda: None, warmup, rounds)


def test_outputs_cannot_overwrite_or_enter_source_tree(cpu_tools, tmp_path):
    smoke, _ = cpu_tools
    source = tmp_path / "dataset" / "MD_steps" / "0" / "RUN.fdf"
    source.parent.mkdir(parents=True)
    source.write_text("SystemLabel reference\n")
    for path in (source, source.parent / "results.json", source.parent.parent / "results.json"):
        with pytest.raises(ValueError, match="output|dataset|source"):
            smoke.validate_paths(source, path)
    output = tmp_path / "results.json"
    assert smoke.validate_paths(source, output) == (source.resolve(), output.resolve())
    output.write_text("preserve existing output")
    with pytest.raises(ValueError, match="exist|overwrite"):
        smoke.validate_paths(source, output)


def test_source_hashes_and_analytic_self_check(cpu_tools, tmp_path):
    smoke, _ = cpu_tools
    source = tmp_path / "RUN.fdf"
    source.write_text("SystemLabel reference\n")
    (tmp_path / "reference.TSHS").write_bytes(b"H/S")
    (tmp_path / "reference.TSDE").write_bytes(b"D/E")
    (tmp_path / "reference.HSX").write_bytes(b"HSX")
    hashes = smoke.source_hashes(source)
    assert len(hashes) == 4
    assert hashes == smoke.source_hashes(source)
    (tmp_path / "reference.TSDE").write_bytes(b"changed")
    assert hashes != smoke.source_hashes(source)
    smoke.self_check()


def test_h_energy_unknown_partner_is_blocked_not_validated(cpu_tools, monkeypatch, tmp_path):
    for module in cpu_tools:
        monkeypatch.setattr(module, "processor_for", lambda *args: object())
        def load(fdf, processor, term=None):
            if term:
                raise ValueError("unknown/unobserved density partner where predicted H can vary")
            return object()
        monkeypatch.setattr(module, "load_sample", load)
        function = module.smoke_case if hasattr(module, "smoke_case") else module.measure_case
        extra = () if hasattr(module, "smoke_case") else (0, 1)
        result = function("source.fdf", "hamiltonian", "energy_weight", tmp_path, *extra)
        assert result["status"] == "blocked"
        with pytest.raises(ValueError, match="unknown"):
            function("source.fdf", "density_matrix", "energy_weight", tmp_path, *extra)


def test_auxiliary_byte_count_includes_batched_k_metadata(cpu_tools):
    import numpy as np
    from graph2mat.core.modules.graph2mat import LAYOUT_ATTR

    class Data(dict):
        metadata = {"physical_reference": {"kpoints": [np.zeros((1, 3))]}}
    data = Data(physical_reference_nodes=np.zeros(2))
    setattr(data, LAYOUT_ATTR, {"physical_node_owner": np.zeros(2, dtype=np.int64)})
    assert cpu_tools[1].auxiliary_bytes(data) == 56
