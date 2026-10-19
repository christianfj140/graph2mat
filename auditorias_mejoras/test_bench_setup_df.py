"""CPU contracts of the setup comparison and diagnostic profile reporting."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def runner():
    path = Path(__file__).with_name("bench_setup_df.py")
    assert path.exists(), "setup/profile runner missing"
    spec = importlib.util.spec_from_file_location("setup_df", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_summary_rejects_different_outputs_and_pairs_rounds():
    module = runner()
    records = [dict(system="w90", variant=v, round=i, samples_sha256="same", setup_ms=t)
               for i in (0, 1) for v, t in (("baseline", 10), ("candidate", 7))]
    assert module.setup_summary(records)["w90"]["paired_ratio"]["median"] == .7
    records[-1]["samples_sha256"] = "different"
    with pytest.raises(AssertionError, match="outputs"):
        module.setup_summary(records)


def test_profile_report_uses_device_not_cpu_time_and_marks_missing_device_data():
    module = runner()
    def event(key, gpu, device="CPU"):
        return SimpleNamespace(key=key, device_time_total=gpu, cpu_time_total=30000, count=2,
                               device_type=SimpleNamespace(name=device))
    rows = module.profile_rows([event("g2m/forward", 10000), event("g2m/change_of_basis", 2000),
                                event("aten::mm", 4000), event("g2m/forward", 100, "CUDA"),
                                event("g2m/change_of_basis", 200000, "CUDA"),
                                event("g2m/backward", 1)])
    assert rows["g2m/change_of_basis"]["cuda_ms"] == 2
    assert rows["g2m/change_of_basis"]["fraction_forward_cuda"] == .2
    assert "aten::mm" not in rows
    assert rows["g2m/backward"]["cuda_ms"] is None
    assert rows["g2m/backward"]["fraction_forward_cuda"] is None
    assert module.profile_rows([event("g2m/forward", 0)])["g2m/forward"]["fraction_forward_cuda"] is None
