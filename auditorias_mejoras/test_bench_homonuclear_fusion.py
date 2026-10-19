"""Never report speedups from incomplete or scientifically unmatched pairs."""

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("fusion_probe", HERE / "bench_homonuclear_fusion.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def records():
    common = dict(system="df72", round=0, initial_sha256="weights", inputs_sha256="batch",
                  recipe_sha256="recipe", peak_allocated_mib=10., peak_reserved_mib=20.,
                  preflight={"passed": True, "reference_loss": 1.5}, warmup=10, measured_steps=30)
    return [dict(copy.deepcopy(common), fusion_variant="off", wall_ms=[10., 10.]),
            dict(copy.deepcopy(common), fusion_variant="on", wall_ms=[8., 8.])]


def test_literal_ratio():
    assert probe.summarize(records())["df72"]["paired_wall_ratio"] == [0.8]


@pytest.mark.parametrize("fault", ["empty", "gap", "infinite_loss"])
def test_incomplete_campaign_or_infinite_reference_rejected(fault):
    rows = records()
    if fault == "empty":
        rows = []
    elif fault == "gap":
        extra = copy.deepcopy(rows)
        for row in extra:
            row["round"] = 2
        rows += extra
    else:
        for row in rows:
            row["preflight"]["reference_loss"] = float("inf")
    with pytest.raises(ValueError):
        probe.summarize(rows)


@pytest.mark.parametrize("key", ["initial_sha256", "inputs_sha256", "recipe_sha256", "warmup", "measured_steps"])
def test_unmatched_pair_rejected(key):
    rows = records()
    rows[1][key] = "different"
    with pytest.raises(ValueError):
        probe.summarize(rows)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "aborted", "numeric", "nonfinite", "loss", "vram"])
def test_invalid_records_rejected(fault):
    rows = copy.deepcopy(records())
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif fault == "aborted":
        rows[1]["aborted"] = True
    elif fault == "numeric":
        rows[1]["preflight"]["passed"] = False
    elif fault == "loss":
        rows[1]["preflight"]["reference_loss"] = 2.5
    elif fault == "vram":
        rows[1]["peak_allocated_mib"] = float("nan")
    else:
        rows[1]["wall_ms"] = [float("nan")]
    with pytest.raises(ValueError):
        probe.summarize(rows)
