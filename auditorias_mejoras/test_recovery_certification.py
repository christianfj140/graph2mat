"""Continuity is certified between fresh interpreters, not only new Trainers."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from run_bilayer_edge_updates import environment
from test_training_recovery import assert_state_equal


SCRIPT = Path(__file__).with_name("recovery_certification.py")


def test_gpu_tolerance_applies_to_derived_scores_but_not_counts_or_rng():
    from recovery_certification import compare
    reference = torch.tensor(1., dtype=torch.float64)
    near = reference + 1e-9
    compare(near, reference, "callbacks.best_model_score", (5e-6, 5e-6))
    with pytest.raises(AssertionError):
        compare(near, reference, "callbacks.best_model_score", (0, 0))
    with pytest.raises(AssertionError):
        compare(near, reference, "optimizer_states.0.state.0.step", (5e-6, 5e-6))
    with pytest.raises(AssertionError):
        compare(torch.tensor([1], dtype=torch.uint8), torch.tensor([0], dtype=torch.uint8),
                "recovery.rng.torch", (5e-6, 5e-6))


def certify(root, case, mode, workers, *, device="cpu", precision="32-true", negative=False, compiled=False):
    assert SCRIPT.is_file(), "fresh-process recovery certification is missing"
    command = [sys.executable, str(SCRIPT), "--root", str(root), "--device", device,
               "--precision", precision, "--case", case, "--mode", mode, "--workers", str(workers)]
    if negative:
        command.append("--negative-no-rng")
    if compiled:
        command.append("--compiled")
    result = subprocess.run(command, env=environment(cuda="" if device == "cpu" else "0"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=720 if compiled else 600)
    assert result.returncode == (1 if negative else 0), result.stdout[-12000:]
    return json.loads((root / "report.json").read_text())


@pytest.mark.parametrize("case", ["toy", "mace"])
@pytest.mark.parametrize("mode", ["fixed", "cost"])
@pytest.mark.parametrize("workers", [0, 4])
def test_cpu_new_processes_preserve_full_state_and_admitted_samples(tmp_path, case, mode, workers):
    root = tmp_path / "proof"
    report = certify(root, case, mode, workers)
    assert report["status"] == "passed"
    assert report["next_epoch"] == 4 and report["admitted_samples"] == 48
    assert report["repeated_samples"] > 0
    assert report["parameter_sha256_control"] == report["parameter_sha256_resumed"]
    assert report["best_tensor_sha256_control"] == report["best_tensor_sha256_resumed"]
    full = torch.load(report["control_checkpoint"], map_location="cpu", weights_only=False)
    resumed = torch.load(report["resumed_checkpoint"], map_location="cpu", weights_only=False)
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "loops"):
        assert_state_equal(resumed[key], full[key])
    assert_state_equal(resumed["g2m_recovery"]["rng"], full["g2m_recovery"]["rng"])
    admitted = json.loads((root / "admitted_trace.json").read_text())
    assert admitted["control"] == admitted["resumed"]
    assert sum(len(row["sample_ids"]) for row in admitted["control"]) == 48
    assert len({row["pid"] for row in report["children"]}) == 4


def test_fresh_process_negative_control_detects_missing_rng_restore(tmp_path):
    report = certify(tmp_path / "negative", "toy", "fixed", 0, negative=True)
    assert report["status"] == "failed" and report["failure_stage"] == "comparison"
    assert "state_dict" in report["error"]


@pytest.mark.skipif(os.environ.get("G2M_GPU_GUARDED") != "1", reason="explicit guarded GPU certification required")
@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
def test_cuda_native_recovery_restores_optimizer_scaler_and_rng(tmp_path, precision):
    report = certify(tmp_path / precision, "mace", "cost", 0, device="cuda", precision=precision)
    assert report["status"] == "passed"
    assert report["admitted_samples"] == 48 and report["repeated_samples"] > 0
    assert report["control_repeat_checked"] is True
    assert all(row["supervised"] for row in report["children"])
    assert max(row.get("release_seconds", 0) for row in report["children"]) < 11
    if precision == "16-mixed":
        assert report["scaler_checked"] is True


@pytest.mark.skipif(os.environ.get("G2M_GPU_GUARDED") != "1", reason="explicit guarded GPU certification required")
def test_cuda_compiled_mace_recovery_preserves_state_with_four_workers(tmp_path):
    report = certify(tmp_path / "compiled", "mace", "cost", 4, device="cuda", compiled=True)
    assert report["status"] == "passed" and report["compiled"] is True
    assert report["control_repeat_checked"] is True
    assert report["admitted_samples"] == 48 and report["repeated_samples"] > 0
    assert len({row["pid"] for row in report["children"]}) == 5
    assert max(row["release_seconds"] for row in report["children"]) < 11
