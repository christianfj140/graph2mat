"""Small fail-closed checks for the six approved 7.2 experiments."""
import copy
from pathlib import Path

import pytest

import run_bilayer_edge_updates as campaign


def test_recipe_only_changes_edge_update_and_output_locations(tmp_path):
    original = {"seed_everything": 2, "model": {"optim_lr": .0018},
                "data": {"basis_files": "basis/*.xml", "train_runs": "train/*/RUN.fdf",
                         "val_runs": "val/*/RUN.fdf", "batch_size": 32},
                "trainer": {"max_epochs": 533, "precision": "32-true", "logger": {
                    "class_path": "TensorBoardLogger", "init_args": {"save_dir": "logs", "name": "old"}}},
                "lr_scheduler": {"T_max": 533}}
    saved = copy.deepcopy(original)
    cfg = campaign.recipe(original, {"optim_lr": .0018, "optim_wdecay": 5e-7},
                          tmp_path / "old", tmp_path / "new")
    assert original == saved
    assert cfg["model"] == {"optim_lr": .0018, "optim_wdecay": 5e-7, "edge_update_layers": 1}
    assert cfg["data"]["batch_size"] == 32
    assert cfg["trainer"]["max_epochs"] == 533
    assert cfg["trainer"]["precision"] == "32-true"
    assert cfg["lr_scheduler"] == original["lr_scheduler"]
    assert cfg["data"]["root_dir"] == str(tmp_path / "old")
    assert cfg["data"]["basis_files"] == "basis/*.xml"
    assert cfg["data"]["train_runs"] == "train/*/RUN.fdf"


def test_manifest_hash_mismatch_blocks(tmp_path):
    artifact = tmp_path / "label.TSHS"
    artifact.write_bytes(b"label")
    with pytest.raises(ValueError, match="hash"):
        campaign.manifest_files({"rows": [{"artifact_paths": {"tshs": str(artifact)},
            "artifact_sha256": {"tshs": "wrong"}}]})


def test_missing_hash_blocks(tmp_path):
    with pytest.raises(ValueError, match="hash"):
        campaign.manifest_files({"rows": [{"artifact_paths": {"tshs": str(tmp_path / "x")},
                                          "artifact_sha256": {}}]})


def test_existing_failed_ancestry_is_not_new_dependency():
    jobs = [{"id": "done"}, {"id": "bad"}, {"id": "paused", "after": ["bad"]},
            {"id": "ready", "after": ["done"]}, {"id": "later", "after": ["ready"]},
            {"id": "missing", "after": ["absent"]}, {"id": "cycle", "after": ["cycle"]}]
    assert campaign.live_dependencies(jobs, {"done": ["done"], "failed": ["bad"]}) == ["ready", "later"]


def test_summary_never_uses_incomplete_seed_pairs():
    assert campaign.paired_summary([]) == {}
    rows = [{"material": "tbg", "seed": s, "candidate": {"id": {"H_MAE_gauge_meV": 8},
            "ood": {"band_rmse_aligned_meV": 12}}, "baseline": {
            "id": {"H_MAE_gauge_meV": 10}, "ood": {"band_rmse_aligned_meV": 10}}} for s in (0, 1)]
    assert campaign.paired_summary(rows)["tbg"]["status"] == "partial"
    rows.append(dict(rows[0], seed=2))
    result = campaign.paired_summary(rows)["tbg"]
    assert result["status"] == "complete"
    assert result["id/H_MAE_gauge_meV"]["ratio_of_means"] == .8
    assert result["ood/band_rmse_aligned_meV"]["ratio_of_means"] == 1.2


def test_output_root_cannot_escape_campaign(tmp_path):
    with pytest.raises(ValueError, match="campaign"):
        campaign.owned(tmp_path, tmp_path / "../outside")
    assert campaign.owned(tmp_path, tmp_path / "job") == tmp_path / "job"


def test_stalled_fit_fails_closed(tmp_path):
    progress = tmp_path / "progress.json"
    campaign.check_progress(progress, started=100, now=399, pilot=True)
    with pytest.raises(RuntimeError, match="no progress"):
        campaign.check_progress(progress, started=100, now=401, pilot=True)
    campaign.check_progress(progress, started=100, now=401, pilot=False)
    with pytest.raises(RuntimeError, match="no progress"):
        campaign.check_progress(progress, started=100, now=1001, pilot=False)


def test_child_environment_survives_external_working_directory():
    env = campaign.environment(cuda="")
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert all(Path(p).is_absolute() for p in env["PYTHONPATH"].split(":"))


def test_fit_does_not_start_unsafe_background_stack_dumps(monkeypatch):
    import faulthandler

    class StartupReached(Exception):
        pass

    def forbidden_timer(*args, **kwargs):
        pytest.fail("background all-thread dump can segfault during training")

    def stop_before_cuda():
        raise StartupReached

    monkeypatch.setattr(faulthandler, "dump_traceback_later", forbidden_timer)
    monkeypatch.setattr(campaign.pipeline, "compiled_extensions", stop_before_cuda)
    with pytest.raises(StartupReached):
        campaign.fit(None)
