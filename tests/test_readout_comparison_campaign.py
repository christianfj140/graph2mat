"""Campaign safety: immutable recipes, paired initial states and append-only queue."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


AUDIT = Path(__file__).resolve().parents[1] / "auditorias_mejoras"


def module(name):
    path = AUDIT / (name + ".py")
    assert path.is_file(), "approved campaign adapter is not implemented"
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize("variant", ["baseline", "low_rank", "sk_residual"])
def test_recipe_preserves_scientific_budget_and_only_changes_readout(tmp_path, variant):
    adapter = module("run_readout_comparison")
    original = {"seed_everything": 2, "model": {"loss": "gauge", "optim_lr": .0018,
                "hidden_irreps": "48x0e", "torch_compile": True},
                "data": {"batch_size": 16}, "trainer": {"max_epochs": 16000,
                "callbacks": [{"class_path": "ModelCheckpoint"}]},
                "lr_scheduler": {"init_args": {"T_max": 16000}}}
    untouched = copy.deepcopy(original)
    cfg = adapter.recipe(original, variant, tmp_path, tmp_path / "shared.pt")
    assert original == untouched
    assert cfg["data"] == original["data"]
    assert cfg["lr_scheduler"] == original["lr_scheduler"]
    assert cfg["trainer"]["max_epochs"] == 16000
    assert cfg["model"]["loss"] == "gauge" and cfg["model"]["optim_lr"] == .0018
    assert cfg["model"]["torch_compile"] is False
    assert cfg["model"]["gradient_checkpointing"] is False
    assert cfg["trainer"]["precision"] == "32-true"
    assert cfg["model"]["edge_block_readout"].endswith({"baseline": "E3nnLinearEdgeBlock",
           "low_rank": "E3nnLowRankEdgeBlock", "sk_residual": "E3nnSlaterKosterEdgeBlock"}[variant])
    assert cfg["model"]["edge_block_readout_kwargs"] == (None if variant == "baseline" else {"rank": 4})


def test_append_queue_is_idempotent_preserves_foreign_bytes_and_rejects_collision(tmp_path):
    adapter = module("run_readout_comparison")
    queue = tmp_path / "queue.jsonl"
    foreign = b'{"id":"foreign","args":[]}\n'
    queue.write_bytes(foreign)
    job = {"id": "readout:campaign", "script": "/own/runner.py", "args": ["run"]}
    assert adapter.append_queue(queue, job) is True
    assert queue.read_bytes().startswith(foreign)
    assert adapter.append_queue(queue, job) is False
    assert len(queue.read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="collision"):
        adapter.append_queue(queue, {**job, "args": ["different"]})
    assert len(queue.read_text().splitlines()) == 2


def test_source_drift_refuses_launch(tmp_path):
    adapter = module("run_readout_comparison")
    source = tmp_path / "input.HSX"
    source.write_bytes(b"original")
    hashes = {str(source): adapter.sha256(source)}
    adapter.verify_files(hashes)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        adapter.verify_files(hashes)


def test_recipe_rebases_relative_globs_for_every_attempt(tmp_path):
    adapter = module("run_readout_comparison")
    basis = tmp_path / "basis"
    basis.mkdir()
    (basis / "C.ion.xml").write_text("basis")
    original = {"seed_everything": 0, "model": {}, "trainer": {},
                "data": {"root_dir": str(tmp_path), "basis_files": "basis/*.ion.xml",
                         "train_runs": "train_samples/*/RUN.fdf",
                         "val_runs": "validation_samples/*/RUN.fdf"}}
    template = adapter.recipe(original, "baseline", tmp_path / "template", tmp_path / "shared.pt")
    cfg = adapter.recipe(template, "baseline", tmp_path / "attempt", tmp_path / "shared.pt")
    root = Path(cfg["data"]["root_dir"])
    assert root == tmp_path / "attempt"
    root.mkdir()
    assert [p.resolve() for p in root.glob(cfg["data"]["basis_files"])] == [basis / "C.ion.xml"]
    assert not Path(cfg["data"]["train_runs"]).is_absolute()
    assert not Path(cfg["data"]["val_runs"]).is_absolute()
    assert sum(cb["class_path"].endswith("ComparisonCallback") for cb in cfg["trainer"]["callbacks"]) == 1


def test_foreign_gpu_process_blocks_but_own_descendants_do_not():
    adapter = module("run_readout_comparison")
    assert adapter.foreign_pids("42, 100\n51, 200\n", {42}) == {51}
    assert adapter.foreign_pids("42, 100\n", {42}) == set()
    with pytest.raises(ValueError):
        adapter.foreign_pids("GPU telemetry unavailable", set())


def test_input_drift_while_waiting_for_gpu_refuses_child_launch(tmp_path, monkeypatch):
    adapter = module("run_readout_comparison")
    source = tmp_path/"RUN.fdf"
    source.write_text("original")
    hashes = {str(source): adapter.sha256(source)}
    monkeypatch.setattr(adapter, "idle_gpu", lambda: source.write_text("changed while queued"))
    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("must reject changed inputs before spawning training")
    monkeypatch.setattr(adapter.subprocess, "Popen", forbidden_spawn)
    with pytest.raises(ValueError, match="changed"):
        adapter.child_fit(tmp_path, hashes=hashes)


def test_probe_config_has_fixed_batch_no_validation_and_identical_updates(tmp_path):
    adapter = module("run_readout_comparison")
    original = {"seed_everything": 0, "model": {}, "data": {"batch_size": 32},
                "trainer": {"max_epochs": 4000, "check_val_every_n_epoch": 8},
                "lr_scheduler": {"init_args": {"T_max": 4000}}}
    cfg = adapter.recipe(original, "sk_residual", tmp_path, tmp_path / "shared.pt", phase="probe")
    assert cfg["trainer"]["max_steps"] == 40
    assert cfg["trainer"]["max_epochs"] == 40
    assert cfg["trainer"]["limit_train_batches"] == 1
    assert cfg["trainer"]["limit_val_batches"] == 0
    assert cfg["trainer"]["num_sanity_val_steps"] == 0
    assert cfg["data"]["batch_size"] == 32
    assert cfg["data"]["train_runs"] == str(tmp_path / "probe_samples/*/RUN.fdf")
    assert cfg["trainer"]["callbacks"][-1]["init_args"]["warmup"] == 10


def test_callback_strictly_pairs_weights_and_reseeds_before_data_iteration(tmp_path):
    import torch
    callback = module("readout_comparison_callback")
    def tiny():
        return SimpleNamespace(model=SimpleNamespace(mace=torch.nn.Linear(2, 3),
            matrix_readouts=SimpleNamespace(self_interactions=torch.nn.Linear(3, 2))))
    torch.manual_seed(11)
    a = tiny()
    torch.manual_seed(99)
    b = tiny()
    trainer = SimpleNamespace(ckpt_path=None)
    path = tmp_path / "shared.pt"
    first = callback.ComparisonCallback(str(path), str(tmp_path / "a.json"), 7)
    first.on_fit_start(trainer, a)
    second = callback.ComparisonCallback(str(path), str(tmp_path / "b.json"), 7)
    second.on_fit_start(trainer, b)
    for left, right in ((a.model.mace, b.model.mace),
                        (a.model.matrix_readouts.self_interactions, b.model.matrix_readouts.self_interactions)):
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[key], atol=0, rtol=0)
    first.on_train_start(trainer, a)
    x = torch.rand(3)
    second.on_train_start(trainer, b)
    torch.testing.assert_close(x, torch.rand(3), atol=0, rtol=0)
    assert first.initial_hashes == second.initial_hashes


def test_callback_records_raw_phases_and_real_gradient_failure(tmp_path):
    import torch
    callback = module("readout_comparison_callback")
    model = torch.nn.Linear(2, 1)
    trainer = SimpleNamespace(global_step=0)
    cb = callback.ComparisonCallback(str(tmp_path / "shared.pt"), str(tmp_path / "proof.json"), 0, warmup=0)
    cb.on_train_batch_start(trainer, model, None, 0)
    loss = model(torch.ones(1, 2)).square().sum()
    cb.on_before_backward(trainer, model, loss)
    loss.backward()
    cb.on_after_backward(trainer, model)
    cb.on_train_batch_end(trainer, model, {"loss": loss}, None, 0)
    model.zero_grad(set_to_none=True)  # Lightning clears gradients before validation/end.
    cb.on_train_end(trainer, model)
    proof = json.loads((tmp_path / "proof.json").read_text())
    assert len(proof["raw_ms"]["step"]) == 1
    assert proof["raw_ms"]["forward"][0] >= 0
    assert proof["raw_ms"]["backward"][0] >= 0
    assert proof["device"] == "cpu" and proof["peak_allocated_bytes"] is None
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    with pytest.raises(ValueError, match="gradient"):
        cb.on_train_batch_end(trainer, model, {"loss": loss}, None, 0)


def test_paired_loader_order_is_independent_of_model_rng_consumption():
    import torch
    from torch_geometric.data import Data
    callback = module("readout_comparison_callback")
    # Mutating the loader back to global RNG would pair seeds but not sample orders.
    a, b = callback.PairedDataModule(), callback.PairedDataModule()
    for value in (a, b):
        value.train_dataset = [Data(x=torch.tensor([[i]])) for i in range(12)]
        value.batch_size = 3
        value.loader_threads = 0
        value.data_processor = SimpleNamespace(symmetric_matrix=False)
    torch.manual_seed(7)
    x = [batch.x.flatten().tolist() for batch in a.train_dataloader()]
    torch.manual_seed(7)
    torch.rand(173)
    y = [batch.x.flatten().tolist() for batch in b.train_dataloader()]
    assert x == y and sorted(i for row in x for i in row) == list(range(12))


def test_scientific_summary_pairs_seeds_and_vetoes_worse_ood(tmp_path):
    adapter = module("run_readout_comparison")
    for system in ("w90", "6x6"):
        for variant, error in (("baseline", 2), ("low_rank", 1), ("sk_residual", 3)):
            for seed in (0, 1, 2):
                for target in ("test", "registry", "ab_md", "moire"):
                    directory = tmp_path/"evaluation"/f"{system}-{variant}-seed{seed}"/target
                    directory.mkdir(parents=True)
                    band = 2.2 if variant == "low_rank" else 2
                    (directory/"per_structure_metrics.csv").write_text(
                        "sample_id,H_MAE_gauge_meV,band_rmse_aligned_meV\nx,"+str(error)+","+str(band)+"\n")
    adapter.summarize(tmp_path)
    result = json.loads((tmp_path/"precision_summary.json").read_text())
    low = result["w90"]["low_rank"]
    assert low["primary_ratio_geomean"] == .5
    assert low["primary_ratio_ci95"] == [.5, .5]
    assert low["promotion"] is False
    assert "registry:band_rmse_aligned_meV" in low["ood_regressions"]
    assert result["6x6"]["sk_residual"]["primary_ratio_geomean"] == 1.5


def test_probe_summary_reports_paired_speed_vram_and_rejects_unpaired_batches(tmp_path):
    adapter = module("run_readout_comparison")
    (tmp_path/"completed").mkdir()
    for index in range(2):
        for variant, scale in (("baseline", 2), ("low_rank", 1), ("sk_residual", 1.5)):
            telemetry = {"raw_ms": {key: [10*scale, 10*scale] for key in ("step", "forward", "backward")},
                         "peak_allocated_bytes": int(1000*scale), "peak_reserved_bytes": int(2000*scale),
                         "initial_hashes": {"backbone": "same", "onsite": "same"}, "batch_hash": "fixed"}
            adapter.write_json(tmp_path/"completed"/f"probe-w90-{index:02d}-{variant}.json", {"telemetry": telemetry})
    adapter.summarize(tmp_path)
    result = json.loads((tmp_path/"probe_summary.json").read_text())["w90"]
    assert result["low_rank"]["paired_ratios_vs_baseline"]["step"] == [.5, .5]
    assert result["low_rank"]["paired_ratios_vs_baseline"]["peak_allocated_bytes"] == [.5, .5]
    assert result["sk_residual"]["paired_ratios_vs_baseline"]["peak_reserved_bytes"] == [.75, .75]
    path = tmp_path/"completed/probe-w90-00-sk_residual.json"
    broken = json.loads(path.read_text())
    broken["telemetry"]["batch_hash"] = "different"
    adapter.write_json(path, broken)
    with pytest.raises(ValueError, match="unpaired"):
        adapter.summarize(tmp_path)
