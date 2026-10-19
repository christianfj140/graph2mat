"""Native loss/selection and fail-closed campaign tests; CPU only."""
import importlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_cost_batching_comparison import tiny_campaign  # noqa: F401


def callback_api():
    assert Path(__file__).with_name("h2oclu_nullspace_callback.py").is_file(), "nullspace callback missing"
    return importlib.import_module("h2oclu_nullspace_callback")


def campaign_api():
    assert Path(__file__).with_name("run_h2oclu_nullspace.py").is_file(), "nullspace campaign missing"
    return importlib.import_module("run_h2oclu_nullspace")


@pytest.fixture
def frozen_nullspace(tiny_campaign, monkeypatch, tmp_path):  # noqa: F811
    recovery, source, pilot = tiny_campaign
    original = recovery.prepare(pilot, source)
    api = campaign_api()
    records = []
    for seed in range(3):
        cfg = copy.deepcopy(original["recipes"][str(seed)])
        cfg["data"]["max_batch_cost"] = None
        cfg["trainer"]["callbacks"] = [{"class_path": "ModelCheckpoint", "init_args": {
            "monitor": "val_loss", "mode": "min", "save_top_k": 1}}]
        initial = original["initial_states"][str(seed)]
        native = dict(initial_state=initial["path"], initial_sha256=initial["fixed_sha256"],
                      sample_ids=original["sample_ids"], input_resolution=recovery.input_resolution(cfg))
        # This boundary substitutes the costly closed historical fits, not the code
        # which creates/validates candidate overlays, configs, hashes or queue bytes.
        folder = pilot/f"fixed_mock{seed}"
        folder.mkdir()
        result = dict(arm="fixed", seed=seed, evaluation={"rows": [dict(
            sample_id=r["sample_id"], split=r["split"], H_MAE_gauge_meV=1.,
            levels_rmse_aligned_meV=2., gap_error_meV=3., spectral_valid=True)
            for r in original["rows"] if r["split"] in ("validation", "test")]})
        result_path = folder/"result.json"
        result_path.write_text(json.dumps(result))
        records.append(dict(seed=seed, config=cfg, native=native, root=str(folder), result=result,
                            result_path=str(result_path)))
    baseline = dict(rows=original["rows"], files={**original["files"], str(pilot/"manifest.json"): recovery.sha256(pilot/"manifest.json")}, versions=original["versions"],
                    records=records, source_root=str(pilot), manifest_sha256=recovery.sha256(pilot/"manifest.json"))
    monkeypatch.setattr(api, "read_baselines", lambda root: baseline)
    return api, pilot, tmp_path/"auditorias_mejoras/nullspace", baseline


def test_prepare_rejects_missing_authorization_before_reading_or_creating(tmp_path, monkeypatch):
    a = campaign_api()
    def forbidden(*args):
        pytest.fail("baseline read without authorization")
    monkeypatch.setattr(a, "read_baselines", forbidden)
    with pytest.raises(ValueError, match="authoriz"):
        a.prepare(tmp_path/"auditorias_mejoras/new")
    assert not (tmp_path/"auditorias_mejoras/new").exists()


def test_prepare_pins_three_jobs_same_initials_and_order(frozen_nullspace):
    a, baseline_root, root, baseline = frozen_nullspace
    m = a.prepare(root, baseline_root=baseline_root, approved=True)
    assert [j["key"] for j in m["jobs"]] == ["null_s0", "null_s1", "null_s2"]
    assert len(m["projection_rows"]) == 36
    for j, old in zip(m["jobs"], baseline["records"]):
        native, cfg = a.recovery.read_job(Path(j["config"]), Path(j["root"]))
        assert native["initial_sha256"] == old["native"]["initial_sha256"]
        assert native["sample_ids"] == old["native"]["sample_ids"]
        assert cfg["data"]["batch_size"] == 8 and cfg["data"]["max_batch_cost"] is None
        assert cfg["model"]["loss_kwargs"] == {"spectral_weight": 1e-4}
        assert cfg["trainer"]["max_epochs"] == 2000
        assert cfg["trainer"]["callbacks"][0]["class_path"].endswith("NullspaceTrainingCheckpoint")
        # The native CLI declares train_runs/val_runs as Optional[str], not list.
        assert cfg["data"]["train_runs"] is None and cfg["data"]["val_runs"] is None
        splits = json.loads(Path(cfg["data"]["runs_json"]).read_text())
        resolved = a.recovery.input_resolution(old["config"])
        for key, split in (("train_runs", "train"), ("val_runs", "val")):
            assert [str(Path(p).resolve()) for p in splits[split]] == [p["target"] for p in resolved[key]]
    a.read_campaign(root)


def test_changed_frozen_input_blocks_before_new_output(frozen_nullspace):
    a, baseline_root, root, old = frozen_nullspace
    path = Path(old["rows"][0]["reference_tshs_path"])
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="changed"):
        a.prepare(root, baseline_root=baseline_root, approved=True)
    assert not root.exists()


def test_existing_destination_or_symlink_cannot_be_overwritten(frozen_nullspace):
    a, baseline_root, root, _ = frozen_nullspace
    root.mkdir()
    with pytest.raises(ValueError, match="exclusive|exist"):
        a.prepare(root, baseline_root=baseline_root, approved=True)
    link = root.with_name("linked")
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        a.prepare(link, baseline_root=baseline_root, approved=True)


def test_overlay_native_tensors_equal_and_source_retarget_rejected(frozen_nullspace):
    a, baseline_root, root, old = frozen_nullspace
    m = a.prepare(root, baseline_root=baseline_root, approved=True)
    assert m["overlay_equivalence"]["samples_checked"] == 36
    first = Path(m["projection_rows"][0]["input"])
    original = first.resolve()
    changed = first.parent/"different.nc"
    changed.write_bytes(original.read_bytes())
    first.unlink()
    first.symlink_to(changed)
    with pytest.raises(ValueError, match="resolution|changed|order"):
        a.read_campaign(root)


def test_enqueue_is_idempotent_and_preserves_previous_bytes(frozen_nullspace):
    a, baseline_root, root, _ = frozen_nullspace
    a.prepare(root, baseline_root=baseline_root, approved=True)
    queue = root/"queue.jsonl"
    original = b'{"id":"old","after":[]}\n'
    queue.write_bytes(original)
    queue.with_name("state.json").write_text(json.dumps({"done": ["old"], "failed": [], "pids": {}}))
    first = a.enqueue(root, queue=queue)
    second = a.enqueue(root, queue=queue)
    assert first == second and second["complete"]
    lines = [json.loads(line) for line in queue.read_text().splitlines()]
    assert len(lines) == 4 and queue.read_bytes().startswith(original)
    assert [j["id"].split(":")[-1] for j in lines[1:]] == ["null_s0", "null_s1", "null_s2"]
    assert lines[2]["after"] == [lines[1]["id"]]


def test_modified_baseline_result_is_not_accepted_after_freeze(frozen_nullspace):
    a, baseline_root, root, old = frozen_nullspace
    a.prepare(root, baseline_root=baseline_root, approved=True)
    Path(old["records"][1]["result_path"]).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        a.read_campaign(root)


def test_partial_enqueue_resumes_without_duplicate_or_changed_submission(frozen_nullspace, monkeypatch):
    a, baseline_root, root, _ = frozen_nullspace
    a.prepare(root, baseline_root=baseline_root, approved=True)
    queue = root/"queue.jsonl"
    queue.write_bytes(b"")
    queue.with_name("state.json").write_text(json.dumps({"done": [], "failed": [], "pids": {}}))
    append = a.append_queue
    def partial(path, job):
        if job["id"].endswith("null_s1"):
            raise OSError("injected append failure")
        return append(path, job)
    monkeypatch.setattr(a, "append_queue", partial)
    with pytest.raises(OSError, match="injected"):
        a.enqueue(root, queue=queue)
    monkeypatch.setattr(a, "append_queue", append)
    a.enqueue(root, queue=queue)
    assert len(queue.read_text().splitlines()) == 3
    manifest = root/"manifest.json"
    m = json.loads(manifest.read_text())
    m["rows"][-1]["n_atoms"] += 1
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="manifest|metadata|rows"):
        a.read_campaign(root)


def test_summary_ratios_use_paired_seed_means_and_exact_27_resamples(frozen_nullspace, monkeypatch):
    a, baseline_root, root, old = frozen_nullspace
    a.prepare(root, baseline_root=baseline_root, approved=True)
    candidates = {}
    for record in old["records"]:
        seed = record["seed"]
        r = copy.deepcopy(record["result"])
        for row in r["evaluation"]["rows"]:
            for metric in a.science.PRIMARY:
                row[metric] *= [.5, 1., 1.5][seed]
        candidates[seed] = r
        (root/f"null_s{seed}/result.json").write_text(json.dumps(r))
    monkeypatch.setattr(a, "checked_result", lambda manifest, job: candidates[job["seed"]])
    s = a.summarize(root)
    assert s["completed"] == 3
    metric = s["metrics"]["test/gap_error_meV"]
    assert metric["ratio_of_means"] == pytest.approx(1.)
    assert metric["paired_ratios"] == pytest.approx([.5, 1., 1.5])
    assert metric["paired_bootstrap_ratio_ci95"] == pytest.approx([.6083333333333333, 1.3916666666666666])
    assert metric["paired_seeds"] == [0, 1, 2] and metric["excluded_pairs"] == []
    bad = next(r for r in candidates[1]["evaluation"]["rows"] if r["split"] == "test")
    bad.update(spectral_valid=False, spectral_failure="controlled raw Hermiticity defect")
    invalid = a.summarize(root)["metrics"]["test/gap_error_meV"]
    assert invalid["paired_seeds"] == [0, 2] and invalid["paired_count"] == 2
    assert invalid["excluded_pairs"] == [{"seed": 1, "failures": [{"arm": "null",
        "sample_id": bad["sample_id"], "reason": "controlled raw Hermiticity defect"}]}]
    assert "paired_bootstrap_ratio_ci95" not in invalid


def test_completed_eval_and_run_reuse_without_touching_evidence(frozen_nullspace, monkeypatch):
    a, baseline_root, root, old = frozen_nullspace
    m = a.prepare(root, baseline_root=baseline_root, approved=True)
    job = m["jobs"][0]
    directory = Path(job["root"])
    paths = [directory/"result.json", directory/"evaluation_receipt.json", directory/"evaluation/physical-H-0.npz"]
    paths[-1].parent.mkdir()
    for p in paths:
        p.write_bytes(b"controlled existing scientific evidence")
    hashes = {str(p): a.sha256(p) for p in paths}
    calls = []
    result = old["records"][0]["result"]
    monkeypatch.setattr(a, "checked_result", lambda manifest, item: calls.append(item["key"]) or result)
    monkeypatch.setattr(a, "summarize", lambda root: None)
    def forbidden(*args, **kwargs):
        pytest.fail("completed evidence must not trigger fit or evaluation")
    monkeypatch.setattr(a.recovery, "guarded_fit", forbidden)
    import h2oclu_batching_metrics
    monkeypatch.setattr(h2oclu_batching_metrics, "evaluate_checkpoint", forbidden)
    for _ in range(2):
        assert a.evaluate(root, job["key"]) == result
        a.run(root, job["key"])
    assert calls == ["null_s0"]*4
    assert hashes == {str(p): a.sha256(p) for p in paths}


@pytest.mark.parametrize("partial", ["evaluation/physical-H-0.npz", "result.json", "evaluation_receipt.json"])
def test_partial_eval_is_preserved_not_overwritten(frozen_nullspace, monkeypatch, partial):
    a, baseline_root, root, _ = frozen_nullspace
    m = a.prepare(root, baseline_root=baseline_root, approved=True)
    job = m["jobs"][0]
    path = Path(job["root"])/partial
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"interrupted evidence")
    digest = a.sha256(path)
    def forbidden(*args, **kwargs):
        pytest.fail("partial evaluation must stop before fitting or replacing files")
    monkeypatch.setattr(a.recovery, "guarded_fit", forbidden)
    with pytest.raises(ValueError, match="partial|existing|incomplete"):
        a.evaluate(root, job["key"])
    with pytest.raises(ValueError, match="partial|existing|incomplete"):
        a.run(root, job["key"])
    assert a.sha256(path) == digest


@pytest.mark.parametrize("rotate", [False, True])
def test_projection_preserves_metric_padding_and_low_space(rotate):
    a = callback_api()
    U = np.eye(3)
    if rotate:
        U[:2, :2] = np.array([[1., -1.], [1., 1.]])/np.sqrt(2)
    S = U@np.diag([.002, .005, 1.])@U.T
    C = a.projection_columns(S, 4)
    assert C.shape == (3, 4)
    np.testing.assert_allclose(C[:, :2].T@S@C[:, :2], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(C[:, :2]@C[:, :2].T, U@np.diag([500., 200., 0.])@U.T, atol=1e-10)
    np.testing.assert_array_equal(C[:, 2:], np.zeros((3, 2)))


@pytest.mark.parametrize("S", [np.diag([-.001, 1.]), np.diag([.0005, 1.]),
                              np.array([[1., np.nan], [np.nan, 1.]]), np.array([[1., .1], [0., 1.]])])
def test_projection_rejects_invalid_or_floor_active_overlap(S):
    with pytest.raises(ValueError):
        callback_api().projection_columns(S, 2)


def test_projection_rejects_truncated_padding_capacity():
    with pytest.raises(ValueError):
        callback_api().projection_columns(np.diag([.002, .005, 1.]), 1)


def test_projection_uses_explicit_symmetric_overlap_after_raw_defect_gate():
    S = np.diag([.002, .005, 1.])
    S[0, 1], S[1, 0] = 1e-10, -1e-10
    C = callback_api().projection_columns(S, 3)
    np.testing.assert_allclose(C.T@((S+S.T)/2)@C, np.diag([1., 1., 0.]), atol=1e-12)


def test_resumed_cli_checkpoint_keeps_base_class_loadable():
    callback = callback_api().NullspaceTrainingCheckpoint()
    table = object()
    state = dict(basis_table=table, hyper_parameters={"basis_table": "Unable to serialize instance basis",
        "root_dir": "/frozen", "basis_files": "*.ion.xml", "loss_kwargs": {"spectral_weight": 1e-4}},
        datamodule_hyper_parameters={"basis_table": "Unable to serialize instance basis"})
    callback.on_save_checkpoint(None, None, state)
    assert state["basis_table"] is table
    assert state["hyper_parameters"] == {"basis_table": None, "root_dir": "/frozen",
        "basis_files": "*.ion.xml", "loss_kwargs": {"spectral_weight": 1e-4}}
    assert state["datamodule_hyper_parameters"]["basis_table"] is None


def test_real_trainer_keeps_basic_validation_and_restores_train_weight(tmp_path):
    import pytorch_lightning as pl
    from torch_geometric.loader import DataLoader
    from graph2mat.tools.lightning.tests.test_gauge_epoch_metrics import gauge_model, gauge_samples
    a = callback_api()
    model = gauge_model(spectral_weight=1e-4)
    samples = gauge_samples()
    callback = a.NullspaceTrainingCheckpoint(dirpath=tmp_path/"best", monitor="val_loss", mode="min")
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[callback])
    for size in (1, 2, 3):
        metrics = trainer.validate(model, dataloaders=DataLoader(samples, batch_size=size), verbose=False)[0]
        assert metrics["val_loss"] == pytest.approx(38/8)
        assert model.loss_fn.metric_kwargs["spectral_weight"] == 1e-4
        assert model.hparams.loss_kwargs["spectral_weight"] == 1e-4
    from torch_geometric.data import Batch
    b = Batch.from_data_list(samples)
    loss, stats = model.loss_fn(nodes_pred=b.node_prediction, nodes_ref=b.point_labels,
        edges_pred=b.edge_prediction, edges_ref=b.edge_labels, batch=b, basis_table=model.basis_table)
    assert float(loss) == pytest.approx(38/8+1e-4*38/21)
    assert "spectral_rmse" in stats


def test_validation_exception_restores_weight_without_mutating_hparams(tmp_path):
    from graph2mat.tools.lightning.tests.test_gauge_epoch_metrics import gauge_model
    a = callback_api()
    model = gauge_model(spectral_weight=1e-4)
    cb = a.NullspaceTrainingCheckpoint(dirpath=tmp_path/"best", monitor="val_loss", mode="min")
    cb.on_validation_start(SimpleNamespace(), model)
    assert model.loss_fn.metric_kwargs["spectral_weight"] == 0.
    assert model.hparams.loss_kwargs["spectral_weight"] == 1e-4
    cb.on_exception(SimpleNamespace(), model, RuntimeError("interrupted"))
    assert model.loss_fn.metric_kwargs["spectral_weight"] == 1e-4
    cb.on_exception(SimpleNamespace(), model, RuntimeError("again"))
    assert model.loss_fn.metric_kwargs["spectral_weight"] == 1e-4


def test_callback_rejects_wrong_loss_before_zeroing_weight(tmp_path):
    from graph2mat.tools.lightning.tests.test_gauge_epoch_metrics import gauge_model
    from graph2mat.core.data.metrics import elementwise_mse
    model = gauge_model(spectral_weight=1e-4)
    model.loss_fn = elementwise_mse()
    cb = callback_api().NullspaceTrainingCheckpoint(dirpath=tmp_path/"best", monitor="val_loss", mode="min")
    with pytest.raises(ValueError):
        cb.on_validation_start(SimpleNamespace(), model)


def test_native_recovery_resume_preserves_basic_best_and_train_weight(tmp_path):
    import pytorch_lightning as pl
    import run_cost_batching_comparison as runner
    from graph2mat.tools.lightning.tests.test_gauge_epoch_metrics import gauge_model, gauge_samples
    a, recovery = callback_api(), runner.recovery_api()
    store = recovery.RecoveryStore(tmp_path/"job", {"selection": "plain gauge"})
    for resume in (False, True):
        pl.seed_everything(7, workers=True)
        model = gauge_model(spectral_weight=1e-4)
        module = recovery.RecoveryDataModule(basis_table=model.basis_table, batch_size=2,
            store_in_memory=True, loader_threads=0, n_matrix_components=2, matrix_component_policy="h_and_overlap")
        module.train_dataset, module.val_dataset = gauge_samples(), gauge_samples()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        model.configure_optimizers = lambda: {"optimizer": optimizer, "lr_scheduler": schedule}
        best = a.NullspaceTrainingCheckpoint(dirpath=store.root/"best", monitor="val_loss", mode="min")
        callback = recovery.RecoveryCallback(store, 7, ["a", "b", "c"])
        class Stop(pl.Callback):
            def on_train_batch_start(self, trainer, model, batch, batch_idx):
                if not resume and trainer.current_epoch == 1:
                    raise InterruptedError("owned test interruption after committed epoch")
        checkpoint = None
        if resume:
            receipt = store.latest()
            store.restore_best(receipt)
            checkpoint = receipt["checkpoint"]
        trainer = pl.Trainer(accelerator="cpu", max_epochs=2,
            logger=False, enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=1, callbacks=[Stop(), best, callback])
        if resume:
            trainer.fit(model, datamodule=module, ckpt_path=checkpoint, weights_only=False)
        else:
            with pytest.raises(InterruptedError, match="owned test interruption"):
                trainer.fit(model, datamodule=module)
        assert model.loss_fn.metric_kwargs["spectral_weight"] == 1e-4
        assert float(best.best_model_score) == pytest.approx(38/8)
        assert store.latest()["next_epoch"] == (2 if resume else 1)
        state = torch.load(store.latest()["checkpoint"], map_location="cpu", weights_only=False)
        assert state["hyper_parameters"]["loss_kwargs"]["spectral_weight"] == 1e-4
        module.teardown(None)
