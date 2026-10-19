"""Exercise supervisor recovery with real owned CPU children, without GPU allocations."""
import importlib
import hashlib
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import torch

from test_training_recovery import FINGERPRINT, SavedTrainer, boundary, recovery


def runner():
    assert Path(__file__).with_name("run_cost_batching_comparison.py").is_file(), "recovery runner is missing"
    return importlib.import_module("run_cost_batching_comparison")


def prepare_job(tmp_path, monkeypatch, epochs=1):
    api = runner()
    monkeypatch.setattr(api, "REPO", tmp_path)
    root = tmp_path / "auditorias_mejoras/campaign/seed0"
    root.mkdir(parents=True)
    config = root / "config.yaml"
    config.write_text(f"seed_everything: 7\ntrainer:\n  max_epochs: {epochs}\n")
    job = {"fingerprint": FINGERPRINT, "config": str(config), "max_epochs": epochs,
           "sample_ids": [str(index) for index in range(12)], "files": {str(config): api.sha256(config)}}
    (root / "job.json").write_text(json.dumps(job))
    return api, root, config


def publish_complete(root):
    store = recovery().RecoveryStore(root, FINGERPRINT)
    best = root / "best/early.ckpt"
    best.parent.mkdir()
    torch.save({"state_dict": {"weight": torch.tensor([1.])}}, best)
    trainer = SavedTrainer()
    trainer.payload["callbacks"]["best"]["best_model_path"] = str(best)
    store.publish(trainer, boundary=boundary(), best_path=best)
    return store


def test_final_commit_without_fit_json_finishes_without_touching_gpu(tmp_path, monkeypatch):
    api, root, config = prepare_job(tmp_path, monkeypatch)
    store = publish_complete(root)
    def forbidden(*args, **kwargs):
        pytest.fail("completed generation must not launch or inspect GPU")
    monkeypatch.setattr(api, "gpu_processes", forbidden)
    monkeypatch.setattr(api.pipeline, "gpu_state", forbidden)
    monkeypatch.setattr(api.pipeline, "guarded_command", forbidden)
    result = api.guarded_fit(config, root)
    assert result["finalized_from_generation"] is True
    assert result["steps"] == 4 and result["samples_committed"] == 12
    assert result["checkpoint"] == store.latest()["best"]["path"]
    assert json.loads((root / "fit.json").read_text())["checkpoint"] == result["checkpoint"]


def test_completed_pilot_keeps_matching_measurements_without_retraining(tmp_path, monkeypatch):
    api, root, config = prepare_job(tmp_path, monkeypatch)
    store = publish_complete(root)
    previous = dict(recovery_checkpoint=store.latest()["checkpoint"],
                    peak_allocated_mib=123., peak_reserved_mib=256., steps=999)
    (root / "fit.json").write_text(json.dumps(previous))
    def forbidden(*args, **kwargs):
        pytest.fail("a completed pilot must not retrain or inspect GPU")
    monkeypatch.setattr(api.pipeline, "gpu_state", forbidden)
    result = api.guarded_fit(config, root, pilot=True)
    assert result["peak_allocated_mib"] == 123.
    assert result["peak_reserved_mib"] == 256.
    assert result["steps"] == 4  # Native state, never the telemetry's untrusted count.


@pytest.mark.parametrize("observed, expected", [(None, 0), ({"steps": 6, "samples_observed": 18}, 6)])
def test_accounting_reports_bounds_not_fictitious_exact_repeated_work(observed, expected):
    api = runner()
    result = api.attempt_accounting(None, observed, boundary())
    assert result["repeated_samples_lower_bound"] == expected
    assert result["repeated_samples_upper_bound"] is None
    assert result["confirmed_samples_end"] == 12


def test_changed_input_hash_blocks_before_gpu_work(tmp_path, monkeypatch):
    api, root, config = prepare_job(tmp_path, monkeypatch)
    config.write_text("different recipe")
    def forbidden(*args, **kwargs):
        pytest.fail("GPU was inspected after frozen inputs changed")
    monkeypatch.setattr(api, "gpu_processes", forbidden)
    with pytest.raises(ValueError, match="changed|hash"):
        api.guarded_fit(config, root)


def test_foreign_output_root_is_not_created(tmp_path):
    api = runner()
    foreign = tmp_path / "foreign"
    with pytest.raises(ValueError, match="Graph2Mat|output|own"):
        api.guarded_fit(tmp_path / "config.yaml", foreign)
    assert not foreign.exists()


def test_changed_software_version_blocks_before_loading_recovery(tmp_path, monkeypatch):
    api, root, config = prepare_job(tmp_path, monkeypatch)
    job = json.loads((root / "job.json").read_text())
    job["versions"] = {"torch": "not-the-installed-version"}
    (root / "job.json").write_text(json.dumps(job))
    with pytest.raises(ValueError, match="version"):
        api.read_job(config, root)


@pytest.mark.parametrize("yields", [1, 2])
def test_supervisor_resumes_owned_child_and_preserves_all_attempts(tmp_path, monkeypatch, yields):
    api, root, config = prepare_job(tmp_path, monkeypatch, epochs=yields + 1)
    child = tmp_path / "controlled_child.py"
    child.write_text('''import argparse, json, os, time
from pathlib import Path
import bench_pipeline_gpu as pipeline
pipeline.compiled_extensions()
from test_training_recovery import SavedTrainer, boundary, FINGERPRINT
from training_recovery import RecoveryStore
from run_readout_comparison import write_json
p=argparse.ArgumentParser()
p.add_argument('command')
for key in ('root','config','attempt','fraction','checkpoint'): p.add_argument('--'+key)
a=p.parse_args()
root, attempt=Path(a.root), Path(a.attempt)
write_json(attempt/'pid.json', {'pid':os.getpid(), 'pgid':os.getpgid(0)})
store=RecoveryStore(root,FINGERPRINT)
trainer=SavedTrainer()
budget=json.loads((root/'job.json').read_text())['max_epochs']
next_epoch=1 if a.checkpoint is None else store.latest()['next_epoch']+1
trainer.advance(boundary(next_epoch))
if next_epoch < budget:
    store.publish(trainer,boundary=boundary(next_epoch),best_path=None)
    write_json(attempt/'progress.json',{'steps':4*next_epoch+2,'samples_observed':12*next_epoch+6,'time':time.time()})
    time.sleep(30)
else:
    assert Path(a.checkpoint).name=='state.ckpt'
    assert store.latest()['global_step']==4*(budget-1)
    best=root/'best/early.ckpt'
    best.parent.mkdir()
    trainer.payload['callbacks']['best']['best_model_path']=str(best)
    trainer.save_checkpoint(best)
    store.publish(trainer,boundary=boundary(next_epoch),best_path=best)
    write_json(attempt/'progress.json',{'steps':4*next_epoch,'samples_observed':12*next_epoch,'time':time.time()})
    write_json(attempt/'fit.json',{'checkpoint':str(best),'steps':4*next_epoch})
''')
    monkeypatch.setattr(api, "__file__", str(child))
    state = {"memory.total": 32000, "memory.free": 31000, "temperature.gpu": 50}
    original_state = lambda: dict(state)
    monkeypatch.setattr(api.pipeline, "gpu_state", original_state)
    yielded = []
    def foreign(own):
        if len(yielded) < yields and len(list(root.glob("attempt-*/progress.json"))) > len(yielded):
            yielded.append(True)
            return {999999}
        return set()
    monkeypatch.setattr(api, "gpu_processes", foreign)
    killpg = os.killpg
    signals = []
    def owned_signal(group, value):
        assert group != 999999, "foreign process must never receive a signal"
        signals.append((group, value))
        return killpg(group, value)
    monkeypatch.setattr(api.pipeline.os, "killpg", owned_signal)
    result = api.guarded_fit(config, root)
    assert len(result["attempts"]) == yields + 1 and yielded == [True] * yields
    first, second = result["attempts"][:2]
    assert first["exit_reason"] == "foreign_gpu"
    assert first["repeated_samples_lower_bound"] == 6
    assert second["resume_checkpoint"].endswith("/state.ckpt")
    assert result["seconds_all_attempts"] >= result["seconds_successful_fit"]
    assert api.pipeline.gpu_state is original_state
    records = [json.loads(path.read_text()) for path in root.glob("attempt-*/pid.json")]
    assert len(records) == yields + 1
    for record in records:
        assert record["pid"] == record["pgid"]
        with pytest.raises(ProcessLookupError):
            os.kill(record["pid"], 0)
    assert all(group in {record["pgid"] for record in records} for group, _ in signals)
    assert any(value == signal.SIGTERM for _, value in signals)


def test_resource_error_is_recorded_and_does_not_retry(tmp_path, monkeypatch):
    api, root, config = prepare_job(tmp_path, monkeypatch)
    monkeypatch.setattr(api, "gpu_processes", lambda own: set())
    original = lambda: {"memory.total": 32000, "memory.free": 31000, "temperature.gpu": 50}
    monkeypatch.setattr(api.pipeline, "gpu_state", original)
    def failed(*args, **kwargs):
        raise RuntimeError("resource guard")
    monkeypatch.setattr(api.pipeline, "guarded_command", failed)
    with pytest.raises(RuntimeError, match="resource guard"):
        api.guarded_fit(config, root)
    attempts = json.loads((root / "attempts.json").read_text())
    assert len(attempts) == 1 and attempts[0]["exit_reason"] == "error"
    assert api.pipeline.gpu_state is original


def test_cancellation_after_final_commit_is_recorded_and_can_finalize(tmp_path, monkeypatch):
    api, root, config = prepare_job(tmp_path, monkeypatch)
    monkeypatch.setattr(api, "gpu_processes", lambda own: set())
    monkeypatch.setattr(api.pipeline, "gpu_state", lambda: {
        "memory.total": 32000, "memory.free": 31000, "temperature.gpu": 50})
    def interrupted(*args, **kwargs):
        publish_complete(root)
        raise KeyboardInterrupt("operator cancellation")
    monkeypatch.setattr(api.pipeline, "guarded_command", interrupted)
    with pytest.raises(KeyboardInterrupt):
        api.guarded_fit(config, root)
    attempts = json.loads((root / "attempts.json").read_text())
    assert len(attempts) == 1 and attempts[0]["exit_reason"] == "cancelled"
    result = api.guarded_fit(config, root)
    assert result["steps"] == 4 and result["samples_committed"] == 12
    assert len(result["attempts"]) == 1 and result["seconds_successful_fit"] == 0


@pytest.mark.parametrize("unavailable", ["recovery", "heartbeat"])
def test_accounting_failure_does_not_mask_child_error_or_lose_attempt(tmp_path, monkeypatch, unavailable):
    api, root, config = prepare_job(tmp_path, monkeypatch, epochs=2)
    store = publish_complete(root)
    checkpoint = Path(store.latest()["checkpoint"])
    monkeypatch.setattr(api, "gpu_processes", lambda own: set())
    monkeypatch.setattr(api.pipeline, "gpu_state", lambda: {
        "memory.total": 32000, "memory.free": 31000, "temperature.gpu": 50})
    def failed(command, log, *args):
        if unavailable == "recovery":
            checkpoint.write_bytes(b"truncated")
        else:
            (Path(log).parent / "progress.json").write_text("truncated")
        raise RuntimeError("original child failure")
    monkeypatch.setattr(api.pipeline, "guarded_command", failed)
    with pytest.raises(RuntimeError, match="original child failure"):
        api.guarded_fit(config, root)
    rows = json.loads((root / "attempts.json").read_text())
    assert len(rows) == 1 and rows[0]["exit_reason"] == "error"
    assert "original child failure" in rows[0]["reason"]
    assert rows[0]["accounting_errors"]
    if unavailable == "recovery":
        assert rows[0]["confirmed_samples_end"] is None
        assert rows[0]["repeated_samples_lower_bound"] is None


def test_completed_generation_handles_old_attempt_without_terminal_reason(tmp_path, monkeypatch):
    api, root, config = prepare_job(tmp_path, monkeypatch)
    publish_complete(root)
    (root / "attempts.json").write_text(json.dumps([dict(seconds=2., repeated_samples_lower_bound=0)]))
    result = api.guarded_fit(config, root)
    assert result["steps"] == 4
    assert result["attempts"][0]["exit_reason"] == "unknown"
    assert result["seconds_successful_fit"] == 0


def test_supervisor_records_corrupt_checkpoint_fallback(tmp_path, monkeypatch):
    api, root, config = prepare_job(tmp_path, monkeypatch, epochs=3)
    store = recovery().RecoveryStore(root, FINGERPRINT)
    trainer = SavedTrainer()
    first = store.publish(trainer, boundary=boundary(), best_path=None)
    trainer.advance(boundary(2))
    second = store.publish(trainer, boundary=boundary(2), best_path=None)
    (second / "state.ckpt").write_bytes(b"broken")
    monkeypatch.setattr(api, "gpu_processes", lambda own: set())
    monkeypatch.setattr(api.pipeline, "gpu_state", lambda: {
        "memory.total": 32000, "memory.free": 31000, "temperature.gpu": 50})
    def stopped(*args, **kwargs):
        raise RuntimeError("controlled stop")
    monkeypatch.setattr(api.pipeline, "guarded_command", stopped)
    with pytest.raises(RuntimeError, match="controlled stop"):
        api.guarded_fit(config, root)
    attempt = json.loads((root / "attempts.json").read_text())[0]
    assert Path(attempt["resume_checkpoint"]).parent == first
    assert attempt["recovery_fallback"][0]["generation"] == second.name


@pytest.mark.parametrize("accelerator", ["cpu", "gpu"])
def test_fit_resumes_real_mace_through_cli_with_explicit_overrides(tmp_path, monkeypatch, accelerator):
    if accelerator == "gpu" and os.environ.get("G2M_GPU_GUARDED") != "1":
        pytest.skip("explicit guarded GPU certification required")
    api, root, config = prepare_job(tmp_path, monkeypatch, epochs=2)
    assert hasattr(api, "fit"), "native CLI recovery fit is missing"
    import sisl
    import yaml
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from graph2mat import AtomicTableWithEdges
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    recover = recovery()

    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1.4)])
    basis = AtomicTableWithEdges([atom])
    geometry = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=atom, lattice=[10.] * 3)
    matrix = sisl.DensityMatrix(geometry)
    matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0] = 1., 2., .25, .25
    matrix.write(root / "sample.nc")
    data = dict(out_matrix="density_matrix", root_dir=str(root), symmetric_matrix=True,
        sub_point_matrix=False, loader_threads=0, store_in_memory=True,
        train_runs="sample.nc", val_runs="sample.nc", batch_size=1)
    model_options = dict(hidden_irreps="2x0e+2x1o", edge_hidden_irreps="2x0e+2x1o",
        num_bessel=2, max_ell=1, num_interactions=2, correlation=1, optim_lr=.002)
    cfg = {"seed_everything": 7, "data": data, "model": model_options,
        "trainer": {"accelerator": accelerator, "devices": 1, "logger": False, "max_epochs": 2,
                    "num_sanity_val_steps": 0, "enable_progress_bar": False},
        "lr_scheduler": {"class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
                         "init_args": {"T_max": 2}}}
    config.write_text(yaml.safe_dump(cfg))
    job = {"fingerprint": FINGERPRINT, "config": str(config), "max_epochs": 2,
        "sample_ids": ["tiny"], "files": {str(config): api.sha256(config),
                                          str(root / "sample.nc"): api.sha256(root / "sample.nc")}}
    (root / "job.json").write_text(json.dumps(job))
    pl.seed_everything(7, workers=True)
    module = recover.RecoveryDataModule(basis_table=basis, **data)
    model = LitMACEMatrixModel(basis_table=basis, symmetric_matrix=True, **model_options)
    optimizer = model.configure_optimizers()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    model.configure_optimizers = lambda: {"optimizer": optimizer, "lr_scheduler": scheduler}
    store = recover.RecoveryStore(root, FINGERPRINT)
    best = ModelCheckpoint(dirpath=root / "best", monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=2, max_steps=1, logger=False,
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
        callbacks=[best, recover.RecoveryCallback(store, 7, ["tiny"])])
    trainer.fit(model, datamodule=module)
    first = store.latest()
    assert first["next_epoch"] == 1
    module.teardown(None)
    attempt = root / "attempt-cli"
    attempt.mkdir()
    api.fit(SimpleNamespace(root=root, config=config, attempt=attempt, fraction=.1,
                            checkpoint=Path(first["checkpoint"])))
    final = store.latest()
    assert final["next_epoch"] == 2 and final["global_step"] == 2
    state = torch.load(final["checkpoint"], map_location="cpu", weights_only=False)
    assert state["hyper_parameters"]["optim_lr"] == .002
    assert state["datamodule_hyper_parameters"]["batch_size"] == 1
    assert state["lr_schedulers"][0]["last_epoch"] == 2
    assert [group["name"] for group in state["optimizer_states"][0]["param_groups"]] == [
        "embedding", "interactions_decay", "interactions_no_decay", "products", "readouts"]
    info = json.loads((attempt / "fit.json").read_text())
    assert Path(info["checkpoint"]).parent == root / "best"
    assert info["samples_committed"] == 2
    assert info["checkpoint_seconds"] > 0 and info["train_batches_seconds"] > 0
    assert len(info["epochs"]) == 1
    assert info["epochs"][0]["epoch"] == 1
    assert info["epochs"][0]["samples"] == info["epochs"][0]["updates"] == 1
    assert info["epochs"][0]["seconds"] > 0
    if accelerator == "gpu":
        assert next(model.parameters()).device.type == "cpu"  # Native Lightning teardown happened.
        assert info["peak_allocated_mib"] > 0
        assert info["peak_reserved_mib"] >= info["peak_allocated_mib"]


@pytest.fixture
def tiny_campaign(tmp_path, monkeypatch):
    """Real molecular files and native MACE, not an external scientific dataset."""
    import sisl
    import yaml
    api = runner()
    monkeypatch.setattr(api, "REPO", tmp_path)
    source = tmp_path / "inputs"
    source.mkdir()
    xml = '''<ion><label>H</label><z>1</z><mass>1.0</mass><paos>
<orbital n="1" l="0" z="1" population="1" ispol="0"><radfunc>
<npts>4</npts><delta>1</delta><cutoff>3</cutoff><data>0 1 1 0.5 2 0.1 3 0</data>
</radfunc></orbital></paos></ion>'''
    basis = source / "H.ion.xml"
    basis.write_text(xml)
    atom = sisl.get_sile(basis).read_basis()
    geometry = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=atom, lattice=[10.] * 3)
    matrix = sisl.Hamiltonian(geometry, orthogonal=False)
    matrix[0, 0], matrix[1, 1] = [-1., 1.], [1., 1.]
    matrix[0, 1], matrix[1, 0] = [.2, .1], [.2, .1]
    rows = []
    for split, count in (("train", 32), ("validation", 4), ("test", 3)):
        for index in range(count):
            folder = source / "g2m_improvements/bench/h2oclu" / split / f"sample{index:02d}"
            folder.mkdir(parents=True)
            geometry.write(folder / "RUN.fdf")
            matrix.write(folder / "matrix.nc")
            (folder / "RUN.out").write_text("Version         : fixture-1\nSCF Convergence by FreeE+DM+H criterion\n")
            (folder / "0_NORMAL_EXIT").touch()
            (folder / "H.ion.xml").write_text(xml)
            (folder / "H.psf").write_text("fixture pseudo")
            key = folder.name + "_" + hashlib.sha1(str(folder.resolve()).encode()).hexdigest()[:8]
            for target in ("h0", "delta") if split != "test" else ("h0",):
                dest = source / "g2m_improvements/delta_h0" / target / key
                dest.mkdir(parents=True)
                label = matrix.copy()
                if target == "delta":
                    label._csr._D[:, 0] = 0
                label.write(dest / "matrix.nc")
                geometry.write(dest / "RUN.fdf")
            rows.append(dict(sample_id=f"{split}{index:02d}", split=split,
                n_atoms=2, net_charge=0, kpts=[1, 1, 1],
                structure_path=str(folder / "RUN.fdf"), reference_tshs_path=str(folder / "matrix.nc")))
    manifest = source / "g2m_improvements/bench/h2oclu/frozen_split_manifest.json"
    manifest.write_text(json.dumps(dict(material="h2oclu", rows=rows)))
    for seed in (0, 1, 2):
        run = source / "runs/bench/h2oclu" / f"h2oclu__ts{seed}__g2mv2_bench_delta"
        run.mkdir(parents=True)
        cfg = dict(seed_everything=seed,
            data=dict(out_matrix="hamiltonian", basis_files=os.path.relpath(basis, run), batch_size=8,
                symmetric_matrix=True, sub_point_matrix=False, store_in_memory=True, loader_threads=0,
                n_matrix_components=2, matrix_component_policy="h_and_overlap",
                train_runs=[], val_runs=[]),
            model=dict(hidden_irreps="2x0e+2x1o", edge_hidden_irreps="2x0e+2x1o",
                num_bessel=2, max_ell=1, num_interactions=2, correlation=1,
                n_output_components=1, loss="graph2mat.metrics.gauge_invariant_mse", optim_lr=.0018),
            trainer=dict(accelerator="gpu", devices=1, logger=False, max_epochs=2000,
                precision="32-true", check_val_every_n_epoch=5),
            lr_scheduler=dict(class_path="torch.optim.lr_scheduler.CosineAnnealingLR",
                              init_args=dict(T_max=2000, eta_min=1e-5)))
        for row in rows:
            if row["split"] == "test":
                continue
            folder = Path(row["structure_path"]).parent
            key = folder.name + "_" + hashlib.sha1(str(folder.resolve()).encode()).hexdigest()[:8]
            cfg["data"]["train_runs" if row["split"] == "train" else "val_runs"].append(
                str(source / "g2m_improvements/delta_h0/delta" / key / "matrix.nc"))
        (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    return api, source, tmp_path / "auditorias_mejoras/campaign"


def test_prepare_is_readonly_and_freezes_full_paired_initializations(tiny_campaign):
    api, source, root = tiny_campaign
    assert hasattr(api, "prepare"), "campaign preparation is missing"
    before = {str(p): api.sha256(p) for p in source.rglob("*") if p.is_file()}
    result = api.prepare(root, source_base=source)
    assert before == {str(p): api.sha256(p) for p in source.rglob("*") if p.is_file()}
    assert result["split_counts"] == {"train": 32, "validation": 4, "test": 3}
    assert len(result["pilots"]) == 6 and result["scientific_authorized"] is False
    for seed in (0, 1, 2):
        initial = result["initial_states"][str(seed)]
        assert initial["fixed_sha256"] == initial["cost_sha256"]
        assert torch.load(initial["path"], weights_only=True)
    for pilot in result["pilots"]:
        job, cfg = api.read_job(Path(pilot["config"]), Path(pilot["root"]))
        assert job["max_epochs"] == cfg["trainer"]["max_epochs"] == 5
        assert cfg["lr_scheduler"]["init_args"]["T_max"] == 2000
    first = result["pilots"][0]
    Path(next(iter(before))).write_text("corrupt external input")
    with pytest.raises(ValueError, match="changed|hash"):
        api.read_job(Path(first["config"]), Path(first["root"]))


@pytest.mark.parametrize("missing", ["reference", "h0", "delta", "basis", "version"])
def test_prepare_rejects_incomplete_inputs_before_creating_output(tiny_campaign, missing):
    api, source, root = tiny_campaign
    assert hasattr(api, "prepare"), "campaign preparation is missing"
    if missing == "version":
        next(source.glob("g2m_improvements/bench/h2oclu/train/*/RUN.out")).write_text("SCF Convergence\n")
    else:
        pattern = {"reference": "g2m_improvements/bench/h2oclu/train/*/matrix.nc",
            "h0": "g2m_improvements/delta_h0/h0/*/matrix.nc",
            "delta": "g2m_improvements/delta_h0/delta/*/matrix.nc", "basis": "H.ion.xml"}[missing]
        next(source.glob(pattern)).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        api.prepare(root, source_base=source)
    assert not root.exists()


def test_pilot_budget_selection_requires_both_rounds_and_no_ood():
    api = runner()
    assert hasattr(api, "select_pilot_budget"), "pilot budget selection is missing"
    rows = [dict(round=r, budget=b, peak_allocated_mib=m)
            for r, values in enumerate(((None, 100), (50, 80), (75, 90),
                                         (None, 100), (75, 101), (50, 81)))
            for b, m in [values]]
    # Round is paired, not global order; a larger budget that regresses once loses.
    for row in rows:
        row["round"] //= 3
    assert api.select_pilot_budget(rows) == {"budget": 50, "reduced_both_rounds": True}
    rows[1]["peak_allocated_mib"] = 102
    assert api.select_pilot_budget(rows) == {"budget": 50, "reduced_both_rounds": False}


def test_reference_species_come_from_fdf_not_tshs_placeholder_atoms(tiny_campaign):
    import sisl
    api, source, _ = tiny_campaign
    row = json.loads((source / "g2m_improvements/bench/h2oclu/frozen_split_manifest.json").read_text())["rows"][0]
    reference = Path(row["reference_tshs_path"])
    folder = reference.parent
    key = folder.name + "_" + hashlib.sha1(str(folder.resolve()).encode()).hexdigest()[:8]
    for path in (reference, *(source / "g2m_improvements/delta_h0" / target / key / "matrix.nc"
                              for target in ("h0", "delta"))):
        with sisl.get_sile(path) as handle:
            matrix = handle.read_hamiltonian()
        matrix.geometry.atoms[:] = sisl.Atom("He", orbitals=matrix.geometry.atoms[0].orbitals)
        matrix.write(path)
    rows, _, _, assets = api.existing_inputs(source)
    assert len(rows) == 39
    assert "H.ion.xml" in assets and "He.ion.xml" not in assets


def test_historical_split_link_retargeting_blocks_before_recovery_or_gpu(tiny_campaign):
    import yaml
    api, source, root = tiny_campaign
    # Historical recipes use globbed directory links, not the canonical label paths.
    for seed in (0, 1, 2):
        run = source / "runs/bench/h2oclu" / f"h2oclu__ts{seed}__g2mv2_bench_delta"
        config = run / "config.yaml"
        cfg = yaml.safe_load(config.read_text())
        for argument in ("train_runs", "val_runs"):
            directory = run / argument
            directory.mkdir()
            for index, label in enumerate(cfg["data"][argument]):
                (directory / f"{index:03d}").symlink_to(Path(label).parent, target_is_directory=True)
            cfg["data"][argument] = f"{argument}/*/matrix.nc"
        config.write_text(yaml.safe_dump(cfg))
    manifest = api.prepare(root, source_base=source)
    link = source / "runs/bench/h2oclu/h2oclu__ts0__g2mv2_bench_delta/train_runs/000"
    link.unlink()
    link.symlink_to(source / "g2m_improvements/bench/h2oclu/test/sample00", target_is_directory=True)
    # Every frozen file is unchanged: the reference is already among audited inputs,
    # but this now trains on physical H instead of the frozen delta target.
    api.verify_files(manifest["files"])
    job = manifest["pilots"][0]
    with pytest.raises(ValueError, match="resolution|input.*changed|order"):
        api.read_job(Path(job["config"]), Path(job["root"]))
