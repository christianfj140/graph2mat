"""Scientific campaign boundaries; external data and GPU never needed here."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from test_cost_batching_comparison import tiny_campaign  # noqa: F401, F811


def science():
    path = Path(__file__).with_name("run_h2oclu_batching_science.py")
    assert path.is_file(), "approved science launcher is missing"
    return importlib.import_module("run_h2oclu_batching_science")


@pytest.fixture
def frozen_pilot(tmp_path, monkeypatch):
    api = science()
    monkeypatch.setattr(api.recovery, "REPO", tmp_path)
    pilot = tmp_path / "auditorias_mejoras/pilot"
    pilot.mkdir(parents=True)
    input_file = pilot / "input.txt"
    input_file.write_text("frozen")
    import torch
    api.recovery.recovery_api()
    from readout_comparison_callback import tensor_hash
    rows = []
    for split,count in (("train",32),("validation",4),("test",3)):
        for index in range(count):
            delta = pilot / f"{split}{index}.nc"
            delta.write_text(f"{split}-{index}")
            rows.append(dict(sample_id=f"{split}{index}",split=split,delta=str(delta)))
    train_links = []
    for row in rows[:32]:
        link = pilot / (row["sample_id"]+"-link.nc")
        link.symlink_to(row["delta"])
        train_links.append(str(link))
    recipes, initials = {}, {}
    for seed in range(3):
        initial = pilot / f"initial-s{seed}.pt"
        state = {"weight":torch.tensor([float(seed)])}
        torch.save(state,initial)
        digest = tensor_hash(state.items())
        initials[str(seed)] = dict(path=str(initial), fixed_sha256=digest, cost_sha256=digest)
        recipes[str(seed)] = dict(seed_everything=seed, data=dict(batch_size=8, max_batch_cost=None,
            root_dir=str(pilot),train_runs=train_links,val_runs=[r["delta"] for r in rows[32:36]]),
            model=dict(optim_lr=.0018), trainer=dict(max_epochs=2000, check_val_every_n_epoch=5),
            lr_scheduler=dict(class_path="torch.optim.lr_scheduler.CosineAnnealingLR", init_args=dict(T_max=2000, eta_min=1e-5)))
    frozen_job = pilot / "pilot-r0-fixed/job.json"
    api.write_json(frozen_job,dict(input_resolution=api.recovery.input_resolution(recipes["0"])))
    manifest = dict(files={str(input_file): api.sha256(input_file)}, versions={}, recipes=recipes,
        sample_ids=[r["sample_id"] for r in rows[:32]], initial_states=initials, rows=rows,
        pilots=[dict(root=str(frozen_job.parent))],
        split_counts=dict(train=32, validation=4, test=3))
    api.write_json(pilot / "manifest.json", manifest)
    api.write_json(pilot / "pilot.json", dict(selected=dict(budget=326283, reduced_both_rounds=True)))
    return api, pilot, tmp_path / "auditorias_mejoras/science"


def test_no_authorization_creates_no_scientific_directory(frozen_pilot):
    api, pilot, root = frozen_pilot
    with pytest.raises(ValueError, match="authoriz"):
        api.prepare(root, pilot)
    assert not root.exists()


def test_prepare_freezes_exact_six_paired_jobs(frozen_pilot):
    api, pilot, root = frozen_pilot
    manifest = api.prepare(root, pilot, approved=True)
    assert {(j["arm"], j["seed"]) for j in manifest["jobs"]} == {(a,s) for a in ("fixed","cost") for s in range(3)}
    assert len(manifest["jobs"]) == 6
    initial_by_seed = {}
    for job in manifest["jobs"]:
        native, cfg = api.recovery.read_job(Path(job["config"]), Path(job["root"]))
        assert cfg["trainer"]["max_epochs"] == 2000
        assert cfg["data"]["max_batch_cost"] == (None if job["arm"] == "fixed" else 326283)
        assert cfg["lr_scheduler"]["init_args"]["T_max"] == 2000
        assert len(native["sample_ids"]) * native["max_epochs"] == 64000
        assert native["scientific_authorized"] and not native["technical_pilot"]
        initial_by_seed.setdefault(job["seed"], native["initial_sha256"])
        assert initial_by_seed[job["seed"]] == native["initial_sha256"]
    api.read_campaign(root)
    config = Path(manifest["jobs"][0]["config"])
    config.write_text(yaml.safe_dump({"changed": True}))
    with pytest.raises(ValueError, match="changed"):
        api.read_campaign(root)


def test_changed_pilot_input_blocks_before_output(frozen_pilot):
    api, pilot, root = frozen_pilot
    (pilot / "input.txt").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        api.prepare(root, pilot, approved=True)
    assert not root.exists()


@pytest.mark.parametrize("change",["split_link","initial","later_recipe"])
def test_preparation_rejects_pilot_identity_changes_before_any_output(frozen_pilot,change):
    api,pilot,root = frozen_pilot
    manifest = json.loads((pilot/"manifest.json").read_text())
    if change == "split_link":
        link = pilot/"train0-link.nc"
        link.unlink()
        link.symlink_to(pilot/"train1.nc")
    elif change == "initial":
        import torch
        torch.save({"weight":torch.tensor([999.])},pilot/"initial-s2.pt")
    else:
        manifest["recipes"]["2"]["trainer"]["max_epochs"] = 1999
        api.write_json(pilot/"manifest.json",manifest)
    with pytest.raises(ValueError,match="resolution|initial|recipe|split"):
        api.prepare(root,pilot,approved=True)
    assert not root.exists()


def test_enqueue_reuses_receipt_after_partial_append(frozen_pilot, monkeypatch):
    api, pilot, root = frozen_pilot
    api.prepare(root, pilot, approved=True)
    queue = root / "queue.jsonl"
    previous = b'{"id":"old"}\n'
    queue.write_bytes(previous)
    api.write_json(queue.with_name("state.json"), dict(done=["old"], failed=[]))
    append = api.append_queue
    calls = []
    def interrupted(path, job):
        calls.append(job["id"])
        if len(calls) == 3:
            raise RuntimeError("controlled interruption")
        return append(path, job)
    monkeypatch.setattr(api, "append_queue", interrupted)
    with pytest.raises(RuntimeError, match="controlled"):
        api.enqueue(root, queue)
    monkeypatch.setattr(api, "append_queue", append)
    api.enqueue(root, queue)
    api.enqueue(root, queue)
    assert queue.read_bytes().startswith(previous)
    rows = [json.loads(line) for line in queue.read_text().splitlines()][1:]
    assert len(rows) == 6 and len({r["id"] for r in rows}) == 6
    assert rows[0]["after"] == []
    assert all(rows[i]["after"] == [rows[i-1]["id"]] for i in range(1,6))
    assert all(r["min_free_vram_frac"] == .30 for r in rows)
    assert json.loads((root / "queue_receipt.json").read_text())["complete"]


def metrics():
    path = Path(__file__).with_name("h2oclu_batching_metrics.py")
    assert path.is_file(), "physical evaluator is missing"
    return importlib.import_module("h2oclu_batching_metrics")


def test_physical_gauge_uses_same_shift_for_levels_and_gap():
    api = metrics()
    S = np.diag([2., 1., 1.])
    H = np.diag([-4., 1., 4.])
    result = api.matrix_metrics(H + 3*S, H, S, np.array([0,0,1]), np.array([1,1]))
    assert result["gauge_c_eV"] == pytest.approx(3)
    assert result["H_MAE_gauge_meV"] == pytest.approx(0, abs=1e-10)
    assert result["levels_rmse_aligned_meV"] == pytest.approx(0, abs=1e-10)
    assert result["gap_error_meV"] == pytest.approx(0, abs=1e-10)
    assert result["reference_gap_eV"] == pytest.approx(3.)
    assert result["window_level_count"] == 2


def test_union_support_counts_false_zeros_and_false_nonzeros():
    api = metrics()
    H = np.array([[-2., .2, 0], [.2, 1, 0], [0,0,4.]])
    P = np.array([[-2., 0, .3], [0,1,0], [.3,0,4.]])
    result = api.matrix_metrics(P, H, np.eye(3), np.array([0,0,1]), np.array([1,1]))
    assert result["support_size"] == 7
    assert result["support_F1"] == pytest.approx(.6)
    assert result["false_zero_count"] == 2
    assert result["false_nonzero_count"] == 2
    assert result["H_MAE_gauge_meV"] == pytest.approx(1000/7)


@pytest.mark.parametrize("failure", ["H", "S"])
def test_invalid_raw_matrices_do_not_get_valid_spectra(failure):
    api = metrics()
    H = np.diag([-2.,1.,4.])
    S = np.eye(3)
    P = H.copy()
    if failure == "H":
        P[0,1] = .5
    else:
        S[0,0] = -1
    result = api.matrix_metrics(P,H,S,np.array([0,0,1]),np.array([1,1]))
    assert result["spectral_valid"] is False
    assert result["levels_rmse_aligned_meV"] is None
    assert result["gap_error_meV"] is None


@pytest.fixture
def native_inputs(tiny_campaign):  # noqa: F811
    api, source, root = tiny_campaign
    import torch
    import pytorch_lightning as pl
    from graph2mat import AtomicTableWithEdges
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    prepared = api.prepare(root,source_base=source)
    before = {str(p): api.sha256(p) for p in source.rglob("*") if p.is_file()}
    model = LitMACEMatrixModel(basis_table=AtomicTableWithEdges.from_basis_glob([source/"H.ion.xml"]),
        symmetric_matrix=True, **{k:api.pipeline.resolve(v) for k,v in prepared["recipes"]["0"]["model"].items()})
    for parameter in model.parameters():
        parameter.data.zero_()
    checkpoint = root / "native.ckpt"
    torch.save(dict(state_dict=model.state_dict(),hyper_parameters=dict(model.hparams),
        datamodule_hyper_parameters=dict(out_matrix="hamiltonian",sub_point_matrix=False),
        **{"pytorch-lightning_version":pl.__version__}),checkpoint)
    return api,source,root,prepared,checkpoint,before


def test_native_checkpoint_prediction_reconstructs_h0_without_external_writes(native_inputs):
    api,source,root,prepared,checkpoint,before = native_inputs
    rows = [r for r in prepared["rows"] if r["split"] == "validation"][:1]
    result = metrics().evaluate_checkpoint(checkpoint,rows,root/"prediction")
    assert result["rows"][0]["H_MAE_gauge_meV"] == pytest.approx(0,abs=1e-8)
    assert result["rows"][0]["levels_rmse_aligned_meV"] == pytest.approx(0,abs=1e-8)
    assert before == {str(p): api.sha256(p) for p in source.rglob("*") if p.is_file()}
    assert Path(result["rows"][0]["matrix"]).is_relative_to(root)


@pytest.mark.parametrize("leaf",["physical-H-0.npz","per_structure.json.tmp","evaluation.json.tmp"])
def test_prediction_rejects_output_symlinks_without_touching_target(native_inputs,leaf):
    _,source,root,prepared,checkpoint,_ = native_inputs
    external = source/"do-not-overwrite.txt"
    external.write_bytes(b"original")
    output = root/"prediction"
    output.mkdir()
    (output/leaf).symlink_to(external)
    rows = [r for r in prepared["rows"] if r["split"] == "validation"][:1]
    with pytest.raises(ValueError,match="symlink|output|Graph2Mat|own"):
        metrics().evaluate_checkpoint(checkpoint,rows,output)
    assert external.read_bytes() == b"original"


def completed_result(api,root,job,rows):
    """Publish real RecoveryStore generation around the native-shaped trainer double."""
    from test_training_recovery import SavedTrainer
    directory = Path(job["root"])
    native,_ = api.recovery.read_job(Path(job["config"]),directory)
    store = api.recovery.recovery_api().RecoveryStore(directory,native["fingerprint"])
    trainer = SavedTrainer()
    trainer.payload["g2m_recovery"]["fingerprint_sha256"] = store.fingerprint_sha256
    trainer.payload["g2m_recovery"]["seed"] = job["seed"]
    boundary = dict(next_epoch=2000,global_step=8000,samples_committed=64000,plan_sha256="a"*64,epoch_complete=True)
    trainer.advance(boundary)
    best = directory/"best/selected.ckpt"
    best.parent.mkdir()
    trainer.payload["callbacks"]["best"]["best_model_path"] = str(best)
    trainer.save_checkpoint(best)
    store.publish(trainer,boundary=boundary,best_path=best)
    receipt = store.latest()
    fit = dict(checkpoint=str(best),recovery_checkpoint=receipt["checkpoint"],samples_committed=64000)
    evaluation = dict(checkpoint=str(best),checkpoint_sha256=api.sha256(best), rows=[dict(sample_id=r["sample_id"],
        split=r["split"],spectral_valid=True,H_MAE_gauge_meV=1.,levels_rmse_aligned_meV=2.,gap_error_meV=3.) for r in rows])
    result = dict(arm=job["arm"],seed=job["seed"],training=fit,evaluation=evaluation)
    api.write_json(directory/"result.json",result)
    # Binding evaluation/result to native checkpoint is checked, not trusted from directory names.
    api.write_json(directory/"evaluation_receipt.json",dict(checkpoint=str(best),checkpoint_sha256=api.sha256(best),
        result_sha256=api.sha256(directory/"result.json")))
    return result


@pytest.mark.parametrize("change",[None,"foreign","missing","duplicate","invalid_spectrum","altered_json"])
def test_summary_checks_result_identity_coverage_and_spectral_gate(frozen_pilot,change):
    api,pilot,root = frozen_pilot
    manifest = api.prepare(root,pilot,approved=True)
    rows = [r for r in manifest["rows"] if r["split"] in ("validation","test")]
    for job in manifest["jobs"]:
        result = completed_result(api,root,job,rows)
        if change:
            if change == "foreign": result.update(arm="foreign",seed=99)
            elif change == "missing": result["evaluation"]["rows"].pop()
            elif change == "duplicate": result["evaluation"]["rows"][-1] = result["evaluation"]["rows"][0]
            elif change == "invalid_spectrum":
                for row in result["evaluation"]["rows"]: row["spectral_valid"] = False
            else: result["evaluation"]["rows"][0]["H_MAE_gauge_meV"] = 999.
            directory = Path(job["root"])
            api.write_json(directory/"result.json",result)
            if change != "altered_json":
                receipt = json.loads((directory/"evaluation_receipt.json").read_text())
                receipt["result_sha256"] = api.sha256(directory/"result.json")
                api.write_json(directory/"evaluation_receipt.json",receipt)
    if change in ("foreign","missing","duplicate","altered_json"):
        with pytest.raises(ValueError,match="result|coverage|identity|hash|structure"):
            api.summarize(root)
    else:
        summary = api.summarize(root)
        record = summary["metrics"]["ood_exploratory/levels_rmse_aligned_meV"]
        assert record["paired_count"] == (0 if change else 3)
        if change:
            assert "paired_bootstrap_ratio_ci95" not in record
        else:
            assert record["paired_bootstrap_ratio_ci95"] == pytest.approx([1.,1.])
