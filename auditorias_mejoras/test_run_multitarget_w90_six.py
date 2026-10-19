"""Training orchestration contracts, CPU fixtures only."""

import json
import math

import pytest


def runner():
    from auditorias_mejoras import run_multitarget_w90_six
    return run_multitarget_w90_six


def test_preparation_never_adopts_existing_directory(tmp_path):
    existing = tmp_path / "campaign"
    existing.mkdir()
    (existing / "original").write_text("preserve")
    with pytest.raises(FileExistsError):
        runner().prepare(existing)
    assert (existing / "original").read_text() == "preserve"


def test_frozen_campaign_artifact_mutation_blocks_worker(tmp_path):
    r = runner()
    artifact = tmp_path / "initial.pt"
    artifact.write_bytes(b"initial")
    frozen = {"initial.pt": r.pilot.sha(artifact)}
    r.verify_artifacts(tmp_path, frozen)
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact"):
        r.verify_artifacts(tmp_path, frozen)


@pytest.mark.parametrize("arm", ["h", "joint"])
def test_real_fit_continuous_lr_order_h_selector_and_reload(tmp_path, arm):
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    from auditorias_mejoras.test_pilot_multitarget_w90 import small_pair
    from graph2mat.bindings.torch.load import load_from_lit_ckpt
    from graph2mat.tools.lightning.data import TypeLayoutCollate

    r = runner()
    torch.set_num_threads(2)
    models, batch = small_pair()
    model = models[0 if arm == "h" else 1]
    if arm == "h":
        batch = r.pilot.h_only_batch(batch)
    samples = [batch.get_example(i) for i in range(2)]
    for i, sample in enumerate(samples):
        sample.sample_id = f"fixture{i}"
    collate = TypeLayoutCollate(Batch.from_data_list, model.basis_table)
    train = DataLoader(samples, batch_sampler=[[0, 1], [1], [1, 0], [0]], collate_fn=collate)
    val = DataLoader(samples, batch_size=2, collate_fn=collate)
    order = [["fixture0", "fixture1"], ["fixture1"], ["fixture1", "fixture0"], ["fixture0"]]
    result = r.fit_model(model, train, val, tmp_path, arm, order, "cpu", updates=4, validate_every=2)
    rows = [json.loads(line) for line in (tmp_path / "updates.jsonl").read_text().splitlines()]
    assert [row["ids"] for row in rows] == order
    assert [row["update"] for row in rows] == [1, 2, 3, 4]
    assert [row["lr"] for row in rows] == pytest.approx([
        1e-5 + (.0018 - 1e-5) * (1 + math.cos(math.pi * i / 4000)) / 2 for i in range(4)])
    assert result["updates"] == 4 and result["samples"] == 6
    assert [row["update"] for row in result["validation"]] == [2, 4]
    assert result["monitor"] == ("val_loss" if arm == "h" else "val/target/hamiltonian/loss")
    assert result["best_h_gauge_mse"] == min(row["h_gauge_mse"] for row in result["validation"])
    loaded, _ = load_from_lit_ckpt(tmp_path / "last.ckpt", as_torch=True)
    model.eval(), loaded.eval()
    example = next(iter(val))
    with torch.no_grad():
        expected, actual = model(example), loaded(example)
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    loaded_best, _ = load_from_lit_ckpt(result["best_checkpoint"], as_torch=False)
    loaded_best.log = lambda *args, **kwargs: None
    common = r.pilot.validate_h(loaded_best, val, arm, "cpu")
    assert common["h_gauge_mse"] == pytest.approx(result["best_h_gauge_mse"], rel=2e-5)


def test_second_supervisor_start_does_not_replace_state(tmp_path):
    r = runner()
    r.pilot.write_json(tmp_path / "started.json", {"original": True})
    with pytest.raises(FileExistsError):
        r.supervise(tmp_path)
    assert json.loads((tmp_path / "started.json").read_text()) == {"original": True}


def test_failed_sequence_stops_pending_without_retry(tmp_path, monkeypatch):
    r = runner()
    jobs = [{"id": "h_s0", "arm": "h", "seed": 0}, {"id": "joint_s0", "arm": "joint", "seed": 0}]
    r.pilot.write_json(tmp_path / "campaign.json", {"jobs": jobs})
    monkeypatch.setattr(r, "verify_campaign", lambda out: {})
    calls = []

    def fail(out, job, stream, deadline):
        calls.append(job["id"])
        raise RuntimeError("controlled worker failure")

    monkeypatch.setattr(r, "supervise_job", fail)
    state = r.supervise(tmp_path)
    assert calls == ["h_s0"]
    assert state["status"] == "aborted"
    assert state["jobs"] == {"h_s0": "failed", "joint_s0": "pending"}
    assert "controlled worker failure" in state["reason"]


@pytest.mark.parametrize("hazard", ["foreign", "VRAM", "temperature", "deadline", "telemetry"])
def test_guard_terminates_only_owned_worker(tmp_path, monkeypatch, hazard):
    import io
    import subprocess
    import sys
    import time

    r = runner()
    job = {"id": "h_s0", "arm": "h", "seed": 0}
    (tmp_path / job["id"]).mkdir()
    real_popen = subprocess.Popen
    children = []
    calls = 0

    def launch(*args, **kwargs):
        child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    def telemetry(**kwargs):
        nonlocal calls
        calls += 1
        if calls > 1 and hazard == "telemetry":
            raise TimeoutError("controlled telemetry timeout")
        state = {"memory.total": 10000., "memory.free": 9000., "temperature.gpu": 40.}
        pids = set()
        if calls > 1:
            if hazard == "foreign":
                pids = {99999999}
            elif hazard == "VRAM":
                state["memory.free"] = 2400.
            elif hazard == "temperature":
                state["temperature.gpu"] = 80.
            elif hazard == "deadline":
                time.sleep(.03)
        return state, pids, "cpu-fixture"

    monkeypatch.setattr(r.subprocess, "Popen", launch)
    monkeypatch.setattr(r.pilot, "query_gpu", telemetry)
    expected = "timeout" if hazard == "telemetry" else hazard
    with pytest.raises((RuntimeError, TimeoutError), match=expected):
        r.supervise_job(tmp_path, job, io.StringIO(), time.monotonic() + (.02 if hazard == "deadline" else 5))
    assert len(children) == 1 and children[0].poll() is not None


def test_process_record_io_failure_terminates_owned_worker_and_stops_queue(tmp_path, monkeypatch):
    import subprocess
    import sys

    r = runner()
    jobs = [{"id": "h_s0", "arm": "h", "seed": 0}, {"id": "joint_s0", "arm": "joint", "seed": 0}]
    for job in jobs:
        (tmp_path / job["id"]).mkdir()
    r.pilot.write_json(tmp_path / "campaign.json", {"jobs": jobs})
    real_write, real_popen = r.atomic_state, subprocess.Popen
    children = []

    def failing_write(path, value):
        if path.name == "process.json":
            raise OSError("controlled process record failure")
        return real_write(path, value)

    def launch(*args, **kwargs):
        child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(r, "verify_campaign", lambda out: {})
    monkeypatch.setattr(r, "atomic_state", failing_write)
    monkeypatch.setattr(r.subprocess, "Popen", launch)
    monkeypatch.setattr(r.pilot, "query_gpu", lambda **kwargs: (
        {"memory.total": 10000., "memory.free": 9000., "temperature.gpu": 40.}, set(), "cpu-fixture"))
    try:
        state = r.supervise(tmp_path)
        assert len(children) == 1 and children[0].poll() is not None
        assert state["jobs"] == {"h_s0": "failed", "joint_s0": "pending"}
        assert "process record failure" in state["reason"]
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
