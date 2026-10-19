"""Fresh-process bootstrap and append-only recovery of the existing three seeds."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def api():
    assert Path(__file__).with_name("relaunch_h2oclu_nullspace.py").exists(), "relaunch entry point missing"
    return importlib.import_module("relaunch_h2oclu_nullspace")


def test_fresh_cpu_entrypoint_bootstraps_without_cython():
    from run_bilayer_edge_updates import environment
    script = Path(__file__).with_name("relaunch_h2oclu_nullspace.py")
    p = subprocess.run([sys.executable, str(script), "--help"], env=environment(cuda=""),
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "usage:" in p.stdout


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    a = api()
    monkeypatch.setattr(a.recovery, "REPO", tmp_path)
    root = tmp_path/"auditorias_mejoras/existing_campaign"
    root.mkdir(parents=True)
    jobs = []
    for seed in range(3):
        folder = root/f"null_s{seed}"
        folder.mkdir()
        jobs.append(dict(key=f"null_s{seed}", root=str(folder), config=str(folder/"config.yaml")))
    m = dict(jobs=jobs)
    (root/"manifest.json").write_text(json.dumps(m))
    old = [dict(id=f"existing_campaign:null_s{s}", after=[] if s == 0 else [f"existing_campaign:null_s{s-1}"]) for s in range(3)]
    (root/"queue_receipt.json").write_text(json.dumps(dict(jobs=old)))
    checkpoint = root/"null_s0/best.ckpt"
    checkpoint.write_bytes(b"existing completed weights")
    (root/"null_s0/fit.json").write_text(json.dumps(dict(samples_committed=64000, steps=8000, checkpoint=str(checkpoint))))
    queue = root/"queue.jsonl"
    queue.write_text("".join(json.dumps(x)+"\n" for x in old))
    queue.with_name("state.json").write_text(json.dumps(dict(failed=[old[0]["id"]], done=[], pids={})))
    monkeypatch.setattr(a.campaign, "read_campaign", lambda path: m)
    return a, root, queue, m


def test_relaunch_needs_explicit_authorization(campaign):
    a, root, queue, _ = campaign
    previous = queue.read_bytes()
    with pytest.raises(ValueError, match="authoriz"):
        a.enqueue(root, queue=queue)
    assert queue.read_bytes() == previous and not (root/"relaunch_receipt.json").exists()


def test_partial_append_resumes_exactly_three_unique_recovery_entries(campaign, monkeypatch):
    a, root, queue, _ = campaign
    previous, state = queue.read_bytes(), queue.with_name("state.json").read_bytes()
    append = a.append_queue
    def interrupted(path, job):
        if job["id"].endswith("null_s1:bootstrap1"):
            raise OSError("controlled interruption")
        return append(path, job)
    monkeypatch.setattr(a, "append_queue", interrupted)
    with pytest.raises(OSError, match="controlled"):
        a.enqueue(root, queue=queue, approved=True)
    monkeypatch.setattr(a, "append_queue", append)
    receipt = a.enqueue(root, queue=queue, approved=True)
    assert a.enqueue(root, queue=queue, approved=True) == receipt
    rows = [json.loads(x) for x in queue.read_text().splitlines()]
    assert len(rows) == 6 and queue.read_bytes().startswith(previous)
    assert [x["id"] for x in rows[3:]] == [f"existing_campaign:null_s{s}:bootstrap1" for s in range(3)]
    assert rows[3]["after"] == []
    assert rows[4]["after"] == [rows[3]["id"]] and rows[5]["after"] == [rows[4]["id"]]
    assert queue.with_name("state.json").read_bytes() == state


def test_changed_completed_fit_or_prefix_blocks_relaunch(campaign):
    a, root, queue, _ = campaign
    a.enqueue(root, queue=queue, approved=True)
    fit = root/"null_s0/fit.json"
    fit.write_text(fit.read_text()+" ")
    with pytest.raises(ValueError, match="changed"):
        a.read_relaunch(root)


def test_changed_queue_prefix_is_never_rewritten(campaign):
    a, root, queue, _ = campaign
    a.enqueue(root, queue=queue, approved=True)
    changed = b" "+queue.read_bytes()
    queue.write_bytes(changed)
    with pytest.raises(ValueError, match="prefix"):
        a.enqueue(root, queue=queue, approved=True)
    assert queue.read_bytes() == changed


@pytest.mark.parametrize("key", ["null_s0", "null_s1"])
def test_completed_fit_is_not_retrained_and_pending_seed_uses_guard(campaign, monkeypatch, key):
    a, root, queue, _ = campaign
    a.enqueue(root, queue=queue, approved=True)
    calls = []
    def fit(config, folder):
        assert Path(folder).name == "null_s1", "completed seed0 must not train again"
        calls.append("guarded_fit")
    def evaluate(path, selected):
        assert selected == key and Path(path) == root
        calls.append("evaluate_cpu")
    monkeypatch.setattr(a.recovery, "guarded_fit", fit)
    monkeypatch.setattr(a.campaign, "evaluate", evaluate)
    monkeypatch.setattr(a.campaign, "summarize", lambda path: calls.append("summary"))
    original = (root/"null_s0/fit.json").read_bytes()
    a.run(root, key)
    assert calls == (["evaluate_cpu", "summary"] if key == "null_s0" else ["guarded_fit", "evaluate_cpu", "summary"])
    assert (root/"null_s0/fit.json").read_bytes() == original
