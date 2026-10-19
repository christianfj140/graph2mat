"""Pilot boundaries and numerical controls; no external data or GPU."""

import hashlib
import json

import numpy as np
import pytest


def pilot():
    from auditorias_mejoras import pilot_multitarget_w90
    return pilot_multitarget_w90


def test_boundary_and_hash_guards(tmp_path, monkeypatch):
    p = pilot()
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(p, "ROOT", root)
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    (root / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        p.check_fdf(root / "escape")
    fdf = root / "RUN.fdf"
    fdf.write_text("%include external.fdf\n")
    with pytest.raises(ValueError, match="include"):
        p.check_fdf(fdf)
    fdf.write_text("SystemLabel ../outside\n")
    with pytest.raises(ValueError, match="(?i)label"):
        p.check_fdf(fdf)
    with pytest.raises(ValueError, match="hash"):
        p.verified_file(root, fdf, "0" * 64)
    assert outside.read_text() == "untouched"


def test_split_is_not_reassigned():
    p = pilot()
    rows = [dict(sample_id=f"md_{i}", split="train" if i < 1198 else
                 "validation" if i < 1348 else "test") for i in range(1498)]
    source = json.dumps(rows)
    result = p.split_ids(dict(rows=rows, split_counts=dict(train=1198, validation=150, test=150)))
    assert list(map(len, (result["train"], result["validation"], result["test"]))) == [1198, 150, 150]
    assert result["validation"][0] == "md_1198"
    assert json.dumps(rows) == source
    rows[1]["sample_id"] = "md_0"
    with pytest.raises(ValueError, match="duplicate"):
        p.split_ids(dict(rows=rows, split_counts=dict(train=1198, validation=150, test=150)))


def test_preflight_no_clobber(tmp_path):
    p = pilot()
    existing = tmp_path / "owned"
    existing.mkdir()
    (existing / "original").write_text("keep")
    with pytest.raises(FileExistsError):
        p.preflight(existing)
    assert (existing / "original").read_text() == "keep"


def test_periodic_floor_and_spectral_controls():
    import sisl
    from scipy.sparse import csr_array
    from graph2mat import AtomicTableWithEdges, MatrixDataProcessor
    from graph2mat.core.data.multitarget import prepare_multitarget_data

    p = pilot()
    atom = sisl.Atom("C", orbitals=[sisl.AtomicOrbital("2s", R=1.),
                                   *[sisl.AtomicOrbital(f"2p{a}", R=1.) for a in "xyz"]])
    geom = sisl.Geometry([[0., 0., 0.]], atoms=atom, lattice=[10.] * 3)
    geom.set_nsc([3, 1, 1])
    s = np.zeros((4, 12))
    s[:, :4] = np.eye(4)
    h = s * np.arange(1., 5.)[:, None]
    # Opposite images lie outside the graph cutoff; both are deliberately present.
    h[0, 4] = h[0, 8] = .01
    def observed(a):
        return csr_array((a.ravel(), np.tile(np.arange(12, dtype=np.int32), 4),
                          np.arange(0, 49, 12, dtype=np.int32)), shape=(4, 12))
    sm = observed(s)
    bundle = dict(overlap=sisl.Overlap.fromsp(geom, sm),
        hamiltonian=sisl.Hamiltonian.fromsp(geom, observed(h), S=sm),
        density=sisl.DensityMatrix.fromsp(geom, observed(.5 * s), S=sm),
        energy_density=sisl.EnergyDensityMatrix.fromsp(geom, observed(.5 * h), S=sm))
    processor = MatrixDataProcessor(basis_table=AtomicTableWithEdges([atom]),
        symmetric_matrix=True, sub_point_matrix=False, out_matrix="hamiltonian",
        matrix_targets=list(p.TARGETS), matrix_target_spin=1)
    data = prepare_multitarget_data(processor, **bundle)
    k = np.array([[0., 0., 0.], [.5, 0., 0.]])
    floor = p.reference_floor(bundle, processor, data, k)
    assert floor["targets"]["hamiltonian"]["omitted_frobenius"] == pytest.approx(np.sqrt(.0002))
    assert floor["targets"]["hamiltonian"]["max_error"] == pytest.approx(.01)
    assert floor["original"]["min_s"] == pytest.approx(1.)
    hs, ss = p.bloch(bundle["hamiltonian"], k), p.bloch(bundle["overlap"], k)
    levels = p.spectrum(hs, ss)["levels"]
    shifted = p.spectrum(hs + 2.5 * ss, ss)["levels"]
    np.testing.assert_allclose(np.array(shifted) - levels, 2.5, atol=1e-12)
    with pytest.raises(ValueError, match="positive|definite"):
        p.spectrum(hs, -ss)
    broken = ss.copy()
    broken[:, 0, 1] = .2j
    with pytest.raises(ValueError, match="Hermit"):
        p.spectrum(hs, broken)
    path = p.band_path()
    assert path.shape == (151, 3)
    np.testing.assert_allclose(path[50], [1 / 3, 2 / 3, 0], rtol=0, atol=0)
    # Signed indirect gap at nominal four filled bands, independent of fractional DFT occupations.
    assert p.gaps(np.array([[0, 1, 2, 4, 5], [0, 1, 2, 1, 3]]))["indirect"] == -1


def test_order_last_batch_and_scheduler():
    import torch
    p = pilot()
    ids = [f"id{i}" for i in range(1198)]
    first = p.batch_order(ids, 0, 120)
    torch.manual_seed(999)
    torch.randn(123)
    assert first == p.batch_order(ids, 0, 120)
    assert len(first[37]) == 14 and set(sum(first[:38], [])) == set(ids)
    assert sum(map(len, first[20:])) == 3146
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.Adam([parameter], lr=.0018)
    scheduler = p.make_scheduler(optimizer)
    for _ in range(21):
        optimizer.step()
        scheduler.step()
    expected = 1e-5 + (.0018 - 1e-5) * (1 + np.cos(np.pi * 21 / 4000)) / 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)


def small_pair():
    import torch
    from graph2mat.bindings.torch.data.tests.test_multitarget_dataset import bundle, processor
    from graph2mat.bindings.torch import TorchBasisMatrixDataset
    from graph2mat.tools.lightning.data import TypeLayoutCollate
    from torch_geometric.data import Batch
    p = pilot()
    proc = processor()
    models = [p.make_arm(a, proc.basis_table, small=True) for a in ("h", "joint")]
    models[1].model.mace.load_state_dict(models[0].model.mace.state_dict(), strict=True)
    data = TorchBasisMatrixDataset([bundle(2), bundle(3)], proc)
    batch = TypeLayoutCollate(Batch.from_data_list, proc.basis_table)([data[0], data[1]])
    return models, batch


def test_shared_backbone_and_h_loss():
    import torch
    p = pilot()
    models, batch = small_pair()
    hbatch = p.h_only_batch(batch)
    for name in ("positions", "node_attrs", "edge_index"):
        torch.testing.assert_close(hbatch[name], batch[name], rtol=0, atol=0)
    for a, b in zip(models[0].model.mace.state_dict().values(), models[1].model.mace.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert p.check_pair(models, batch)["h_loss_difference"] < 1e-6
    assert models[0].hparams.n_output_components == 1
    assert models[1].hparams.n_matrix_components == 4


def test_cpu_smoke_both_arms():
    import torch
    p = pilot()
    models, batch = small_pair()
    original = batch.point_labels.clone()
    for arm, model in zip(("h", "joint"), models):
        state = [v.clone() for v in model.parameters()]
        result = p.train_updates(model, [batch, batch], arm, "cpu", warmup=0)
        assert len(result["updates"]) == 2
        assert all(np.isfinite(r["loss"]) for r in result["updates"])
        assert any(not torch.equal(a, b) for a, b in zip(state, model.parameters()))
        val = p.validate_h(model, [batch], arm, "cpu")
        assert val["samples"] == 2 and np.isfinite(val["h_gauge_mse"])
    torch.testing.assert_close(batch.point_labels, original, rtol=0, atol=0)


def test_validation_h_aggregation():
    import torch
    from graph2mat.tools.lightning.data import TypeLayoutCollate
    from torch_geometric.data import Batch
    p = pilot()
    models, batch = small_pair()
    model = models[1].eval()
    collate = TypeLayoutCollate(Batch.from_data_list, model.basis_table)
    separate = [collate([batch.get_example(i)]) for i in range(2)]
    a = p.validate_h(model, [batch], "joint", "cpu")
    b = p.validate_h(model, separate, "joint", "cpu")
    assert a["h_gauge_mse"] == pytest.approx(b["h_gauge_mse"], rel=1e-5)
    assert a["elements"] == b["elements"]
    changed = batch.clone()
    changed.point_labels.fill_(999.)
    changed.edge_labels.fill_(-999.)
    model.eval()
    with torch.no_grad():
        out, again = model(batch), model(changed)
    torch.testing.assert_close(out["node_labels"], again["node_labels"], rtol=0, atol=0)


def test_nonfinite_fails():
    import torch
    p = pilot()
    with pytest.raises(ValueError, match="finite"):
        p.require_finite([torch.tensor(float("nan"))], "loss")


def test_effective_config_serializes_basis_without_memory_addresses():
    p = pilot()
    models, _ = small_pair()
    value = p.serializable(dict(models[0].hparams))
    text = json.dumps(value, allow_nan=False)
    assert "0x" not in text
    assert value["basis_table"]["basis_size"] == [1, 1]


@pytest.mark.parametrize("free,temp,pids,now,reason", [
    (249, 30, set(), 0, "VRAM"), (250, 30, set(), 0, None),
    (500, 80, set(), 0, "temperature"), (500, 30, {55}, 0, "foreign"),
    (500, 30, {10}, 0, None), (500, 30, set(), 900, "deadline"),
    (float("nan"), 30, set(), 0, "telemetry"),
])
def test_guard_boundaries(free, temp, pids, now, reason):
    p = pilot()
    got = p.guard_reason({"memory.total": 1000., "memory.free": free, "temperature.gpu": temp}, pids, {10}, now, 900)
    assert got == reason


def test_allocator_cap():
    p = pilot()
    assert p.memory_fraction(10000, 10000) == .55
    assert p.memory_fraction(10000, 6000) == .3
    with pytest.raises(RuntimeError):
        p.memory_fraction(1000, 500)


def test_process_cleanup_scope_and_deadline(monkeypatch):
    p = pilot()
    from types import SimpleNamespace
    seen = []
    monkeypatch.setattr(p.os, "killpg", lambda pgid, sig: seen.append((pgid, sig)))
    monkeypatch.setattr(p, "group_alive", lambda pgid: False)
    proc = SimpleNamespace(pid=7654321, wait=lambda timeout=None: 0)
    p.terminate_group(proc)
    assert seen == [(7654321, p.signal.SIGTERM), (7654321, p.signal.SIGKILL)]
    # Even a malformed state cannot authorize extending the shared deadline.
    assert p.guard_reason({}, set(), set(), 901, 900) == "deadline"


def test_pilot_one_shot(tmp_path):
    p = pilot()
    out = tmp_path / "pilot"
    out.mkdir()
    (out / "started.json").write_text("keep")
    with pytest.raises(FileExistsError):
        p.pilot(out)
    assert (out / "started.json").read_text() == "keep"


def test_summary_incomplete_and_weighted_cost():
    p = pilot()
    assert p.summarize({}, []) ["status"] == "incomplete"
    rows = [dict(update=i + 1, measured=i >= 20, samples=14 if i in (37, 75, 113) else 32,
                 ids=list(range(14 if i in (37, 75, 113) else 32)),
                 wall_s=1. if i % 2 else 2., loss=.5,
                 lr=1e-5 + (.0018 - 1e-5) * (1 + np.cos(np.pi * i / 4000)) / 2)
            for i in range(120)]
    result = dict(updates=rows, validation=dict(samples=150, h_gauge_mse=.1, seconds=3.),
                  measured_peaks=dict(allocated_mib=100, reserved_mib=200), setup_s=5., order_sha256="same")
    both = p.summarize(dict(h=result, joint=result), [], expected_order=[r["ids"] for r in rows], integrity=True)
    assert both["status"] == "complete"
    assert both["arms"]["h"]["samples_per_s"] == pytest.approx(3146 / 150)
    assert both["gpu_peak_used_mib"] is None
    result["updates"] = rows[:-1]
    assert p.summarize(dict(h=result, joint=result), [])["status"] == "incomplete"


def test_uncertified_candidates_rejected_before_reader(tmp_path, monkeypatch):
    p = pilot()
    root = tmp_path / "data"
    basis = root / "material_basis"
    basis.mkdir(parents=True)
    outside = tmp_path / "outside.ion.xml"
    outside.write_text("must not be read")
    (basis / "extra.ion.xml").symlink_to(outside)
    monkeypatch.setattr(p, "ROOT", root)
    with pytest.raises(ValueError, match="outside"):
        p.certified_table(dict(sources={}))
    (basis / "extra.ion.xml").unlink()
    (basis / "new.ion.xml").write_text("uncertified")
    with pytest.raises(ValueError, match="uncertified"):
        p.certified_table(dict(sources={}))
    assert outside.read_text() == "must not be read"


def test_real_reader_pair_gate_rejects_different_labels():
    p = pilot()
    models, batch = small_pair()
    actual_h = p.h_only_batch(batch)
    actual_h.point_labels[0, 0] += .123
    with pytest.raises((ValueError, AssertionError), match="labels|not equal|not close|mismatch"):
        p.check_pair(models, batch, actual_h)


def test_invalid_pair_suppresses_all_estimates():
    p = pilot()
    rows = [dict(update=i + 1, measured=i >= 20, samples=14 if i in (37, 75, 113) else 32,
                 ids=list(range(14 if i in (37, 75, 113) else 32)), wall_s=1., loss=.5,
                 lr=1e-5 + (.0018 - 1e-5) * (1 + np.cos(np.pi * i / 4000)) / 2) for i in range(120)]
    result = dict(updates=rows, validation=dict(samples=150, h_gauge_mse=.1, seconds=3.),
                  measured_peaks={}, setup_s=5., order_sha256="a")
    other = dict(result, order_sha256="b")
    value = p.summarize(dict(h=result, joint=other), [], expected_order=[r["ids"] for r in rows], integrity=True)
    assert value["status"] == "incomplete"
    assert all(v["estimate_4000_s"] is None for v in value["arms"].values())
    other = dict(result, validation={})
    assert p.summarize(dict(h=result, joint=other), [])["status"] == "incomplete"


def test_cleanup_waits_for_exited_leader_descendant(monkeypatch):
    from types import SimpleNamespace
    p = pilot()
    seen = []
    alive = [True]
    def kill(pgid, sig):
        seen.append((pgid, sig))
        if sig == p.signal.SIGKILL:
            alive[0] = False
    monkeypatch.setattr(p.os, "killpg", kill)
    monkeypatch.setattr(p, "group_alive", lambda pgid: alive[0])
    p.terminate_group(SimpleNamespace(pid=7654321, wait=lambda timeout=None: 0))
    assert not alive[0]
    assert seen == [(7654321, p.signal.SIGTERM), (7654321, p.signal.SIGKILL)]


def test_initial_telemetry_timeout_is_terminal(tmp_path, monkeypatch):
    p = pilot()
    monkeypatch.setattr(p, "gpu_state", lambda: (_ for _ in ()).throw(TimeoutError("driver")))
    with pytest.raises(TimeoutError):
        p.initial_gpu_state()


def test_metric_reset_really_resets_epoch_rmse():
    p = pilot()
    models, batch = small_pair()
    model = models[0]
    model.training_step(p.h_only_batch(batch), 0)
    metric = model._epoch_rmse["train_gauge_h_rmse"]
    assert float(metric.count) > 0
    p.reset_metrics(model)
    assert float(metric.count) == 0


def test_support_mask_includes_represented_zeros():
    from graph2mat.bindings.torch.data.tests.test_multitarget_dataset import bundle, processor
    from graph2mat.core.data.multitarget import prepare_multitarget_data
    p = pilot()
    proc = processor()
    data = prepare_multitarget_data(proc, **bundle())
    data.point_labels[:] = 0.
    data.edge_labels[:] = 0.
    mask = p.graph_support(proc, data, "hamiltonian")
    assert np.count_nonzero(mask.tocsr(0).toarray()) == 4


@pytest.mark.parametrize("stage", ["initial", "query"])
def test_telemetry_failure_never_launches_worker(tmp_path, monkeypatch, stage):
    p = pilot()
    state = {"memory.total": 10000., "memory.free": 9000., "temperature.gpu": 30.}
    for name, value in (("manifest.json", {}), ("preflight.json", dict(status="complete", structures=1498)),
                        ("prepared.json", dict(hashes={})), ("order.json", [])):
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(p, "verify_sources", lambda manifest: None)
    def fail(*args, **kwargs):
        raise TimeoutError("driver timeout")
    monkeypatch.setattr(p, "initial_gpu_state", fail if stage == "initial" else lambda: state)
    monkeypatch.setattr(p, "query_gpu", fail)
    def forbidden(*args, **kwargs):
        raise AssertionError("worker must not launch")
    monkeypatch.setattr(p.subprocess, "Popen", forbidden)
    result = p.pilot(tmp_path)
    assert result["status"] == "blocked_or_aborted"
    assert "driver timeout" in result["reason"]
    assert not list(tmp_path.glob("worker-*.log"))


def test_initial_telemetry_has_real_timeout(monkeypatch):
    p = pilot()
    monkeypatch.setattr(p, "gpu_state", lambda: p.time.sleep(.1))
    with pytest.raises(TimeoutError, match="timeout"):
        p.initial_gpu_state(timeout=.01)


def test_deadline_between_arms_does_not_launch_second_worker(tmp_path, monkeypatch):
    from types import SimpleNamespace
    p = pilot()
    state = {"memory.total": 10000., "memory.free": 9000., "temperature.gpu": 30.}
    for name, value in (("manifest.json", {}), ("preflight.json", dict(status="complete", structures=1498)),
                        ("prepared.json", dict(hashes={})), ("order.json", []), ("result-h.json", dict(updates=[]))):
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(p, "verify_sources", lambda manifest: None)
    monkeypatch.setattr(p, "initial_gpu_state", lambda: state)
    monkeypatch.setattr(p, "query_gpu", lambda **kwargs: (state, set(), "GPU-fixture"))
    clock, launches = [0.], []
    monkeypatch.setattr(p.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(p.os, "killpg", lambda *args: None)
    monkeypatch.setattr(p, "group_alive", lambda pid: False)
    def launch(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        launches.append(1)
        clock[0] = 901.
        return SimpleNamespace(pid=7654321, poll=lambda: 0, returncode=0, wait=lambda timeout=None: 0)
    monkeypatch.setattr(p.subprocess, "Popen", launch)
    result = p.pilot(tmp_path)
    assert len(launches) == 1
    assert result["status"] == "blocked_or_aborted" and "deadline" in result["reason"]


def test_pilot_own_pid_survives_rss_exit_race(tmp_path, monkeypatch):
    """The unreaped Popen leader remains ours even after /proc stops reporting RSS."""
    from types import SimpleNamespace
    p = pilot()
    state = {"memory.total": 10000., "memory.free": 9000., "temperature.gpu": 30.}
    (tmp_path / "order.json").write_text("[]")
    digest = hashlib.sha256(b"[]").hexdigest()
    for name, value in (("manifest.json", {}), ("preflight.json", dict(status="complete", structures=1498)),
                        ("prepared.json", dict(hashes={"order.json": digest})),
                        ("result-h.json", dict(updates=[], order_sha256=digest)),
                        ("result-joint.json", dict(updates=[], order_sha256=digest))):
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(p, "verify_sources", lambda manifest: None)
    monkeypatch.setattr(p, "initial_gpu_state", lambda: state)
    driver = dict(pid=None, active=False, stale=False)
    launched = []
    def query(**kwargs):
        pid = driver["pid"]
        if driver["active"] or driver["stale"]:
            driver["stale"] = False
            return dict(state), {pid}, "GPU-fixture"
        return dict(state), set(), "GPU-fixture"
    def launch(*args, **kwargs):
        pid = 7654321 + len(launched)
        launched.append(pid)
        driver.update(pid=pid, active=True)
        calls = [0]
        def poll():
            calls[0] += 1
            if calls[0] == 1:
                return None
            driver.update(active=False, stale=True)
            return 0
        return SimpleNamespace(pid=pid, poll=poll, returncode=0, wait=lambda timeout=None: 0)
    monkeypatch.setattr(p, "query_gpu", query)
    monkeypatch.setattr(p.subprocess, "Popen", launch)
    monkeypatch.setattr(p, "process_tree_memory", lambda pid: dict(pids=[]))
    monkeypatch.setattr(p, "group_alive", lambda pid: False)
    monkeypatch.setattr(p.os, "killpg", lambda *args: None)
    monkeypatch.setattr(p.time, "sleep", lambda seconds: None)
    result = p.pilot(tmp_path)
    assert result["reason"] is None
    assert len(launched) == 2
    # Dummy workers carry no training rows: lifecycle success must not certify a fit.
    assert result["status"] == "incomplete"


@pytest.mark.parametrize("reused", [False, True])
def test_context_drain_rejects_foreign_or_reused_pid(monkeypatch, reused):
    from pathlib import Path
    p = pilot()
    state = {"memory.total": 10000., "memory.free": 9000., "temperature.gpu": 30.}
    monkeypatch.setattr(p, "query_gpu", lambda **kwargs: (state, {7654321 if reused else 99}, "GPU-fixture"))
    if reused:
        monkeypatch.setattr(Path, "exists", lambda path: True)
    with pytest.raises(RuntimeError, match="foreign"):
        p.wait_context_exit(7654321, "GPU-fixture", p.time.monotonic() + 900)


@pytest.mark.parametrize("limit,reason", [(900., "context"), (.05, "deadline")])
def test_context_drain_is_bounded(monkeypatch, limit, reason):
    p = pilot()
    state = {"memory.total": 10000., "memory.free": 9000., "temperature.gpu": 30.}
    clock = [0.]
    monkeypatch.setattr(p.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(p.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(p, "query_gpu", lambda **kwargs: (state, {7654321}, "GPU-fixture"))
    with pytest.raises(RuntimeError, match=reason):
        p.wait_context_exit(7654321, "GPU-fixture", limit)
    assert clock[0] <= min(limit, 2.) + 1e-6
