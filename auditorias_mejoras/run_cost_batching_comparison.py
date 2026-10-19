"""Resumable, opt-in cost-batching experiments. No automatic scientific submission."""
from contextlib import contextmanager
from collections import Counter
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

import yaml

import bench_pipeline_gpu as pipeline
from run_bilayer_edge_updates import ForeignGPU, check_progress, environment
from run_readout_comparison import PREVIOUS, REPO, gpu_processes, process_tree, sha256, verify_files, write_json


PACKAGES = ("torch", "pytorch-lightning", "torch-geometric", "numpy", "sisl", "e3nn", "mace-torch")


def existing_inputs(source_base):
    """Read existing labels only; never import external generation/evaluation scripts."""
    import numpy as np
    import sisl
    source = Path(source_base).resolve(strict=True)
    path = source / "g2m_improvements/bench/h2oclu/frozen_split_manifest.json"
    original = json.loads(path.read_text())
    rows = copy.deepcopy(original["rows"])
    if original["material"] != "h2oclu" or Counter(r["split"] for r in rows) != {
            "train": 32, "validation": 4, "test": 3}:
        raise ValueError("h2oclu frozen split mismatch")
    if len({r["sample_id"] for r in rows}) != 39:
        raise ValueError("duplicate frozen sample IDs")
    files, versions, species_assets = {path}, set(), {}
    for row in rows:
        fdf, reference = Path(row["structure_path"]), Path(row["reference_tshs_path"])
        folder = reference.parent.resolve(strict=True)
        key = folder.name + "_" + hashlib.sha1(str(folder).encode()).hexdigest()[:8]
        h0 = source / "g2m_improvements/delta_h0/h0" / key / reference.name
        required = [fdf, reference, h0, folder / "RUN.out", folder / "0_NORMAL_EXIT"]
        delta = None
        if row["split"] != "test":
            delta = source / "g2m_improvements/delta_h0/delta" / key / reference.name
            required += [delta, delta.parent / "RUN.fdf"]
        for item in required:
            if not item.is_file():
                raise FileNotFoundError("existing reference/H0/delta/output required: " + str(item))
        output = (folder / "RUN.out").read_text(errors="replace")
        version = re.search(r"^Version\s*:\s*(\S+)", output, re.MULTILINE)
        if version is None or "SCF Convergence" not in output:
            raise ValueError("missing SIESTA version/convergence: " + str(folder))
        versions.add(version[1])
        ref = sisl.get_sile(reference).read_hamiltonian()
        atomic = sisl.get_sile(h0).read_hamiltonian()
        matrices = [ref, atomic] + ([] if delta is None else [sisl.get_sile(delta).read_hamiltonian()])
        geometry = sisl.get_sile(fdf).read_geometry()
        if (not ref.spin.is_unpolarized or ref.orthogonal or list(ref.nsc) != [1, 1, 1]
                or ref.geometry.na != row["n_atoms"] or row["net_charge"] != 0 or row["kpts"] != [1, 1, 1]
                or not np.array_equal(ref.geometry.orbitals, geometry.orbitals)
                or not np.allclose(ref.geometry.xyz, geometry.xyz, rtol=0, atol=1e-6)):
            raise ValueError("incompatible molecular reference: " + row["sample_id"])
        for matrix in matrices:
            if (matrix.shape != ref.shape or not matrix.spin.is_unpolarized or matrix.orthogonal
                    or not np.allclose(matrix.geometry.xyz, ref.geometry.xyz, rtol=0, atol=1e-6)
                    or not np.array_equal(matrix.geometry.atoms.Z, ref.geometry.atoms.Z)
                    or not np.array_equal(matrix.geometry.orbitals, ref.geometry.orbitals)
                    or not np.isfinite(matrix._csr._D).all()):
                raise ValueError("incompatible/nonfinite H0/delta: " + row["sample_id"])
        H, S = ref.tocsr(0).toarray(), ref.tocsr(ref.S_idx).toarray()
        defect = lambda x: float(np.linalg.norm(x - x.T) / max(np.linalg.norm(x), 1e-30))
        if defect(H) > 1e-8 or defect(S) > 1e-8:
            raise ValueError("non-Hermitian reference: " + row["sample_id"])
        spectrum = np.linalg.eigvalsh((S + S.T) * .5)
        if spectrum[0] <= 1e-10:
            raise ValueError("reference overlap not positive definite: " + row["sample_id"])
        error = 0.
        if delta is not None:
            error = float(np.max(np.abs(matrices[2].tocsr(0).toarray() - H + atomic.tocsr(0).toarray())))
            if error > 1e-10 or not np.array_equal(matrices[2].tocsr(matrices[2].S_idx).toarray(), S):
                raise ValueError("delta is not H-H0 with reference S: " + row["sample_id"])
        # TSHS stores orbital counts, not atomic numbers; sisl may use He for O.
        for atom in geometry.atoms:
            for suffix in (".ion.xml", ".psf"):
                asset = folder / (atom.symbol + suffix)
                digest = sha256(asset)
                identity = atom.symbol + suffix
                if identity in species_assets and species_assets[identity] != digest:
                    raise ValueError("inconsistent basis/pseudo: " + identity)
                species_assets[identity] = digest
                files.add(asset)
        files.update(required)
        row.update(h0=str(h0), delta=None if delta is None else str(delta), n_orbitals=ref.geometry.no,
            overlap_min_eigenvalue=float(spectrum[0]), overlap_condition=float(spectrum[-1] / spectrum[0]),
            H_hermiticity=defect(H), S_hermiticity=defect(S), delta_error_eV=error)
    if len(versions) != 1:
        raise ValueError("mixed SIESTA reference versions")
    return rows, files, versions.pop(), species_assets


def input_resolution(cfg):
    """Freeze ordered loader paths as well as targets behind historical split links."""
    data, result = cfg.get("data", {}), {}
    base = Path(data.get("root_dir", ".")).absolute()
    for argument in ("basis_files", "train_runs", "val_runs"):
        specification = data.get(argument)
        if specification is None:
            continue
        paths = base.glob(specification) if isinstance(specification, str) else (base / p for p in specification)
        result[argument] = [dict(path=str(path.absolute()), target=str(path.resolve(strict=True))) for path in paths]
    return result


def make_job(root, cfg, hashes, initial, initial_hash, identifiers, versions):
    root = own_output(root)
    root.mkdir()
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    files = {**hashes, str(config): sha256(config), str(initial): sha256(initial)}
    resolution = input_resolution(cfg)
    fingerprint = dict(files=files, versions=versions, seed=cfg["seed_everything"], input_resolution=resolution,
        policy="seed+epoch/frozen-plan/independent-loader-rng/nonpersistent-workers/v2", effective_scheduler=cfg["lr_scheduler"])
    write_json(root / "job.json", dict(fingerprint=fingerprint, config=str(config), files=files,
        max_epochs=cfg["trainer"]["max_epochs"], sample_ids=identifiers,
        initial_state=str(initial), initial_sha256=initial_hash, versions=versions,
        input_resolution=resolution,
        technical_pilot=True, scientific_authorized=False))
    return dict(root=str(root), config=str(config), budget=cfg["data"]["max_batch_cost"])


def prepare(root, source_base=PREVIOUS):
    root = own_output(root)
    if root.exists():
        raise ValueError("new exclusive campaign directory required")
    api = recovery_api()
    import torch
    import pytorch_lightning as pl
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    from graph2mat.tools.lightning._helpers import glob as data_glob
    from readout_comparison_callback import tensor_hash
    rows, files, siesta, assets = existing_inputs(source_base)
    recipes, modules, identifiers = {}, [], None
    try:
        for seed in (0, 1, 2):
            source = Path(source_base) / "runs/bench/h2oclu" / f"h2oclu__ts{seed}__g2mv2_bench_delta"
            original = source / "config.yaml"
            cfg = yaml.safe_load(original.read_text())
            if cfg["seed_everything"] != seed:
                raise ValueError("historical recipe seed mismatch")
            cfg["data"]["root_dir"] = str(source.resolve())
            bases = list(data_glob(source, cfg["data"]["basis_files"]))
            if not bases:
                raise FileNotFoundError("historical basis missing")
            for basis in bases:
                if assets.get(basis.name) != sha256(basis):
                    raise ValueError("training basis differs from reference: " + str(basis))
            files.update(Path(b).resolve() for b in bases)
            files.add(original)
            id_by_label = {Path(r["delta"]).resolve(): r["sample_id"] for r in rows if r["delta"]}
            actual_ids = []
            for split, arg in (("train", "train_runs"), ("validation", "val_runs")):
                spec = cfg["data"][arg]
                paths = data_glob(source, spec) if isinstance(spec, str) else [source / p for p in spec]
                resolved = [Path(p).resolve() for p in paths]
                labels = {Path(r["delta"]).parent if p.suffix == ".fdf" else Path(r["delta"])
                          for r in rows if r["split"] == split for p in [resolved[0]]}
                actual = {p.parent if p.suffix == ".fdf" else p for p in resolved}
                if actual != labels or len(resolved) != len(labels):
                    raise ValueError("historical recipe split/labels differ from manifest")
                if split == "train":
                    actual_ids = [id_by_label[p.parent / Path(rows[0]["reference_tshs_path"]).name
                                             if p.suffix == ".fdf" else p] for p in resolved]
                files.update(resolved)
            if identifiers is not None and identifiers != actual_ids:
                raise ValueError("historical seed data order differs")
            identifiers = actual_ids
            cfg["trainer"].update(devices=1, logger=False, reload_dataloaders_every_n_epochs=1,
                                   enable_progress_bar=False)
            recipes[str(seed)] = cfg
        # Freeze all sources and native extensions; do not use historical checkpoints as baselines.
        files.update((REPO / "src/graph2mat").rglob("*.py"))
        files.update(Path(__file__).parent / name for name in (
            "training_recovery.py", "run_cost_batching_comparison.py", "bench_pipeline_gpu.py",
            "run_readout_comparison.py", "run_bilayer_edge_updates.py", "readout_comparison_callback.py"))
        files.update(Path(sys.modules[name].__file__) for name in (
            "graph2mat.core.data._sparse", "graph2mat.core.modules._labels_resort"))
        hashes = {str(path.resolve()): sha256(path) for path in sorted(files)}
        versions = {name: importlib.metadata.version(name) for name in PACKAGES}
        # Validate and construct everything before creating any output.
        states, initial_hashes, costs = {}, {}, None
        for seed in (0, 1, 2):
            cfg = recipes[str(seed)]
            pl.seed_everything(seed, workers=True)
            module = api.RecoveryDataModule(**{**cfg["data"], "max_batch_cost": 1})
            modules.append(module)
            module.setup("fit")
            model = LitMACEMatrixModel(basis_table=module.basis_table,
                symmetric_matrix=cfg["data"]["symmetric_matrix"], n_matrix_components=cfg["data"]["n_matrix_components"],
                **{k: pipeline.resolve(v) for k, v in cfg["model"].items()})
            states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            initial_hashes[seed] = tensor_hash(states[seed].items())
            current = list(module.train_dataloader().batch_sampler.costs)
            if costs is not None and costs != current:
                raise ValueError("train costs differ between seeds")
            costs = current
        ceiling = sum(sorted(costs, reverse=True)[:8])
        budgets = sorted({max(1, int(ceiling * fraction)) for fraction in (.5, .75)})
        verify_files(hashes)
        root.mkdir(parents=True)
        initials, pilots = {}, []
        for seed, state in states.items():
            path = root / f"initial-s{seed}.pt"
            torch.save(state, path)
            initials[str(seed)] = dict(path=str(path), fixed_sha256=initial_hashes[seed],
                                      cost_sha256=initial_hashes[seed])
        arms = [None, *budgets]
        for round_index in (0, 1):
            for budget in arms if round_index == 0 else reversed(arms):
                cfg = copy.deepcopy(recipes["0"])
                cfg["data"]["max_batch_cost"] = budget
                cfg["trainer"].update(max_epochs=5, default_root_dir=str(root))
                name = f"pilot-r{round_index}-" + ("fixed" if budget is None else f"cost{budget}")
                row = make_job(root / name, cfg, hashes, Path(initials["0"]["path"]),
                               initial_hashes[0], identifiers, versions)
                pilots.append(dict(row, round=round_index, name=name))
        result = dict(rows=rows, split_counts=dict(Counter(r["split"] for r in rows)),
            files=hashes, versions=versions, siesta_version=siesta, recipes=recipes, sample_ids=identifiers,
            initial_states=initials, training_costs=costs, fixed_proxy_ceiling=ceiling,
            pilot_budgets=budgets, pilots=pilots, scientific_authorized=False,
            effective_scheduler="explicit cosine2000; historical checkpoints had NO scheduler (constant LR)",
            primary_metrics="physical H=H0+deltaH gauge aligned; molecular levels/gap conditional on reference S",
            limits="frozen manual split; incomplete trajectory lineage; previously observed OOD; exploratory")
        write_json(root / "manifest.json", result)
        verify_files(hashes)
        return result
    finally:
        for module in modules:
            module.teardown(None)


def select_pilot_budget(rows):
    fixed = {r["round"]: r["peak_allocated_mib"] for r in rows if r["budget"] is None}
    budgets = sorted({r["budget"] for r in rows if r["budget"] is not None})
    if set(fixed) != {0, 1} or not budgets:
        raise ValueError("two complete paired pilot rounds required")
    passing = []
    for budget in budgets:
        candidate = {r["round"]: r["peak_allocated_mib"] for r in rows if r["budget"] == budget}
        if set(candidate) != {0, 1} or not all(candidate[r] > 0 for r in candidate):
            raise ValueError("incomplete/nonpositive pilot measurement")
        if all(candidate[r] < fixed[r] for r in (0, 1)):
            passing.append(budget)
    return dict(budget=max(passing) if passing else min(budgets), reduced_both_rounds=bool(passing))


def pilot(root):
    root = own_output(root)
    manifest = json.loads((root / "manifest.json").read_text())
    verify_files(manifest["files"])
    if gpu_processes(process_tree(os.getpid())):
        raise ForeignGPU("pilot postponed: GPU is occupied; no indefinite waiting")
    rows = []
    for item in manifest["pilots"]:
        job, _ = read_job(Path(item["config"]), Path(item["root"]))
        if not job.get("technical_pilot") or job["max_epochs"] != 5:
            raise ValueError("pilot is limited to five complete epochs")
        result = guarded_fit(Path(item["config"]), Path(item["root"]), pilot=True)
        rows.append({**item, **result})
        write_json(root / "pilot-partial.json", rows)
    selected = select_pilot_budget(rows)
    verify_files(manifest["files"])
    result = dict(rows=rows, selected=selected, scientific_authorized=False)
    write_json(root / "pilot.json", result)
    return result


def own_output(path):
    path = Path(path).absolute()
    allowed = REPO / "auditorias_mejoras"
    if allowed not in path.parents or path.resolve() != path:
        raise ValueError("output must be an owned Graph2Mat audit subdirectory")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("output symlink is not owned")
    return path


def recovery_api():
    pipeline.compiled_extensions()
    import training_recovery
    return training_recovery


def read_job(config, root):
    root = own_output(root)
    config = Path(config).absolute()
    if root not in config.parents or config.resolve() != config:
        raise ValueError("config must be owned by this job")
    job = json.loads((root / "job.json").read_text())
    for name, version in job.get("versions", {}).items():
        if importlib.metadata.version(name) != version:
            raise ValueError("frozen software version changed: " + name)
    if job["config"] != str(config) or str(config) not in job["files"]:
        raise ValueError("frozen config provenance mismatch")
    verify_files(job["files"])
    cfg = yaml.safe_load(config.read_text())
    if "input_resolution" in job and input_resolution(cfg) != job["input_resolution"]:
        raise ValueError("frozen input resolution/order changed (split links or glob results)")
    if (type(job["max_epochs"]) is not int or job["max_epochs"] < 1
            or cfg["trainer"]["max_epochs"] != job["max_epochs"]):
        raise ValueError("invalid/frozen training epoch budget")
    identifiers = job["sample_ids"]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("frozen sample IDs must be unique and nonempty")
    return job, cfg


@contextmanager
def supervisor_lock(root):
    path = root / "supervisor.lock"
    if path.is_symlink():
        raise ValueError("supervisor lock symlink")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another supervisor owns this job") from error
        yield
    finally:
        os.close(descriptor)


def attempt_accounting(start, end, receipt):
    initial = 0 if start is None else start["samples_committed"]
    confirmed = initial if receipt is None else receipt["samples_committed"]
    observed = confirmed if end is None else end["samples_observed"]
    return {"confirmed_samples_start": initial, "confirmed_samples_end": confirmed,
            "observed_samples_end": None if end is None else observed,
            "repeated_samples_lower_bound": max(0, observed - confirmed),
            "repeated_samples_upper_bound": None,
            "heartbeat_is_exact_at_exit": False}


def finalize(store, job, attempts, last_info=None):
    import torch
    receipt = store.latest()
    if receipt is None or receipt["next_epoch"] != job["max_epochs"]:
        raise RuntimeError("full training budget has not been committed")
    if receipt["samples_committed"] != len(job["sample_ids"]) * job["max_epochs"]:
        raise RuntimeError("final training coverage mismatch")
    best = store.restore_best(receipt)
    if best is None:
        raise RuntimeError("completed training has no scientific best-val_loss checkpoint")
    state = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=False)
    selected = [value for value in state["callbacks"].values()
                if isinstance(value, dict) and value.get("best_model_path") == str(best)]
    if len(selected) != 1 or selected[0].get("best_model_score") is None:
        raise RuntimeError("scientific best score absent/ambiguous")
    score = float(selected[0]["best_model_score"])
    if not torch.isfinite(torch.tensor(score)):
        raise RuntimeError("scientific best score is nonfinite")
    result = {**(last_info or {}), "checkpoint": str(best), "val_loss": score,
        "recovery_checkpoint": receipt["checkpoint"], "steps": receipt["global_step"],
        "samples_committed": receipt["samples_committed"], "attempts": attempts,
        "seconds_all_attempts": sum(row["seconds"] for row in attempts),
        "seconds_successful_fit": sum(row["seconds"] for row in attempts if row.get("exit_reason") == "success"),
        "seconds_waiting_gpu": sum(row.get("seconds_waiting_gpu", 0) for row in attempts),
        "repeated_samples_lower_bound": sum(row.get("repeated_samples_lower_bound") or 0 for row in attempts),
        "attempts_with_unknown_repeated_work": sum(row.get("repeated_samples_lower_bound") is None for row in attempts),
        "finalized_from_generation": True}
    return result


def guarded_fit(config, job_root, pilot=False):
    root = own_output(job_root)
    job, _ = read_job(config, root)
    store = recovery_api().RecoveryStore(root, job["fingerprint"])
    with supervisor_lock(root):
        attempts_file = root / "attempts.json"
        attempts = json.loads(attempts_file.read_text()) if attempts_file.exists() else []
        for previous in attempts:
            previous.setdefault("exit_reason", "unknown")
        while True:
            job, _ = read_job(config, root)
            receipt = store.latest()
            if receipt is not None and receipt["next_epoch"] >= job["max_epochs"]:
                previous = root / "fit.json"
                info = json.loads(previous.read_text()) if previous.is_file() else None
                if info is not None and info.get("recovery_checkpoint") != receipt["checkpoint"]:
                    info = None  # Never attribute a rolled-back generation's resources to another fit.
                result = finalize(store, job, attempts, info)
                write_json(root / "fit.json", result)
                return result
            if receipt is not None:
                store.restore_best(receipt)
            waiting = time.monotonic()
            while gpu_processes(process_tree(os.getpid())):
                print("WAIT foreign GPU; recovery retained", flush=True)
                time.sleep(30)
            wait_seconds = time.monotonic() - waiting
            original = pipeline.gpu_state
            state = original()
            fraction = pipeline.memory_fraction(state["memory.total"], state["memory.free"])
            attempt = own_output(root / ("attempt-" + str(time.time_ns())))
            attempt.mkdir()
            started_wall, start = time.time(), time.monotonic()
            row = {"attempt": str(attempt), "resume_checkpoint": None if receipt is None else receipt["checkpoint"],
                   "recovery_fallback": [] if receipt is None else receipt["fallback"],
                   "seconds_waiting_gpu": wait_seconds, "started": started_wall, "exit_reason": "unknown"}

            def exclusive_state():
                if gpu_processes(process_tree(os.getpid())):
                    raise ForeignGPU("yielding to foreign compute; no foreign signals")
                check_progress(attempt / "progress.json", started_wall, time.time(), pilot)
                return original()

            pipeline.gpu_state = exclusive_state
            info, error = None, None
            try:
                command = [sys.executable, str(Path(__file__).resolve()), "fit", "--root", str(root),
                           "--config", str(config), "--fraction", str(fraction), "--attempt", str(attempt)]
                if receipt is not None:
                    command += ["--checkpoint", receipt["checkpoint"]]
                status = pipeline.guarded_command(command, attempt / "train.log",
                                                  attempt / "telemetry.json", environment())
                if status:
                    raise RuntimeError(f"fit failed (rc={status}); see {attempt / 'train.log'}")
                info = json.loads((attempt / "fit.json").read_text())
                row["exit_reason"] = "success"
            except ForeignGPU as reason:
                row.update(exit_reason="foreign_gpu", reason=str(reason))
            except (KeyboardInterrupt, SystemExit) as reason:
                row.update(exit_reason="cancelled", reason=repr(reason))
                error = reason
            except Exception as reason:
                row.update(exit_reason="error", reason=str(reason))
                error = reason
            finally:
                pipeline.gpu_state = original
                row["seconds"] = time.monotonic() - start
                progress = attempt / "progress.json"
                end, diagnostics = None, []
                try:
                    end = json.loads(progress.read_text()) if progress.exists() else None
                except Exception as reason:
                    diagnostics.append("heartbeat: " + repr(reason))
                try:
                    row.update(attempt_accounting(receipt, end, store.latest()))
                except Exception as reason:
                    diagnostics.append("recovery/accounting: " + repr(reason))
                    row.update(confirmed_samples_start=0 if receipt is None else receipt["samples_committed"],
                        confirmed_samples_end=None, observed_samples_end=None,
                        repeated_samples_lower_bound=None, repeated_samples_upper_bound=None,
                        heartbeat_is_exact_at_exit=False)
                if diagnostics:
                    row["accounting_errors"] = diagnostics
                    if error is None:
                        error = RuntimeError("accounting unavailable: " + str(diagnostics))
                        row.update(original_exit_reason=row["exit_reason"], exit_reason="error", reason=str(error))
                row["telemetry"] = str(attempt / "telemetry.json")
                attempts.append(row)
                write_json(attempts_file, attempts)
                write_json(attempt / "result.json", row)
            if error is not None:
                raise error
            if row["exit_reason"] == "success":
                result = finalize(store, job, attempts, info)
                write_json(root / "fit.json", result)
                return result


def fit(args):
    root, attempt = own_output(args.root), own_output(args.attempt)
    if root not in attempt.parents:
        raise ValueError("attempt output must belong to job")
    job, cfg = read_job(args.config, root)
    api = recovery_api()
    import torch
    from pytorch_lightning import Callback
    from graph2mat.tools.lightning import OrbitalMatrixCLI, SaveConfigSkipBasisTableCallback
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    from readout_comparison_callback import tensor_hash

    if not 0 < args.fraction <= .55:
        raise ValueError("allocator fraction exceeds recovery guard")
    if cfg["trainer"].get("accelerator") == "gpu":
        torch.cuda.set_per_process_memory_fraction(args.fraction)
    store = api.RecoveryStore(root, job["fingerprint"])
    receipt = store.latest()
    checkpoint = str(Path(args.checkpoint).absolute()) if args.checkpoint is not None else None
    if checkpoint != (None if receipt is None else receipt["checkpoint"]):
        raise ValueError("fit must use the last verified recovery checkpoint")
    if receipt is not None:
        store.restore_best(receipt)
    cfg = copy.deepcopy(cfg)
    schedule = cfg.pop("lr_scheduler", {})
    if schedule.get("class_path") != "torch.optim.lr_scheduler.CosineAnnealingLR":
        raise ValueError("recovery fit requires explicit cosine scheduler configuration")
    cfg["ckpt_path"] = checkpoint
    cfg["weights_only"] = False  # Only the owned, checksum-validated recovery checkpoint is accepted.
    cfg["trainer"].update(default_root_dir=str(attempt), enable_progress_bar=False)
    callbacks = cfg["trainer"].get("callbacks", [])
    if callbacks:
        if (len(callbacks) != 1 or callbacks[0]["init_args"].get("monitor") != "val_loss"
                or callbacks[0]["init_args"].get("mode", "min") != "min"
                or callbacks[0]["init_args"].get("save_top_k", 1) != 1):
            raise ValueError("fit requires the frozen single best-val_loss policy")
        callbacks[0]["init_args"]["dirpath"] = str(root / "best")
    else:
        callbacks = [{"class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                      "init_args": {"dirpath": str(root / "best"), "monitor": "val_loss",
                                    "mode": "min", "save_top_k": 1}}]
    cfg["trainer"]["callbacks"] = callbacks
    logger = cfg["trainer"].get("logger")
    if isinstance(logger, dict):
        logger.setdefault("init_args", {})["save_dir"] = str(attempt / "logs")
    recover = api.RecoveryCallback(store, cfg["seed_everything"], job["sample_ids"])
    started = time.monotonic()

    class Progress(Callback):
        def __init__(self):
            self.samples = 0 if receipt is None else receipt["samples_committed"]
            self.setup_seconds, self.train_seconds, self.val_seconds = None, 0., 0.
            self.epochs = []
            self.peak_allocated_mib, self.peak_reserved_mib = None, None

        def on_fit_start(self, trainer, model):
            if model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            if checkpoint is None and job.get("initial_state"):
                initial = own_output(job["initial_state"])
                if str(initial) not in job["files"]:
                    raise ValueError("paired initialization is not frozen")
                model.load_state_dict(torch.load(initial, map_location="cpu", weights_only=True), strict=True)
                if tensor_hash(model.state_dict().items()) != job["initial_sha256"]:
                    raise ValueError("paired initialization tensor fingerprint mismatch")

        def on_train_end(self, trainer, model):
            if model.device.type == "cuda":
                torch.cuda.synchronize()
                self.peak_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
                self.peak_reserved_mib = torch.cuda.max_memory_reserved() / 2**20

        def on_train_start(self, trainer, model):
            self.setup_seconds = time.monotonic() - started

        def on_train_epoch_start(self, trainer, model):
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            self.epoch_started = time.monotonic()
            self.epoch_samples, self.epoch_step = self.samples, trainer.global_step

        def on_train_epoch_end(self, trainer, model):
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            self.epochs.append(dict(epoch=trainer.current_epoch, seconds=time.monotonic() - self.epoch_started,
                samples=self.samples - self.epoch_samples, updates=trainer.global_step - self.epoch_step))

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            self.batch_started = time.monotonic()

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            self.train_seconds += time.monotonic() - self.batch_started
            self.samples += batch.num_graphs
            write_json(attempt / "progress.json", {"steps": trainer.global_step,
                "epoch": trainer.current_epoch, "batch_idx": batch_idx,
                "samples_observed": self.samples, "time": time.time()})

        def on_validation_batch_start(self, trainer, model, batch, batch_idx, dataloader_idx=0):
            self.val_started = time.monotonic()

        def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
            if not trainer.sanity_checking:
                self.val_seconds += time.monotonic() - self.val_started

    progress = Progress()

    class RecoveryCLI(OrbitalMatrixCLI):
        def before_fit(self):
            # Runtime callbacks stay out of the YAML and follow scientific best selection.
            self.trainer.callbacks.insert(0, progress)
            self.trainer.callbacks.append(recover)
            for key in ("batch_size", "loader_threads", "store_in_memory", "max_batch_cost",
                        "n_matrix_components", "matrix_component_policy", "symmetric_matrix"):
                if key in cfg["data"] and self.datamodule.hparams[key] != cfg["data"][key]:
                    raise ValueError("CLI replaced explicit data override: " + key)
            for key in ("optim_lr", "torch_compile", "hidden_irreps", "num_interactions"):
                if key in cfg["model"] and self.model.hparams[key] != cfg["model"][key]:
                    raise ValueError("CLI replaced explicit model override: " + key)
            native_optimizer = self.model.configure_optimizers

            def configure_with_scheduler():
                # CLI otherwise ignores a scheduler without optimizer=; keep MACE's Adam groups.
                optimizer = native_optimizer()
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **schedule.get("init_args", {}))
                return {"optimizer": optimizer, "lr_scheduler": {
                    "scheduler": scheduler, "interval": "epoch", "frequency": 1}}

            self.model.configure_optimizers = configure_with_scheduler

    cli = RecoveryCLI(LitMACEMatrixModel, api.RecoveryDataModule, args={"fit": cfg},
                     auto_configure_optimizers=False, save_config_callback=SaveConfigSkipBasisTableCallback)
    result = finalize(store, job, [], {"setup_before_training_seconds": progress.setup_seconds,
        "train_batches_seconds": progress.train_seconds, "validation_batches_seconds": progress.val_seconds,
        "checkpoint_seconds": recover.save_seconds, "child_seconds": time.monotonic() - started,
        "epochs": progress.epochs,
        "peak_allocated_mib": progress.peak_allocated_mib,
        "peak_reserved_mib": progress.peak_reserved_mib,
        "effective_lr_scheduler": schedule,
        "timing_scope": "setup includes sanity/possible compile; synchronized epoch wall includes loading/validation, "
            "excludes checkpoint save; first epoch may include lazy compile (not separately isolated); "
            "batch hooks exclude H2D/collate; parent wall includes all"})
    write_json(attempt / "fit.json", result)
    cli.datamodule.teardown(None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "pilot"):
        subcommand = commands.add_parser(name)
        subcommand.add_argument("--root", type=Path, required=True)
    command = commands.add_parser("fit")
    for name in ("root", "config", "attempt"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--fraction", type=float, required=True)
    command.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root)
    elif args.command == "pilot":
        pilot(args.root)
    else:
        fit(args)


if __name__ == "__main__":
    main()
