"""Fixed-resource w90 pilot. No queue, defaults, original data or results writes."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np

try:
    from auditorias_mejoras.audit_multitarget_w90 import ROOT, checked_path, verified_file
    from auditorias_mejoras.bench_pipeline_gpu import gpu_state, memory_fraction, process_tree_memory
except ModuleNotFoundError:
    from audit_multitarget_w90 import ROOT, checked_path, verified_file
    from bench_pipeline_gpu import gpu_state, memory_fraction, process_tree_memory

REPO = Path(__file__).resolve().parents[1]
CERT = REPO / "auditorias_mejoras/audit_multitarget_w90_20261018.json"
TARGETS = ("hamiltonian", "overlap", "density_matrix", "energy_density_matrix")
KEYS = ("hamiltonian", "overlap", "density", "energy_density")
WEIGHTS = dict(zip(TARGETS, (1., 114.9749324374332, 211.0388115121933, 3.133677534317286)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_fdf(path):
    path = checked_path(ROOT, path)
    text = path.read_text()
    if re.search(r"^\s*%include\b", text, re.I | re.M):
        raise ValueError(f"FDF include needs separate authorization: {path}")
    match = re.search(r"^\s*SystemLabel\s+(\S+)", text, re.I | re.M)
    label = match[1] if match else "siesta"
    if not re.fullmatch(r"[\w.-]+", label) or label in (".", ".."):
        raise ValueError(f"unsafe SystemLabel: {path}")
    return path


def split_ids(split):
    rows = split["rows"]
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sample ID")
    counts = Counter(r["split"] for r in rows)
    if counts != Counter(dict(train=1198, validation=150, test=150)) or counts != Counter(split["split_counts"]):
        raise ValueError("frozen split counts differ")
    return {name: [r["sample_id"] for r in rows if r["split"] == name]
            for name in ("train", "validation", "test")}


def checked_candidates(directory, manifest):
    candidates = []
    for path in sorted(checked_path(ROOT, directory).iterdir()):
        resolved = checked_path(ROOT, path)
        if resolved.is_dir():
            continue
        key = str(resolved.relative_to(ROOT.resolve()))
        if key not in manifest["sources"]:
            raise ValueError(f"uncertified reader candidate: {path}")
        verified_file(ROOT, resolved, manifest["sources"][key])
        candidates.append(resolved)
    return candidates


def certified_table(manifest):
    from graph2mat import AtomicTableWithEdges
    paths = checked_candidates(ROOT / "material_basis", manifest)
    return AtomicTableWithEdges.from_basis_glob([p for p in paths if p.name.endswith(".ion.xml")])


def graph_support(processor, data, name):
    predictions = dict(node_labels=np.ones_like(data.point_labels), edge_labels=np.ones_like(data.edge_labels))
    return processor.matrix_from_data(data, predictions=predictions, matrix_target=name, threshold=None)


def code_hashes():
    files = sorted((REPO / "src/graph2mat").rglob("*.py"))
    files += [Path(__file__), REPO / "auditorias_mejoras/audit_multitarget_w90.py",
              REPO / "auditorias_mejoras/bench_pipeline_gpu.py"]
    return {str(p.relative_to(REPO)): sha(p) for p in files}


def freeze_sources():
    if sha(CERT) != "509d94308a37d26d552449a41dd5506d2f46cdf5f58bef125b638bc220ef40f0":
        raise ValueError("certificate hash changed")
    cert = json.loads(CERT.read_text())
    for name, digest in cert["source_hashes"].items():
        path = checked_path(ROOT, ROOT / name)
        verified_file(ROOT, path, digest)
        if path.suffix.lower() == ".fdf":
            check_fdf(path)
    split = json.loads(checked_path(ROOT, ROOT / "frozen_split_manifest.json").read_text())
    ids = split_ids(split)
    paths = {r["sample_id"]: str(check_fdf(ROOT / "MD_steps" / r["sample_id"].removeprefix("md_") / "RUN.fdf"))
             for r in split["rows"]}
    for fdf in paths.values():
        checked_candidates(Path(fdf).parent, dict(sources=cert["source_hashes"]))
    return dict(dataset=str(ROOT.resolve()), sources=cert["source_hashes"], code=code_hashes(),
        split_ids=ids, paths=paths, declared_split_hash=split["split_hash"],
        excluded=["md_1198", "md_1349"],
        warnings=["md_0/md_300 train duplicate md_1200 validation; do not change splits",
                  "shared MD blocks; no independent/general OOD certificate"],
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        dirty=subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True),
        versions={n: importlib.metadata.version(n) for n in ("numpy", "scipy", "sisl", "torch", "mace-torch", "pytorch-lightning")})


def verify_sources(manifest):
    if manifest["code"] != code_hashes():
        raise ValueError("code changed since preflight")
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip() != "g2m-v2":
        raise ValueError("wrong branch")
    for name, digest in manifest["sources"].items():
        verified_file(ROOT, ROOT / name, digest)


def band_path():
    vertices = np.array([[0., 0., 0.], [1 / 3, 2 / 3, 0.], [.5, .5, 0.], [0., 0., 0.]])
    return np.concatenate([np.linspace(a, b, 50, endpoint=False) for a, b in zip(vertices[:-1], vertices[1:])]
                          + [vertices[-1:]])


def bloch(matrix, kpoints):
    g = matrix.geometry
    cells = matrix.tocsr(0).toarray().reshape(g.no, -1, g.no)
    return np.einsum("kr,irj->kij", np.exp(2j * np.pi * kpoints @ g.sc_off.T), cells)


def gaps(levels):
    if levels.shape[1] <= 4:
        return None
    return dict(direct=(levels[:, 4] - levels[:, 3]).tolist(),
                indirect=float(levels[:, 4].min() - levels[:, 3].max()))


def spectrum(h, s):
    from scipy.linalg import eigh
    for a in (h, s):
        if not np.isfinite(a).all():
            raise ValueError("nonfinite reference")
        norm = np.linalg.norm(a, axis=(1, 2))
        residual = np.linalg.norm(a - a.conj().transpose(0, 2, 1), axis=(1, 2))
        if np.any(residual / np.maximum(norm, np.finfo(float).tiny) > 1e-8):
            raise ValueError("non-Hermitian reference")
    ev = np.linalg.eigvalsh(s)
    if ev.min() <= 0:
        raise ValueError("reference overlap not positive definite")
    levels = np.array([eigh(a, b, eigvals_only=True) for a, b in zip(h, s)])
    chol = np.linalg.cholesky(s)
    # Independent full-space reduction, not a cutoff or regularization.
    reduced = np.linalg.solve(chol, h)
    reduced = np.linalg.solve(chol, reduced.conj().transpose(0, 2, 1)).conj().transpose(0, 2, 1)
    error = float(np.max(np.abs(levels - np.linalg.eigvalsh(reduced))))
    if error > 1e-8:
        raise ValueError(f"independent eigensolver disagreement: {error}")
    return dict(min_s=float(ev.min()), levels=levels.tolist(), gaps=gaps(levels), chol_error_eV=error)


def image_arrays(matrix, images):
    g = matrix.geometry
    blocks = matrix.tocsr(0).toarray().reshape(g.no, -1, g.no).transpose(1, 0, 2)
    mapping = {tuple(v): block for v, block in zip(g.sc_off, blocks)}
    return np.array([mapping.get(v, np.zeros((g.no, g.no))) for v in images])


def reference_floor(bundle, processor, data, kpoints):
    from graph2mat.core.data.physical import _weighted_reference_rows
    reconstructed = {key: processor.matrix_from_data(data, matrix_target=name, threshold=None)
                     for name, key in zip(TARGETS, KEYS)}
    targets = {}
    for name, key in zip(TARGETS, KEYS):
        a, b = bundle[key], reconstructed[key]
        images = sorted(set(map(tuple, a.geometry.sc_off)) | set(map(tuple, b.geometry.sc_off)))
        av, bv = image_arrays(a, images), image_arrays(b, images)
        # Ones on every graph entry, even when its reference label is zero.
        mask = graph_support(processor, data, name)
        support = image_arrays(mask, images) != 0
        residual = av - bv
        targets[name] = dict(max_error=float(np.abs(residual).max()),
            frobenius=float(np.linalg.norm(residual)), omitted_frobenius=float(np.linalg.norm(av[~support])))
    physics = {}
    for label, matrices in (("original", bundle), ("reconstructed", reconstructed)):
        h, s, d, e = (matrices[k] for k in KEYS)
        try:
            n = _weighted_reference_rows(d, 0, s.tocsr(0), "density").sum()
            hd = _weighted_reference_rows(d, 0, h.tocsr(0), "density")
            se = _weighted_reference_rows(e, 0, s.tocsr(0), "energy_density")
            physics[label] = dict(N=float(n), max_HD_SE=float(np.max(np.abs(hd - se))))
        except ValueError as error:
            physics[label] = dict(unsupported=str(error))
    return dict(targets=targets, physics=physics,
        original=spectrum(bloch(bundle["hamiltonian"], kpoints), bloch(bundle["overlap"], kpoints)),
        reconstructed=spectrum(bloch(reconstructed["hamiltonian"], kpoints), bloch(reconstructed["overlap"], kpoints)))


def preflight(out):
    out = Path(out)
    out.mkdir()  # Exclusive: never adopt/overwrite an old attempt.
    started = time.monotonic()
    try:
        from graph2mat import MatrixDataProcessor
        from graph2mat.core.data.multitarget import prepare_multitarget_data
        from graph2mat.core.data.physical import _read_reference_bundle
        manifest = freeze_sources()
        if manifest["branch"] != "g2m-v2":
            raise ValueError("wrong branch")
        write_json(out / "manifest.json", manifest)
        table = certified_table(manifest)
        processor = MatrixDataProcessor(basis_table=table, out_matrix="hamiltonian", sub_point_matrix=False,
            symmetric_matrix=True, matrix_targets=list(TARGETS), matrix_target_spin=1)
        rows, mesh = [], None
        for i, (sample_id, fdf) in enumerate(manifest["paths"].items()):
            bundle, _ = _read_reference_bundle(check_fdf(fdf), need_density=True, need_hamiltonian=True,
                need_energy_density=True, require_mesh=True, processor=processor)
            if mesh is None:
                mesh = bundle["kpoints"]
                write_json(out / "kpoints.json", dict(mesh=mesh.tolist(), weights=bundle["kweights"].tolist(), path=band_path().tolist()))
            elif not np.array_equal(mesh, bundle["kpoints"]):
                raise ValueError(f"kmesh changed: {sample_id}")
            data = prepare_multitarget_data(processor, **bundle)
            if not data.matrix_known_nodes.all() or not data.matrix_known_edges.all():
                raise ValueError(f"incomplete labels: {sample_id}")
            result = reference_floor(bundle, processor, data, np.concatenate((mesh, band_path())))
            result.update(sample_id=sample_id, mesh_size=len(mesh), path_size=151)
            rows.append(result)
            if (i + 1) % 100 == 0:
                print(f"preflight {i+1}/1498", flush=True)
        # Training setup is CPU-only; defined by the second task below.
        prepare_pair(out, table, processor, manifest)
        verify_sources(manifest)
        result = dict(status="complete", structures=len(rows), rows=rows, seconds=time.monotonic() - started,
                      kpoints_sha256=sha(out / "kpoints.json"))
        write_json(out / "preflight.json", result)
        return result
    except Exception as error:
        write_json(out / "blocked.json", dict(status="blocked", reason=f"{type(error).__name__}: {error}"))
        raise


def batch_order(ids, seed, updates):
    import torch
    result, epoch = [], 0
    if not ids or updates < 1:
        raise ValueError("empty train order")
    while len(result) < updates:
        order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        result.extend([[ids[i] for i in order[j:j + 32]] for j in range(0, len(ids), 32)])
        epoch += 1
    return result[:updates]


def make_scheduler(optimizer):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    return CosineAnnealingLR(optimizer, T_max=4000, eta_min=1e-5)


def make_arm(arm, basis_table, backbone_state=None, *, small=False):
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    from graph2mat.core.data.metrics import gauge_invariant_mse
    from graph2mat.bindings.e3nn import E3nnLinearEdgeBlock
    from graph2mat.bindings.e3nn.modules.node_operations import E3nnLinearTSQNodeBlock
    if arm not in ("h", "joint"):
        raise ValueError("invalid arm")
    kwargs = dict(basis_table=basis_table, symmetric_matrix=True, hidden_irreps="48x0e+48x1o+48x2e+48x3o",
        edge_hidden_irreps="4x0e+4x1o+4x2e", num_interactions=3, correlation=3,
        num_bessel=10, max_ell=3, num_polynomial_cutoff=6, keep_last_layer_irreps=True,
        lean_mace=True, avg_num_neighbors=1., node_block_readout=E3nnLinearTSQNodeBlock,
        edge_block_readout=E3nnLinearEdgeBlock, optim_lr=.0018, optim_wdecay=5e-7, optim_amsgrad=True,
        torch_compile=False, gradient_checkpointing=False, output_scaleshift=False,
        edge_update_layers=0, readout_per_interaction=False, pair_envelope=False,
        long_range=False, hamiltonian_local_context=False, coefficient_training=False, cueq=False)
    if small:  # CPU unit fixtures only; CLI cannot select a different architecture.
        kwargs.update(hidden_irreps="2x0e+2x1o", edge_hidden_irreps="2x0e+2x1o", max_ell=1,
                      num_interactions=2, correlation=1, num_bessel=2)
    if arm == "h":
        kwargs.update(n_matrix_components=2, n_output_components=1, loss=gauge_invariant_mse,
                      loss_kwargs=dict(edge_weight=1., overlap_weight=0.))
    else:
        kwargs.update(matrix_targets=list(TARGETS), matrix_target_spin=1, matrix_target_weights=WEIGHTS,
            matrix_target_losses={t: dict(name="gauge_invariant_mse" if t == "hamiltonian" else "elementwise_mse",
                                         kwargs=dict(edge_weight=1.)) for t in TARGETS})
    model = LitMACEMatrixModel(**kwargs).float()
    model.log = lambda *args, **kwargs: None  # Retain actual metric computation, no Trainer backend.
    if backbone_state is not None:
        model.model.mace.load_state_dict(backbone_state, strict=True)
    return model


def h_only_batch(batch):
    result = batch.clone()
    result.point_labels = result.point_labels[:, :2]
    result.edge_labels = result.edge_labels[:, :2]
    return result


def require_finite(tensors, name):
    import torch
    tensors = [t for t in tensors if t is not None]
    if not tensors or not bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all()):
        raise ValueError(f"nonfinite/absent {name}")


def check_pair(models, batch, actual_h=None):
    import torch
    h, joint = models
    for key, value in h.model.mace.state_dict().items():
        torch.testing.assert_close(value, joint.model.mace.state_dict()[key], rtol=0, atol=0)
    h.eval(), joint.eval()
    hb = h_only_batch(batch) if actual_h is None else actual_h
    projected = h_only_batch(batch)
    for field in ("positions", "node_attrs", "edge_index", "shifts", "cell", "point_types",
                  "edge_types", "neigh_isc", "nsc", "point_labels", "edge_labels", "ptr"):
        try:
            torch.testing.assert_close(hb[field], projected[field], rtol=0, atol=0)
        except AssertionError as error:
            raise ValueError(f"legacy/joint labels or graph mismatch: {field}") from error
    with torch.no_grad():
        a, b = h(hb), joint(batch)
        torch.testing.assert_close(a["node_feats"], b["node_feats"], rtol=0, atol=0)
        # Use exactly the same H prediction for the two loss implementations.
        pn, pe = b["node_labels"].clone(), b["edge_labels"].clone()
        pn[:, 1:], pe[:, 1:] = batch.point_labels[:, 1:], batch.edge_labels[:, 1:]
        lh, _ = h.loss_fn(nodes_pred=pn[:, :1], edges_pred=pe[:, :1], nodes_ref=hb.point_labels,
            edges_ref=hb.edge_labels, batch=hb, basis_table=h.basis_table)
        _, stats = joint.loss_fn(nodes_pred=pn, edges_pred=pe, nodes_ref=batch.point_labels,
            edges_ref=batch.edge_labels, batch=batch, basis_table=joint.basis_table)
        lj = stats["target/hamiltonian/matrix_sum"] / stats["target/hamiltonian/matrix_count"]
        torch.testing.assert_close(lh, lj, rtol=1e-6, atol=1e-7)
    return dict(shared_features_exact=True, backbone_exact=True, h_loss_difference=float(abs(lh - lj)))


def serializable(value):
    from graph2mat import BasisTableWithEdges
    if isinstance(value, BasisTableWithEdges):
        return dict(basis_size=value.basis_size.tolist(), R=value.R.tolist(), description=str(value))
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, type) or callable(value):
        return value.__module__ + "." + value.__qualname__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unserializable config: {type(value)}")


def prepare_pair(out, table, processor, manifest):
    import torch
    from graph2mat.bindings.torch import TorchBasisMatrixDataset
    from graph2mat.tools.lightning.data import TypeLayoutCollate
    from torch_geometric.data import Batch
    torch.manual_seed(0)
    h = make_arm("h", table)
    joint = make_arm("joint", table, h.model.mace.state_dict())
    paths = [manifest["paths"][s] for s in manifest["split_ids"]["train"][:2]]
    sample = TorchBasisMatrixDataset(paths, processor)
    batch = TypeLayoutCollate(Batch.from_data_list, table)([sample[0], sample[1]])
    legacy = TorchBasisMatrixDataset(paths, arm_processor(table, "h"))
    hbatch = TypeLayoutCollate(Batch.from_data_list, table)([legacy[0], legacy[1]])
    checks = check_pair((h, joint), batch, hbatch)
    order = batch_order(manifest["split_ids"]["train"], 0, 120)
    write_json(out / "order.json", order)
    files = ["order.json"]
    for arm, model in (("h", h), ("joint", joint)):
        optimizer = model.configure_optimizers()
        names = {id(p): n for n, p in model.named_parameters()}
        groups = [{**{k: serializable(v) for k, v in group.items() if k != "params"},
                   "params": [names[id(p)] for p in group["params"]]} for group in optimizer.param_groups]
        config = dict(model=serializable(dict(model.hparams)), optimizer=dict(defaults=serializable(optimizer.defaults), groups=groups),
            data=dict(batch_size=32, drop_last=False, loader_threads=0, pin_memory=True, store_in_memory=True,
                sub_point_matrix=False, symmetric_matrix=True, labels="H/S" if arm == "h" else "H/S/DM/EDM"),
            seed=0, fp32=True, tf32=False, scheduler=dict(T_max=4000, eta_min=1e-5, interval="optimizer_update"))
        write_json(out / f"effective-config-{arm}.json", config)
        with (out / f"initial-{arm}.pt").open("xb") as stream:
            torch.save(model.state_dict(), stream)
        files += [f"effective-config-{arm}.json", f"initial-{arm}.pt"]
    write_json(out / "prepared.json", dict(checks=checks, hashes={name: sha(out / name) for name in files}))


def reset_metrics(model):
    for attr in ("_epoch_losses", "_epoch_rmse", "_epoch_errors"):
        for metric in getattr(model, attr, {}).values():
            metric.reset()


def train_updates(model, batches, arm, device, *, warmup=20, stream=None):
    import torch
    model.to(device).train()
    optimizer = model.configure_optimizers()
    scheduler = make_scheduler(optimizer)
    cuda = device == "cuda"
    rows, warmup_peaks = [], None
    iterator = iter(batches)
    iteration = 0
    while True:
        if iteration == warmup and cuda:
            torch.cuda.synchronize()
            warmup_peaks = dict(allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                                reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
            torch.cuda.reset_peak_memory_stats()
        if iteration % 38 == 0:
            reset_metrics(model)
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            cpu = next(iterator)
        except StopIteration:
            break
        if arm == "h":
            cpu = h_only_batch(cpu)
        data_s = time.perf_counter() - start
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if cuda else None
        if cuda:
            events[0].record()
        batch = model.transfer_batch_to_device(cpu, device, 0)
        if cuda:
            events[1].record()
        optimizer.zero_grad(set_to_none=True)
        result = model.training_step(batch, iteration)
        require_finite([result["loss"]], "loss")
        if cuda:
            events[2].record()
        result["loss"].backward()
        require_finite([p.grad for p in model.parameters() if p.grad is not None], "gradients")
        if cuda:
            events[3].record()
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        if cuda:
            events[4].record()
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        row = dict(update=iteration + 1, measured=iteration >= warmup, samples=batch.num_graphs,
                   ids=getattr(batch, "sample_id", None),
                   loss=float(result["loss"].detach()), lr=lr, wall_s=elapsed, data_s=data_s)
        if cuda:
            row["cuda_ms"] = [a.elapsed_time(b) for a, b in zip(events[:-1], events[1:])]
        rows.append(row)
        if stream is not None:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        del result, batch, cpu
        iteration += 1
    require_finite(list(model.parameters()), "final parameters")
    return dict(updates=rows, warmup_peaks=warmup_peaks,
        measured_peaks=None if not cuda else dict(allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                                                  reserved_mib=torch.cuda.max_memory_reserved() / 2**20))


def validate_h(model, batches, arm, device):
    import torch
    from graph2mat.core.data.metrics import gauge_invariant_mse
    metric = gauge_invariant_mse(edge_weight=1., overlap_weight=0.)
    total, elements, samples = 0., 0, 0
    model.eval()
    reset_metrics(model)
    started = time.perf_counter()
    with torch.no_grad():
        for index, cpu in enumerate(batches):
            if arm == "h":
                cpu = h_only_batch(cpu)
            batch = model.transfer_batch_to_device(cpu, device, 0)
            out = model.validation_step(batch, index)
            require_finite([out["loss"]], "validation")
            hb = h_only_batch(batch)
            pn = out["node_labels"].reshape(-1, 1) if arm == "h" else out["node_labels"][:, :1]
            pe = out["edge_labels"].reshape(-1, 1) if arm == "h" else out["edge_labels"][:, :1]
            loss, _ = metric(nodes_pred=pn, edges_pred=pe,
                nodes_ref=hb.point_labels, edges_ref=hb.edge_labels, batch=hb, basis_table=model.basis_table)
            count = len(hb.point_labels) + len(hb.edge_labels)
            total += float(loss) * count
            elements += count
            samples += batch.num_graphs
    if device == "cuda":
        torch.cuda.synchronize()
    return dict(samples=samples, elements=elements, h_gauge_mse=total / elements,
                seconds=time.perf_counter() - started)


def arm_processor(table, arm):
    from graph2mat import MatrixDataProcessor
    kw = dict(matrix_targets=list(TARGETS), matrix_target_spin=1) if arm == "joint" else dict(matrix_component_policy="h_and_overlap", n_matrix_components=2)
    return MatrixDataProcessor(basis_table=table, out_matrix="hamiltonian", sub_point_matrix=False,
                               symmetric_matrix=True, **kw)


def run_arm(out, arm, device, warmup=20, steps=100):
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    from graph2mat.bindings.torch import TorchBasisMatrixDataset, InMemoryData
    from graph2mat.tools.lightning.data import TypeLayoutCollate
    started = time.perf_counter()
    manifest = json.loads((out / "manifest.json").read_text())
    verify_sources(manifest)
    table = certified_table(manifest)
    processor = arm_processor(table, arm)
    ids = manifest["split_ids"]
    for name in ("train", "validation"):
        for sample_id in ids[name]:
            checked_candidates(Path(manifest["paths"][sample_id]).parent, manifest)
    datasets = {name: InMemoryData(TorchBasisMatrixDataset([check_fdf(manifest["paths"][s]) for s in ids[name]], processor))
                for name in ("train", "validation")}
    for name, dataset in datasets.items():
        for sample, sample_id in zip(dataset.data_objects, ids[name]):
            sample.sample_id = sample_id
    order = json.loads((out / "order.json").read_text())[:warmup + steps]
    indices = {s: i for i, s in enumerate(ids["train"])}
    collate = TypeLayoutCollate(Batch.from_data_list, table)
    train = DataLoader(datasets["train"], batch_sampler=[[indices[s] for s in b] for b in order],
                       collate_fn=collate, num_workers=0, pin_memory=device == "cuda")
    val = DataLoader(datasets["validation"], batch_size=32, collate_fn=collate,
                     num_workers=0, pin_memory=device == "cuda")
    model = make_arm(arm, table)
    model.load_state_dict(torch.load(out / f"initial-{arm}.pt", map_location="cpu", weights_only=True), strict=True)
    setup_s = time.perf_counter() - started
    with (out / f"updates-{arm}.jsonl").open("x") as stream:
        result = train_updates(model, train, arm, device, warmup=warmup, stream=stream)
    result.update(arm=arm, setup_s=setup_s, order_sha256=sha(out / "order.json"),
                  validation=validate_h(model, val, arm, device))
    result["total_s"] = time.perf_counter() - started
    verify_sources(manifest)
    write_json(out / f"result-{arm}.json", result)
    return result


def guard_reason(state, compute_pids, owned_pids, now, deadline):
    if now >= deadline:
        return "deadline"
    try:
        total, free, temp = (float(state[k]) for k in ("memory.total", "memory.free", "temperature.gpu"))
        if not all(math.isfinite(x) for x in (total, free, temp)) or total <= 0 or not 0 <= free <= total:
            return "telemetry"
    except (KeyError, TypeError, ValueError):
        return "telemetry"
    if free < .25 * total:
        return "VRAM"
    if temp >= 80:
        return "temperature"
    if compute_pids - owned_pids:
        return "foreign"
    return None


def query_gpu(timeout=2.):
    # Unlike the historical helper, bound each in-flight driver query by the deadline.
    args = ["nvidia-smi", "-i", "0"]
    text = subprocess.check_output(args + ["--query-gpu=memory.total,memory.free,temperature.gpu,uuid",
        "--format=csv,noheader,nounits"], text=True, timeout=timeout).strip().split(",")
    state = dict(zip(("memory.total", "memory.free", "temperature.gpu"), map(float, text[:3])))
    uuid = text[3].strip()
    output = subprocess.check_output(args + ["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                                     text=True, timeout=timeout)
    pids = set()
    for row in output.splitlines():
        gpu, pid = row.split(",")
        if gpu.strip() == uuid:
            pids.add(int(pid.strip()))
    return state, pids, uuid


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def terminate_group(proc):
    # Only called for a Popen we created with start_new_session=True.
    deadline = time.monotonic() + 10
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=max(.001, deadline - time.monotonic()))
    while group_alive(proc.pid):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"owned process group {proc.pid} still alive after cleanup")
        time.sleep(.02)


def initial_gpu_state(timeout=2.):
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError("cannot safely bound initial telemetry: existing timer")
    def expired(*args):
        raise TimeoutError("initial GPU telemetry timeout")
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return gpu_state()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def wait_context_exit(pid, uuid, deadline):
    """Allow only an absent, just-exited leader's stale CUDA record, at most2s."""
    until = min(deadline, time.monotonic() + 2.)
    observations = []
    while True:
        now = time.monotonic()
        left = until - now
        if left <= 0:
            raise RuntimeError("deadline" if now >= deadline else "CUDA context release timeout")
        state, pids, current_uuid = query_gpu(timeout=min(2., left / 2))
        if current_uuid != uuid:
            raise RuntimeError("GPU identity changed")
        # A reused PID has a /proc entry and must never inherit this exemption.
        stale = {pid} if not Path(f"/proc/{pid}").exists() else set()
        reason = guard_reason(state, pids, stale, time.monotonic(), deadline)
        if reason:
            raise RuntimeError(reason)
        state.update(time=time.time(), phase="context_release", compute_pids=sorted(pids))
        observations.append(state)
        if not pids:
            return observations
        time.sleep(min(.1, max(0., until - time.monotonic())))


def summarize(results, telemetry, *, expected_order=None, integrity=False):
    arms = {}
    complete = set(results) == {"h", "joint"} and integrity and expected_order is not None
    for name, result in results.items():
        rows = result.get("updates", [])
        measured = [r for r in rows if r["measured"]]
        valid = (len(rows) == 120 and len(measured) == 100
                 and [r["update"] for r in rows] == list(range(1, 121))
                 and sum(r["samples"] for r in measured) == 3146
                 and result.get("validation", {}).get("samples") == 150
                 and math.isfinite(result["validation"].get("h_gauge_mse", float("nan"))))
        valid = valid and all(math.isfinite(r["loss"]) and math.isfinite(r["wall_s"]) and r["wall_s"] > 0
            and math.isclose(r["lr"], 1e-5 + (.0018 - 1e-5) * (1 + math.cos(math.pi * i / 4000)) / 2, rel_tol=1e-8)
            for i, r in enumerate(rows))
        valid = valid and expected_order is not None and [r.get("ids") for r in rows] == expected_order
        complete &= valid
        if measured:
            wall = np.array([r["wall_s"] for r in measured])
            validation = result.get("validation", {})
            val = validation.get("seconds")
            setup = result.get("setup_s")
            arms[name] = dict(updates=len(rows), measured=len(measured),
                wall_s=dict(median=float(np.median(wall)), p95=float(np.percentile(wall, 95)),
                            min=float(wall.min()), max=float(wall.max())),
                samples_per_s=sum(r["samples"] for r in measured) / float(wall.sum()),
                measured_peaks=result.get("measured_peaks"), warmup_peaks=result.get("warmup_peaks"),
                setup_s=setup, validation=validation, estimate_4000_s=None)
            if valid and setup is not None and val is not None:
                overhead = setup + 20 * val
                arms[name]["estimate_4000_s"] = dict(central=float(wall.mean() * 4000 + overhead),
                    low=float(np.percentile(wall, 10) * 4000 + overhead),
                    high_with_25pct_margin=float((np.percentile(wall, 90) * 4000 + overhead) * 1.25))
        else:
            arms[name] = dict(updates=len(rows), measured=0, estimate_4000_s=None)
    if set(results) == {"h", "joint"}:
        complete &= (results["h"].get("order_sha256") is not None
                     and results["h"].get("order_sha256") == results["joint"].get("order_sha256"))
    if not complete:
        for arm in arms.values():
            arm["estimate_4000_s"] = None
    return dict(status="complete" if complete else "incomplete", arms=arms,
        gpu_peak_used_mib=max((r["memory.total"] - r["memory.free"] for r in telemetry), default=None),
        gpu_min_free_mib=min((r["memory.free"] for r in telemetry), default=None),
        limitations="Fixed H→joint order, seed0, no independent repeats/causal CI or accuracy conclusion; "
                    "20% reserve is best-effort against external allocations; 4000-update estimates are not guarantees.")


def pilot(out):
    out = Path(out)
    write_json(out / "started.json", dict(time=time.time(), pid=os.getpid()))
    results, telemetry, reason, deadline = {}, [], None, None
    try:
        manifest = json.loads((out / "manifest.json").read_text())
        verify_sources(manifest)
        certificate = json.loads((out / "preflight.json").read_text())
        if certificate["status"] != "complete" or certificate["structures"] != 1498:
            raise ValueError("preflight incomplete")
        prepared = json.loads((out / "prepared.json").read_text())
        for name, digest in prepared["hashes"].items():
            if sha(out / name) != digest:
                raise ValueError(f"prepared artifact changed: {name}")
        initial = initial_gpu_state()
        reason = guard_reason(initial, set(), set(), 0, 1)
        if reason or initial["temperature.gpu"] >= 75:
            raise RuntimeError(reason or "initial temperature >=75C")
        with (out / "telemetry.jsonl").open("x") as stream:
            for arm in ("h", "joint"):
                state, pids, uuid = query_gpu()
                now = time.monotonic()
                reason = guard_reason(state, pids, set(), now, deadline or now + 1)
                if reason or state["temperature.gpu"] >= 75:
                    raise RuntimeError(reason or "initial temperature >=75C")
                fraction = memory_fraction(state["memory.total"], state["memory.free"])
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONDONTWRITEBYTECODE="1",
                           OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
                if deadline is None:
                    deadline = time.monotonic() + 900
                with (out / f"worker-{arm}.log").open("x") as log:
                    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--out", str(out),
                        "--arm", arm, "--fraction", str(fraction)], env=env, cwd=REPO,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        while proc.poll() is None:
                            left = deadline - time.monotonic()
                            if left <= 0:
                                raise RuntimeError("deadline")
                            state, pids, current_uuid = query_gpu(timeout=min(2., left / 2))
                            if current_uuid != uuid:
                                raise RuntimeError("GPU identity changed")
                            host = process_tree_memory(proc.pid)
                            # poll() has not reaped this leader: its PID cannot be reused,
                            # even when a zombie/exit race leaves no RSS in /proc.
                            owned = {proc.pid}
                            for pid in host["pids"]:
                                try:
                                    if os.getpgid(pid) == proc.pid:
                                        owned.add(pid)
                                except ProcessLookupError:
                                    pass
                            reason = guard_reason(state, pids, owned, time.monotonic(), deadline)
                            state.update(time=time.time(), arm=arm, host=host, compute_pids=sorted(pids))
                            telemetry.append(state)
                            stream.write(json.dumps(state, allow_nan=False) + "\n")
                            stream.flush()
                            if reason:
                                raise RuntimeError(reason)
                            time.sleep(min(.5, max(0., deadline - time.monotonic())))
                        if proc.returncode:
                            raise RuntimeError(f"{arm} worker exit {proc.returncode}")
                    finally:
                        original_error = sys.exc_info()[1]
                        try:
                            terminate_group(proc)
                        except Exception as cleanup_error:
                            raise RuntimeError(f"{original_error or 'worker finished'}; cleanup: {cleanup_error}") from cleanup_error
                for state in wait_context_exit(proc.pid, uuid, deadline):
                    state["arm"] = arm
                    telemetry.append(state)
                    stream.write(json.dumps(state, allow_nan=False) + "\n")
                    stream.flush()
                results[arm] = json.loads((out / f"result-{arm}.json").read_text())
        if time.monotonic() > deadline:
            raise RuntimeError("deadline")
        verify_sources(manifest)
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
    for arm in ("h", "joint"):
        if arm not in results and (out / f"updates-{arm}.jsonl").exists():
            rows = []
            for line in (out / f"updates-{arm}.jsonl").read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
            results[arm] = dict(updates=rows)
    expected = json.loads((out / "order.json").read_text()) if (out / "order.json").exists() else None
    if reason is None and any(r.get("order_sha256") != prepared["hashes"]["order.json"] for r in results.values()):
        reason = "order hash does not match frozen preparation"
    summary = summarize(results, telemetry, expected_order=expected, integrity=reason is None)
    summary.update(reason=reason, status=summary["status"] if reason is None else "blocked_or_aborted")
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "pilot", "worker"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arm", choices=("h", "joint"))
    parser.add_argument("--fraction", type=float)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != REPO / "auditorias_mejoras":
        parser.error("output must be a new direct child of graph2mat/auditorias_mejoras")
    import torch
    import graph2mat
    if Path(graph2mat.__file__).resolve() != REPO / "src/graph2mat/__init__.py":
        raise RuntimeError("wrong graph2mat import")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.command == "preflight":
        preflight(out)
    elif args.command == "pilot":
        sys.exit(0 if pilot(out)["status"] == "complete" else 2)
    else:
        if not args.arm or args.fraction is None or not (0 < args.fraction <= .55):
            parser.error("internal worker needs arm and safe allocator fraction")
        torch.cuda.set_per_process_memory_fraction(args.fraction)
        run_arm(out, args.arm, "cuda")


if __name__ == "__main__":
    main()
