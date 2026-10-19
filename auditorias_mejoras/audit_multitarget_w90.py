"""Fixed-resource CPU audit. Sources are read-only; JSON goes only to stdout."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np


ROOT = Path("/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/"
            "graphene_w90_joint/graphene_w90_phase1_iid1500")


def checked_path(root, path):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f"source outside authorized dataset: {path}")
    return resolved


def verified_file(root, path, expected):
    path = checked_path(root, path)
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"source hash mismatch: {path}")
    return actual


def scf_summary(text):
    steps = list(re.finditer(r"Begin MD step\s*=\s*(\d+)", text))
    cycles = []
    for index, match in enumerate(steps):
        segment = text[match.end():steps[index + 1].start() if index + 1 < len(steps) else len(text)]
        cycles.append(len(re.findall(r"SCF cycle converged after \d+ iterations", segment)))
    ids = [int(match[1]) for match in steps]
    return dict(md_steps=len(steps), converged_cycles=sum(cycles),
        all_steps_converged=bool(steps) and len(set(ids)) == len(ids)
            and all(count == 1 for count in cycles) and "Job completed" in text
            and not re.search(r"SCF.*(?:did not|not converged|NOT_CONV)", text),
        electron_counts=sorted(set(float(x) for x in
            re.findall(r"Total number of electrons:\s*([\d.]+)", text))))


def validate_native_geometry(native, attached):
    # TSHS has coordinates/orbital counts, not reliable species: validate Z via FDF/ORB_INDX/XML.
    if (native.na != attached.na or native.no != attached.no
            or not np.array_equal(native.atoms.orbitals, attached.atoms.orbitals)
            or not np.allclose(native.xyz, attached.xyz, rtol=0, atol=1e-8)
            or not np.allclose(native.cell, attached.cell, rtol=0, atol=1e-8)):
        raise ValueError("native TSHS and FDF geometry differ")
    return dict(xyz_max=float(np.max(np.abs(native.xyz - attached.xyz))),
                cell_max=float(np.max(np.abs(native.cell - attached.cell))))


def main():
    from graph2mat import AtomicTableWithEdges, MatrixDataProcessor
    from graph2mat.core.data.multitarget import prepare_multitarget_data
    from graph2mat.core.data.physical import _read_reference_bundle, _weighted_reference_rows
    import sisl

    started = time.monotonic()
    root = ROOT.resolve(strict=True)
    hashes = {}

    def record(path, expected=None):
        path = checked_path(root, path)
        digest = verified_file(root, path, expected)
        if path in hashes and hashes[path] != digest:
            raise ValueError(f"source changed during audit: {path}")
        hashes[path] = digest
        return path

    def read_json(path):
        return json.loads(record(path).read_text())

    manifest = read_json(root / "benchmark_dataset_manifest.json")
    split = read_json(root / "frozen_split_manifest.json")
    split_policy = read_json(root / "splits/split_summary.json")
    blocks = read_json(root / "md_temperature_blocks_manifest.json")
    artifact_validation = read_json(root / "artifact_validation.json")
    for name, digest in manifest["basis_hashes"].items():
        record(root / "material_basis" / name, digest)
    for name, digest in manifest["pseudopotential_hashes"].items():
        record(root / (name + ".psf"), digest)
    table = AtomicTableWithEdges.from_basis_dir(root / "material_basis")
    targets = ("hamiltonian", "overlap", "density_matrix", "energy_density_matrix")
    processor = MatrixDataProcessor(basis_table=table, out_matrix="hamiltonian",
        sub_point_matrix=False, symmetric_matrix=True, matrix_targets=list(targets), matrix_target_spin=1)
    result = dict(dataset=str(root), source_manifest_ready=manifest["benchmark_ready"],
        source_manifest_valid=manifest["validation_status"], artifact_invalid=artifact_validation["invalid_snapshots"],
        descriptor=processor.matrix_target_layout.to_dict(), basis=repr(table), blocks={}, samples=[],
        split_counts=split["split_counts"], split_hash=split["split_hash"], split_policy=split_policy,
        scope="CPU read-only, no training/inference/GPU; no source repair or split changes")
    for block in blocks["blocks"]:
        path = record(root / "md_temperature_blocks" / block["block_id"] / "RUN.out")
        summary = scf_summary(path.read_text(errors="replace"))
        summary["declared_snapshots"] = block["n_snapshots"]
        result["blocks"][block["block_id"]] = summary

    rows_by_id = {}
    for row in split["rows"]:
        if row["sample_id"] in rows_by_id:
            raise ValueError("duplicate sample ID in frozen splits")
        rows_by_id[row["sample_id"]] = row
    if Counter(row["split"] for row in split["rows"]) != Counter(split["split_counts"]):
        raise ValueError("frozen split counts disagree with rows")
    for row in manifest["samples"]:
        for name, expected in row["artifact_sha256"].items():
            record(row["artifact_paths"][name], expected)
    for row in split["rows"]:
        for name, expected in row["artifact_sha256"].items():
            record(row["artifact_paths"][name], expected)

    content_splits = defaultdict(set)
    block_splits = defaultdict(Counter)
    label_moments = {name: {target: dict(count=0, sum=0., sum_square=0.) for target in targets}
                     for name in split["split_counts"]}
    seen = set()
    for index, sample in enumerate(manifest["samples"]):
        directory = checked_path(root, sample["sample_dir"])
        sample_id = "md_" + directory.name
        metadata = read_json(directory / "metadata.json")
        fdf_path = record(directory / "RUN.fdf")
        if re.search(r"^\s*%include\b", fdf_path.read_text(), re.IGNORECASE | re.MULTILINE):
            raise ValueError(f"FDF includes require separate path authorization: {fdf_path}")
        fdf = sisl.get_sile(fdf_path)
        label = fdf.get("SystemLabel", default="siesta")
        tshs_path = record(directory / (label + ".TSHS"))
        record(directory / (label + ".TSDE"))
        raw_geometry = sisl.get_sile(tshs_path).read_geometry()
        bundle, _ = _read_reference_bundle(fdf_path, need_density=True, need_hamiltonian=True,
            need_energy_density=True, require_mesh=True, processor=processor)
        data = prepare_multitarget_data(processor, **bundle)
        s, h, d, e = (bundle[key] for key in ("overlap", "hamiltonian", "density", "energy_density"))
        geometry = s.geometry
        geometry_delta = validate_native_geometry(raw_geometry, geometry)
        orbital_rows = []
        for line in record(directory / (label + ".ORB_INDX")).read_text().splitlines():
            fields = line.split()
            if len(fields) >= 16 and fields[0].isdigit() and int(fields[0]) <= geometry.no:
                orbital_rows.append((int(fields[1]) - 1, fields[3], *(int(fields[k]) for k in (5, 6, 7))))
        expected_orbitals = [(i, atom.symbol, orbital.n, orbital.l, orbital.m)
                             for i, atom in enumerate(geometry.atoms) for orbital in atom.orbitals]
        if orbital_rows != expected_orbitals:
            raise ValueError(f"orbital ordering differs from ion basis: {sample_id}")
        phase = np.exp(2j * np.pi * bundle["kpoints"] @ geometry.sc_off.T)
        sk = np.einsum("kr,irj->kij", phase,
                       s.tocsr(0).toarray().reshape(geometry.no, -1, geometry.no))
        if np.max(np.abs(sk - sk.conj().transpose(0, 2, 1))) > 1e-8:
            raise ValueError(f"non-Hermitian S(k): {sample_id}")
        eigenvalues = np.linalg.eigvalsh(sk)
        if eigenvalues.min() <= 0:
            raise ValueError(f"non-positive S(k), no clipping allowed: {sample_id}")
        physics_error = None
        try:
            populations = _weighted_reference_rows(d, 0, s.tocsr(0), "density")
            hd = _weighted_reference_rows(d, 0, h.tocsr(0), "density")
            se = _weighted_reference_rows(e, 0, s.tocsr(0), "energy_density")
            population = float(populations.sum())
            energy_error = float(np.max(np.abs(hd - se)))
        except ValueError as error:
            population, energy_error, physics_error = None, None, str(error)
        row = rows_by_id.get(sample_id)
        split_name = row["split"] if row else "excluded_gap"
        if row:
            for key in ("tshs", "tsde", "hsx"):
                if sample["artifact_sha256"][key] != row["artifact_sha256"]["reference_" + key]:
                    raise ValueError(f"split/source matrices differ: {sample_id}")
            content_splits[(sample["artifact_sha256"]["tshs"], sample["artifact_sha256"]["tsde"])].add(split_name)
            block_splits[metadata["source_block_id"]][split_name] += 1
            for target, columns in processor.matrix_target_layout.slices.items():
                moment = label_moments[split_name][target]
                for values, known in ((data.point_labels[:, columns], data.matrix_known_nodes[:, columns]),
                                      (data.edge_labels[:, columns], data.matrix_known_edges[:, columns])):
                    valid = values[known]
                    moment["count"] += int(valid.size)
                    moment["sum"] += float(valid.sum())
                    moment["sum_square"] += float(np.square(valid).sum())
        seen.add(sample_id)
        result["samples"].append(dict(sample_id=sample_id, split=split_name,
            source_block=metadata["source_block_id"], source_frame=metadata["source_frame_index"],
            temperature_K=metadata["temperature_K"], atoms=int(geometry.na), orbitals=int(geometry.no),
            native_geometry_delta=geometry_delta,
            nsc=geometry.nsc.tolist(), mesh_size=len(bundle["kpoints"]), min_s=float(eigenvalues.min()),
            max_s_condition=float(np.max(eigenvalues[:, -1] / eigenvalues[:, 0])),
            population=population, energy_reference_max_orbital_residual_eV=energy_error,
            physical_support_error=physics_error,
            known_nodes=data.matrix_known_nodes.sum(axis=0).tolist(),
            known_edges=data.matrix_known_edges.sum(axis=0).tolist(),
            total_node_rows=len(data.point_labels), total_edge_rows=len(data.edge_labels)))
        if (index + 1) % 100 == 0:
            print(f"audited {index + 1}/{len(manifest['samples'])}", file=sys.stderr, flush=True)
    if set(rows_by_id) - seen:
        raise ValueError("split references unknown source snapshots")
    result["block_split_counts"] = {key: dict(value) for key, value in block_splits.items()}
    result["shared_source_blocks_warning"] = [key for key, value in block_splits.items() if len(value) > 1]
    result["cross_split_exact_matrix_duplicates"] = sum(len(value) > 1 for value in content_splits.values())
    result["label_moments"] = label_moments
    result["excluded_source_ids"] = sorted(seen - set(rows_by_id))
    for path, digest in hashes.items():
        verified_file(root, path, digest)
    result["source_hashes_before_after_equal"] = True
    result["source_hashes"] = {str(path.relative_to(root)): digest for path, digest in hashes.items()}
    result["elapsed_seconds"] = time.monotonic() - started
    result["limitations"] = ["No occupation-resolved validation or McWeeny assumption",
        "SPD certified only on archived 20x20x1 mesh, no S clipping",
        "Shared MD blocks and one-frame gap do not prove statistical independence or general OOD",
        "No external OOD resource, historical recipe, GPU or scientific training inspected/executed"]
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
