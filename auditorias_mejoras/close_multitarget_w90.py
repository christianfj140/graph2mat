"""CPU-only, label-free inference for the six frozen w90 checkpoints.

CUDA_VISIBLE_DEVICES= python auditorias_mejoras/close_multitarget_w90.py --out \
    auditorias_mejoras/cierre_multitarget_w90_20261019_v1
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

try:
    from auditorias_mejoras import run_multitarget_w90_six as run
except ModuleNotFoundError:
    import run_multitarget_w90_six as run

pilot = run.pilot
CAMPAIGN = pilot.REPO / "auditorias_mejoras/campana_multitarget_w90_20261019_v1"


def packed_errors(refn, refe, predn, prede):
    refs = np.concatenate((refn, refe)).astype(float)
    pn = np.asarray(predn).reshape(len(refn), -1)
    pe = np.asarray(prede).reshape(len(refe), -1)
    predicted = np.concatenate((pn, pe)).astype(float)
    if not np.isfinite(refs).all() or not np.isfinite(predicted).all():
        raise ValueError("nonfinite packed reference/prediction")
    dh, s = predicted[:, 0] - refs[:, 0], refs[:, 1]
    den = float(s @ s)
    if den <= 0:
        raise ValueError("zero overlap norm")
    shift = float(dh @ s / den)
    residual = predicted - refs[:, :predicted.shape[1]]
    residual[:, 0] -= shift * s
    result = dict(shift_eV=shift, count=len(refs))
    for index, target in enumerate(pilot.TARGETS[:predicted.shape[1]]):
        values = residual[:, index]
        result[target] = dict(mse=float(np.mean(values**2)), rmse=float(np.sqrt(np.mean(values**2))),
                              mae=float(np.mean(np.abs(values))), max_abs=float(np.max(np.abs(values))))
    return result


def spectral_errors(h, s, reference_levels, shift):
    # Same single structure gauge, no k-dependent alignment and no S cutoff.
    result = dict(status="invalid", levels_rmse_eV=None, direct_gap_mae_eV=None,
                  indirect_gap_error_eV=None, min_s=None, non_spd_k=[])
    if not np.isfinite(h).all() or not np.isfinite(s).all():
        raise ValueError("nonfinite Bloch matrix")
    eigenvalues = np.linalg.eigvalsh(s)
    result.update(min_s=float(eigenvalues.min()),
                  non_spd_k=np.flatnonzero(eigenvalues.min(axis=1) <= 0).tolist())
    if result["non_spd_k"]:
        result["reason"] = "overlap not positive definite; no repair/cutoff"
        return result
    try:
        spectrum = pilot.spectrum(h - shift * s, s)
    except (ValueError, np.linalg.LinAlgError) as error:
        result["reason"] = f"{type(error).__name__}: {error}"
        return result
    levels = np.array(spectrum["levels"])
    reference_levels = np.asarray(reference_levels)
    if levels.shape != reference_levels.shape or levels.shape[1] != 8:
        raise ValueError("frozen eight-orbital reference shape changed")
    gap, refgap = pilot.gaps(levels), pilot.gaps(reference_levels)
    result.update(status="valid", levels_rmse_eV=float(np.sqrt(np.mean((levels-reference_levels)**2))),
        direct_gap_mae_eV=float(np.mean(np.abs(np.array(gap["direct"])-refgap["direct"]))),
        indirect_gap_error_eV=abs(gap["indirect"]-refgap["indirect"]),
        predicted_indirect_gap_eV=gap["indirect"], reference_indirect_gap_eV=refgap["indirect"],
        cholesky_max_disagreement_eV=spectrum["chol_error_eV"])
    return result


def spectral_summary(rows, grid, route):
    values = [(row["sample_id"], row["spectra"][grid][route]) for row in rows]
    valid = [value for _, value in values if value["status"] == "valid"]
    keys = ("levels_rmse_eV", "direct_gap_mae_eV", "indirect_gap_error_eV")
    return dict(total=len(values), valid=len(valid),
        invalid_ids=[sid for sid, value in values if value["status"] != "valid"],
        mean_valid={key: float(np.mean([v[key] for v in valid]))
                    for key in keys if valid and key in valid[0]})


def predict_one(model, processor, bundle):
    import torch
    from torch_geometric.data import Batch
    from graph2mat.bindings.torch import TorchBasisMatrixDataset
    from graph2mat.tools.lightning.data import TypeLayoutCollate

    data = TorchBasisMatrixDataset([bundle["overlap"].geometry], processor, load_labels=False)[0]
    if getattr(data, "point_labels", None) is not None or getattr(data, "edge_labels", None) is not None:
        raise ValueError("inference received reference labels")
    batch = TypeLayoutCollate(Batch.from_data_list, processor.basis_table)([data])
    with torch.no_grad():
        prediction = model(batch)
    prediction = {key: prediction[key].detach().cpu().clone() for key in ("node_labels", "edge_labels")}
    if not all(bool(torch.isfinite(value).all()) for value in prediction.values()):
        raise ValueError("nonfinite inference")
    return data, prediction


def auxiliary_physics(matrices, reference):
    h, s, d, e = (matrices[k] for k in pilot.KEYS)
    sr = reference["overlap"]

    def rows(a, b):
        # Finite writer outputs have zero tails, not unknown references. CSR
        # column numbers alone do not identify the same image in different nsc.
        images = sorted(set(map(tuple, a.geometry.sc_off)) | set(map(tuple, b.geometry.sc_off)))
        av, bv = pilot.image_arrays(a, images), pilot.image_arrays(b, images)
        if not np.isfinite(av).all() or not np.isfinite(bv).all():
            raise ValueError("nonfinite real-space physical contraction")
        return np.einsum("rij,rij->i", av, bv)

    def population(overlap):
        return float(rows(d, overlap).sum())
    nref = float(rows(reference["density"], sr).sum())
    energy = rows(d, h) - rows(e, s)
    return dict(N_reference=nref, N_pred_fixed_Sref=population(sr), N_pred_own_Spred=population(s),
        HD_minus_SE_max_abs_per_orbital_eV=float(np.abs(energy).max()))


def evaluate(out):
    out.mkdir()  # Immutable new closure, never overwrite fits or old evaluations.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("closure requires CUDA_VISIBLE_DEVICES empty")
    import torch
    from graph2mat.bindings.torch import TorchBasisMatrixDataset
    from graph2mat.bindings.torch.load import load_from_lit_ckpt
    from graph2mat.core.data.physical import _read_reference_bundle

    started = time.monotonic()
    campaign = run.verify_campaign(CAMPAIGN)
    source_manifest = json.loads((CAMPAIGN / "manifest.json").read_text())
    final = json.loads((CAMPAIGN / "summary.json").read_text())
    if final["status"] != "complete" or set(final["jobs"].values()) != {"complete"}:
        raise ValueError("six fits incomplete")
    artifacts = {"campaign.json": pilot.sha(CAMPAIGN / "campaign.json"),
                 "summary.json": pilot.sha(CAMPAIGN / "summary.json")}
    for job in campaign["jobs"]:
        job_id = job["id"]
        result = json.loads((CAMPAIGN / job_id / "result.json").read_text())
        best = Path(result["best_checkpoint"])
        if not best.is_relative_to(CAMPAIGN / job_id) or pilot.sha(best) != result["best_sha256"]:
            raise ValueError(f"checkpoint binding mismatch: {job_id}")
        if result["best_h_gauge_mse"] != min(v["h_gauge_mse"] for v in result["validation"]):
            raise ValueError("checkpoint not selected by H validation")
        for name in (f"{job_id}/result.json", str(best.relative_to(CAMPAIGN))):
            artifacts[name] = pilot.sha(CAMPAIGN / name)
    code = {str(p.relative_to(pilot.REPO)): pilot.sha(p) for p in
            (Path(__file__), pilot.REPO / "auditorias_mejoras/test_close_multitarget_w90.py")}
    pilot.write_json(out / "provenance.json", dict(campaign=str(CAMPAIGN), artifacts=artifacts, code=code,
        source_manifest_sha256=pilot.sha(CAMPAIGN / "manifest.json"),
        kpoints_sha256=pilot.sha(run.V2 / "kpoints.json"),
        command="CUDA_VISIBLE_DEVICES= python auditorias_mejoras/close_multitarget_w90.py --out " + str(out),
        gauge="packed node+unique-edge weight1, one c per structure; H-cSref common, H-cSpred separate",
        nominal_filling="8 electrons, 4 doubly occupied bands; signed indirect gap, not molecular HOMO-LUMO",
        limitations=source_manifest["warnings"]))
    grids_json = json.loads((run.V2 / "kpoints.json").read_text())
    grids = {name: np.array(grids_json[name]) for name in ("mesh", "path")}
    table = pilot.certified_table(source_manifest)
    reference_processor = pilot.arm_processor(table, "joint")
    cases = []
    for split in ("validation", "test"):
        for sid in source_manifest["split_ids"][split]:
            fdf = pilot.check_fdf(source_manifest["paths"][sid])
            pilot.checked_candidates(fdf.parent, source_manifest)
            bundle, _ = _read_reference_bundle(fdf, need_density=True, need_hamiltonian=True,
                need_energy_density=True, require_mesh=True, processor=reference_processor)
            if not np.array_equal(bundle["kpoints"], grids["mesh"]):
                raise ValueError(f"reference mesh changed: {sid}")
            data = TorchBasisMatrixDataset([bundle], reference_processor)[0]
            if not bool(data.matrix_known_nodes.all() and data.matrix_known_edges.all()):
                raise ValueError("incomplete reference labels")
            refs = {}
            for name, k in grids.items():
                s = pilot.bloch(bundle["overlap"], k)
                h = pilot.bloch(bundle["hamiltonian"], k)
                refs[name] = dict(s=s, levels=pilot.spectrum(h, s)["levels"])
            cases.append((sid, split, bundle, data, refs))
        print(f"references {split} complete", flush=True)
    summary = dict(status="complete", jobs={})
    for job in campaign["jobs"]:
        job_id, arm = job["id"], job["arm"]
        destination = out / job_id
        destination.mkdir()
        result = json.loads((CAMPAIGN / job_id / "result.json").read_text())
        model, processor = load_from_lit_ckpt(result["best_checkpoint"], cpu=True, as_torch=True)
        model.float().eval()
        rows = []
        with (destination / "rows.jsonl").open("x") as stream:
            for index, (sid, split, bundle, gold, refs) in enumerate(cases):
                data, prediction = predict_one(model, processor, bundle)
                for field in ("positions", "edge_index", "point_types", "edge_types", "shifts"):
                    torch.testing.assert_close(getattr(data, field), getattr(gold, field), rtol=0, atol=0)
                rn, re = (gold.ensure_numpy(getattr(gold, name)) for name in ("point_labels", "edge_labels"))
                packed = packed_errors(rn, re, prediction["node_labels"], prediction["edge_labels"])
                with (destination / f"{sid}.npz").open("xb") as archive:
                    np.savez_compressed(archive, **prediction)
                targets = pilot.TARGETS if arm == "joint" else pilot.TARGETS[:1]
                matrices = {key: processor.matrix_from_data(data, predictions=prediction,
                    matrix_target=target, threshold=None) for target, key in zip(targets, pilot.KEYS)}
                spectra = {}
                for name, k in grids.items():
                    hp = pilot.bloch(matrices["hamiltonian"], k)
                    spectra[name] = dict(common=spectral_errors(hp, refs[name]["s"], refs[name]["levels"], packed["shift_eV"]))
                    if arm == "joint":
                        spectra[name]["predicted_s"] = spectral_errors(hp, pilot.bloch(matrices["overlap"], k),
                            refs[name]["levels"], packed["shift_eV"])
                row = dict(sample_id=sid, split=split, packed=packed, spectra=spectra)
                if arm == "joint":
                    row["physics"] = auxiliary_physics(matrices, bundle)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                if (index + 1) % 50 == 0:
                    print(f"{job_id} {index+1}/{len(cases)}", flush=True)
        split_summary = {}
        for split in ("validation", "test"):
            selected = [row for row in rows if row["split"] == split]
            counts = sum(row["packed"]["count"] for row in selected)
            packed_summary = {target: dict(mse=sum(row["packed"][target]["mse"] * row["packed"]["count"]
                for row in selected) / counts,
                mean_structure_rmse=float(np.mean([row["packed"][target]["rmse"] for row in selected])))
                for target in targets}
            split_summary[split] = dict(samples=len(selected), packed=packed_summary,
                spectra={name: {route: spectral_summary(selected, name, route)
                    for route in (("common", "predicted_s") if arm == "joint" else ("common",))} for name in grids})
            if arm == "joint":
                split_summary[split]["physics"] = {
                    "mean_abs_N_error_fixed_Sref": float(np.mean([abs(row["physics"]["N_pred_fixed_Sref"]-row["physics"]["N_reference"]) for row in selected])),
                    "mean_abs_N_error_own_Spred": float(np.mean([abs(row["physics"]["N_pred_own_Spred"]-row["physics"]["N_reference"]) for row in selected])),
                    "mean_HD_minus_SE_max_abs_per_orbital_eV": float(np.mean([row["physics"]["HD_minus_SE_max_abs_per_orbital_eV"] for row in selected]))}
        reproduced = split_summary["validation"]["packed"]["hamiltonian"]["mse"]
        if not np.isclose(reproduced, result["best_h_gauge_mse"], rtol=.002, atol=1e-10):
            raise ValueError(f"validation H not reproduced: {job_id}: {reproduced} vs {result['best_h_gauge_mse']}")
        summary["jobs"][job_id] = dict(job=job, selected_checkpoint=result["best_checkpoint"],
            original_validation_H_mse=result["best_h_gauge_mse"], splits=split_summary,
            rows_sha256=pilot.sha(destination / "rows.jsonl"))
        del model, processor
    run.verify_campaign(CAMPAIGN)
    run.verify_artifacts(CAMPAIGN, artifacts)
    for name, digest in code.items():
        if pilot.sha(pilot.REPO / name) != digest:
            raise ValueError("closure source changed")
    summary.update(seconds=time.monotonic()-started, integrity_verified=True)
    pilot.write_json(out / "summary.json", summary)
    print("CPU closure complete", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != pilot.REPO / "auditorias_mejoras":
        parser.error("output must be a new direct child of graph2mat/auditorias_mejoras")
    import torch
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    evaluate(out)


if __name__ == "__main__":
    main()
