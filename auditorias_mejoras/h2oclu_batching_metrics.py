"""Predeclared molecular H0+delta evaluation, conditional on reference overlap."""
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import save_npz


def owned_json(path, value):
    import run_cost_batching_comparison as recovery
    from run_readout_comparison import write_json
    path = recovery.own_output(path)
    recovery.own_output(Path(str(path)+".tmp"))
    write_json(path,value)


def matrix_metrics(predicted, reference, overlap, orbital_atoms, atomic_numbers):
    P, H, S = (np.asarray(x, dtype=np.float64) for x in (predicted, reference, overlap))
    owners, species = np.asarray(orbital_atoms), np.asarray(atomic_numbers)
    if (H.ndim != 2 or H.shape[0] != H.shape[1] or P.shape != H.shape or S.shape != H.shape
            or owners.shape != (len(H),) or not np.isfinite([P,H,S]).all()
            or not np.isin(species, [1,8]).all() or owners.min() < 0 or owners.max() >= len(species)):
        raise ValueError("incompatible/nonfinite molecular matrices or H/O basis")
    # Exact nonzero union, including S; never dilute with implicit zeros.
    ref_support, pred_support = H != 0, P != 0
    support = ref_support | pred_support | (S != 0)
    normS = float(np.sum(S[support]**2))
    if normS <= 0:
        raise ValueError("zero reference overlap")
    shift = float(np.sum(S[support] * (P-H)[support]) / normS)
    residual = P-H-shift*S
    absolute = np.abs(residual[support])
    defect = lambda x: float(np.linalg.norm(x-x.T) / max(np.linalg.norm(x), 1e-30))
    false_zero = int(np.sum(ref_support & ~pred_support))
    false_nonzero = int(np.sum(pred_support & ~ref_support))
    true_positive = int(np.sum(ref_support & pred_support))
    result = dict(gauge_c_eV=shift, support_size=int(support.sum()),
        support_policy="exact nonzero union(Href,H0+deltaHpred,Sref); threshold=0",
        H_MAE_gauge_meV=float(absolute.mean()*1000),
        H_RMSE_gauge_meV=float(np.sqrt(np.mean(absolute**2))*1000),
        H_relative_frobenius=float(np.linalg.norm(residual)/max(np.linalg.norm(H),1e-30)),
        H_p95_gauge_meV=float(np.quantile(absolute,.95)*1000),
        H_p99_gauge_meV=float(np.quantile(absolute,.99)*1000),
        H_MAE_raw_meV=float(np.abs((P-H)[support]).mean()*1000),
        H_pred_hermiticity=defect(P), H_ref_hermiticity=defect(H), S_ref_hermiticity=defect(S),
        support_F1=2*true_positive/max(2*true_positive+false_zero+false_nonzero,1),
        support_precision=true_positive/max(int(pred_support.sum()),1),
        support_recall=true_positive/max(int(ref_support.sum()),1),
        false_zero_count=false_zero, false_nonzero_count=false_nonzero,
        false_zero_rate=false_zero/max(int(ref_support.sum()),1),
        levels_rmse_aligned_meV=None, gap_error_meV=None, spectral_valid=False)
    onsite = owners[:,None] == owners[None,:]
    masks = {"onsite": onsite, "offsite": ~onsite}
    z = species[owners]
    for a,b in ((1,1),(1,8),(8,8)):
        masks[f"Z{a}-Z{b}"] = ((z[:,None] == a) & (z[None,:] == b)) | ((z[:,None] == b) & (z[None,:] == a))
    result["decomposition"] = {name: dict(count=int((mask & support).sum()),
        H_MAE_gauge_meV=float(np.abs(residual[mask & support]).mean()*1000) if (mask & support).any() else None)
        for name,mask in masks.items()}
    symmetric_S = (S+S.T)*.5
    eigS = np.linalg.eigvalsh(symmetric_S)
    result.update(S_min_eigenvalue=float(eigS[0]), S_condition=float(eigS[-1]/eigS[0]) if eigS[0]>0 else None,
                  S_directions_below_1e_2=int((eigS < 1e-2).sum()))
    if max(result[k] for k in ("H_pred_hermiticity","H_ref_hermiticity","S_ref_hermiticity")) > 1e-8 or eigS[0] <= 1e-10:
        result["spectral_failure"] = "raw Hermiticity >1e-8 or overlap minimum <=1e-10; no spectral claim"
        return result
    electrons = int(np.where(species == 8, 6, 1).sum())
    occupied = electrons//2
    if electrons % 2 or not 0 < occupied < len(H):
        raise ValueError("neutral unpolarized occupancy has no HOMO/LUMO")
    # Projection is explicit and only follows the raw defect gate; no S cutoff.
    ref_levels = eigh((H+H.T)*.5, symmetric_S, eigvals_only=True)
    pred_levels = eigh((P+P.T)*.5, symmetric_S, eigvals_only=True)-shift
    window = (ref_levels >= ref_levels[occupied-1]-2) & (ref_levels <= ref_levels[occupied]+2)
    errors = pred_levels-ref_levels
    ref_gap, pred_gap = (float(levels[occupied]-levels[occupied-1]) for levels in (ref_levels,pred_levels))
    result.update(spectral_valid=True, spectral_scope="Gamma, float64 generalized eigh, symmetric projection after gate; reference S conditional",
        occupied_levels=occupied, window_level_count=int(window.sum()),
        reference_levels_eV=ref_levels.tolist(), predicted_aligned_levels_eV=pred_levels.tolist(),
        reference_gap_eV=ref_gap, predicted_gap_eV=pred_gap, gap_error_meV=abs(pred_gap-ref_gap)*1000,
        levels_rmse_aligned_meV=float(np.sqrt(np.mean(errors[window]**2))*1000),
        occupied_levels_rmse_meV=float(np.sqrt(np.mean(errors[:occupied]**2))*1000))
    return result


def evaluate_checkpoint(checkpoint, rows, output):
    """Caller must validate checkpoint provenance before trusted pickle loading."""
    import run_cost_batching_comparison as recovery
    recovery.recovery_api()
    import sisl
    import torch
    from graph2mat import MatrixDataProcessor
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    from run_readout_comparison import sha256

    output = recovery.own_output(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)["datamodule_hyper_parameters"]
    model = LitMACEMatrixModel.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False).eval()
    if (model.hparams.n_output_components != 1 or metadata.get("sub_point_matrix") is not False
            or metadata.get("out_matrix") != "hamiltonian"):
        raise ValueError("evaluation requires H-only delta output, no isolated-point subtraction")
    processor = MatrixDataProcessor(basis_table=model.basis_table, symmetric_matrix=True, sub_point_matrix=False,
        out_matrix="hamiltonian", n_matrix_components=1, matrix_component_policy="h_only",
        node_attr_getters=model.initial_node_feats)
    results = []
    for index,row in enumerate(rows):
        geometry = sisl.get_sile(row["structure_path"]).read_geometry()
        ref = sisl.get_sile(row["reference_tshs_path"]).read_hamiltonian()
        atomic = sisl.get_sile(row["h0"]).read_hamiltonian()
        delta = processor.torch_predict(model, geometry)
        if (delta.shape[:2] != ref.shape[:2] or list(delta.nsc) != [1,1,1]
                or not np.array_equal(delta.geometry.orbitals, ref.geometry.orbitals)
                or not np.allclose(delta.geometry.xyz, ref.geometry.xyz, rtol=0, atol=1e-6)):
            raise ValueError("prediction/reference representation mismatch: "+row["sample_id"])
        # Never take S from the H-only prediction or regenerate H0.
        prediction = delta.tocsr(0).astype(np.float64) + atomic.tocsr(0).astype(np.float64)
        owners = geometry.o2a(np.arange(geometry.no))
        metrics = matrix_metrics(prediction.toarray(), ref.tocsr(0).toarray(), ref.tocsr(ref.S_idx).toarray(),
                                 owners, geometry.atoms.Z)
        path = recovery.own_output(output / f"physical-H-{index}.npz")
        save_npz(path, prediction)
        results.append(dict(sample_id=row["sample_id"], split=row["split"], matrix=str(path), matrix_sha256=sha256(path), **metrics))
        owned_json(output / "per_structure.json", results)
        print("EVAL", row["sample_id"], "H", metrics["H_MAE_gauge_meV"], flush=True)
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), rows=results,
                  seconds=time.monotonic()-started, device="cpu", precision="FP32 inference / float64 physical matrices and eigh")
    owned_json(output / "evaluation.json", result)
    return result
