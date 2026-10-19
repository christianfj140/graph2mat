"""CPU-only attribution of existing h2oclu matrices; no model loading or fitting.

All spectra retain the full reference-S basis. Oracle interventions remove parts
of the gauge-aligned residual, not directions of S; they require reference labels.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import load_npz

from run_readout_comparison import REPO, sha256, verify_files

ROOT = REPO / "auditorias_mejoras/experimentos_batching_h2oclu_20261016_science_v1"
LOW = 1e-2  # Existing diagnostic partition, never a basis truncation.


def symmetric(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all()):
        raise ValueError("nonfinite or nonsquare matrix")
    defect = np.linalg.norm(matrix-matrix.T) / max(np.linalg.norm(matrix), 1e-30)
    if defect > 1e-8:
        raise ValueError("raw Hermiticity defect exceeds 1e-8")
    return (matrix+matrix.T)/2


def spectral_metrics(A, frame):
    levels = np.linalg.eigvalsh(symmetric(A))
    n, ref, window = frame["nocc"], frame["levels"], frame["window"]
    gap = float(levels[n]-levels[n-1])
    return dict(gap_eV=gap, gap_error_eV=abs(gap-frame["gap"]),
                levels_rmse_eV=float(np.sqrt(np.mean((levels[window]-ref[window])**2))),
                level_count=len(levels))


def reference_frame(H, S, H0, nocc):
    h, s, h0 = (symmetric(m) for m in (H, S, H0))
    if h.shape != s.shape or h0.shape != h.shape or not 0 < nocc < len(h):
        raise ValueError("incompatible reference/occupation")
    values, U = np.linalg.eigh(s)
    if values[0] <= 1e-10 or np.min(np.abs(values-LOW)) < 1e-10:
        raise ValueError("nonpositive overlap or ambiguous low-S partition boundary")
    D = U.T / np.sqrt(values)[:, None]
    np.testing.assert_allclose(D@s@D.T, np.eye(len(s)), atol=1e-10, rtol=0)
    A = symmetric(D@h@D.T)
    levels, vectors = np.linalg.eigh(A)
    window = (levels >= levels[nocc-1]-2) & (levels <= levels[nocc]+2)
    frame = dict(H=np.asarray(H), S=np.asarray(S), H0=np.asarray(H0), D=D, A=A,
                 overlap_values=values, low=values < LOW, levels=levels, vectors=vectors,
                 window=window, nocc=nocc, gap=float(levels[nocc]-levels[nocc-1]))
    c0 = float(np.sum((H0-H)*S)/np.sum(S*S))
    h0_residual = H0-H-c0*S
    h0_metrics = spectral_metrics(D@h0@D.T-c0*np.eye(len(h)), frame)
    support = (H != 0) | (H0 != 0) | (S != 0)
    h0_metrics.update(gauge_c_eV=c0, H_MAE_gauge_meV=float(np.abs(h0_residual[support]).mean()*1000),
                      residual_frobenius_eV=float(np.linalg.norm(h0_residual)),
                      residual_whitened_frobenius_eV=float(np.linalg.norm(D@h0_residual@D.T)))
    frame["h0_metrics"] = h0_metrics
    return frame


def subspace_low_weight(vectors, low):
    """Trace(P_low P_states)/rank(P_states), invariant under internal rotations."""
    return float(np.sum(vectors[low]**2)/vectors.shape[1]) if vectors.shape[1] else None


def complete_degenerate_mask(levels, mask):
    """Do not split a numerically degenerate cluster at a state-selection boundary."""
    clusters = np.cumsum(np.r_[True, np.diff(levels) > 1e-8])
    return np.isin(clusters, np.unique(clusters[mask]))


def diagnose(P, frame, owners, species, families):
    p = symmetric(P)
    H, S, H0, D = (frame[k] for k in ("H", "S", "H0", "D"))
    owners, species, families = np.asarray(owners), np.asarray(species), np.asarray(families)
    if (p.shape != H.shape or owners.shape != (len(H),) or families.shape != (len(H),)
            or owners.min() < 0 or owners.max() >= len(species) or not np.isin(species, [1, 8]).all()):
        raise ValueError("prediction/basis mapping mismatch")
    c = float(np.sum((P-H)*S)/np.sum(S*S))
    raw_R = P-H-c*S
    # Raw H/S gates already passed; cancellation changes the relative defect denominator.
    R = (raw_R+raw_R.T)/2
    T = symmetric(D@R@D.T)
    A = frame["A"]+T
    levels, vectors = np.linalg.eigh(A)
    w2, r2 = float(np.sum(T*T)), float(np.sum(R*R))
    # <T,D R_g D^T> = <D^T T D,R_g>; individual signed shares can cancel.
    score = R*(D.T@T@D)
    baseline = spectral_metrics(A, frame)

    def group(mask, oracle=True):
        out = dict(raw_squared_share=float(np.sum((R*R)[mask])/r2) if r2 else 0.,
                   whitened_signed_share=float(score[mask].sum()/w2) if w2 else 0.)
        if oracle:
            contribution = D@np.where(mask, R, 0.)@D.T
            out["oracle"] = spectral_metrics(A-contribution, frame)
        return out

    onsite = owners[:, None] == owners[None, :]
    site = {"onsite": group(onsite), "offsite": group(~onsite)}
    z = species[owners]
    pairs = {}
    for a, b, label in ((1, 1, "H-H"), (1, 8, "H-O"), (8, 8, "O-O")):
        mask = ((z[:, None] == a) & (z[None, :] == b)) | ((z[:, None] == b) & (z[None, :] == a))
        pairs[label] = group(mask)
    orbital_groups = {}
    names = sorted(set(families))
    for i, a in enumerate(names):
        for b in names[i:]:
            mask = ((families[:, None] == a) & (families[None, :] == b)) | ((families[:, None] == b) & (families[None, :] == a))
            orbital_groups[a+" / "+b] = group(mask)
    for groups in (site, pairs, orbital_groups):
        if w2:
            np.testing.assert_allclose(sum(g["whitened_signed_share"] for g in groups.values()), 1., atol=1e-9)
        if r2:
            np.testing.assert_allclose(sum(g["raw_squared_share"] for g in groups.values()), 1., atol=1e-12)

    low = frame["low"]
    ll, hh = low[:, None] & low[None, :], ~low[:, None] & ~low[None, :]
    sectors = {}
    for name, mask in (("low_low", ll), ("cross", ~(ll | hh)), ("high_high", hh), ("low_touch", ~hh)):
        sectors[name] = dict(whitened_squared_share=float(np.sum((T*T)[mask])/w2) if w2 else 0.,
                             oracle=spectral_metrics(A-np.where(mask, T, 0.), frame))
    if w2:
        np.testing.assert_allclose(sum(sectors[k]["whitened_squared_share"] for k in ("low_low", "cross", "high_high")), 1., atol=1e-12)

    natoms = len(species)
    block_index = (owners[:, None]*natoms+owners[None, :]).ravel()
    block_score = np.bincount(block_index, weights=score.ravel(), minlength=natoms**2).reshape(natoms, natoms)
    block_r2 = np.bincount(block_index, weights=(R*R).ravel(), minlength=natoms**2).reshape(natoms, natoms)
    blocks = []
    for i in range(natoms):
        for j in range(i, natoms):
            factor = 1 if i == j else 2
            blocks.append(dict(atoms=[i, j], species=[int(species[i]), int(species[j])],
                               whitened_signed_share=float(factor*block_score[i, j]/w2) if w2 else 0.,
                               raw_squared_share=float(factor*block_r2[i, j]/r2) if r2 else 0.))
    signed_sum = sum(b["whitened_signed_share"] for b in blocks)
    if w2:
        np.testing.assert_allclose(signed_sum, 1., atol=1e-9)
    blocks.sort(key=lambda b: abs(b["whitened_signed_share"]), reverse=True)
    top_union = np.zeros_like(R, dtype=bool)
    for block in blocks[:3]:
        i, j = block["atoms"]
        mask = ((owners[:, None] == i) & (owners[None, :] == j)) | ((owners[:, None] == j) & (owners[None, :] == i))
        block["oracle"] = group(mask)["oracle"]
        top_union |= mask

    ref = frame["levels"]
    in_gap = complete_degenerate_mask(levels, (levels > ref[frame["nocc"]-1]+.05) & (levels < ref[frame["nocc"]]-.05))
    near = complete_degenerate_mask(levels, (levels >= ref[frame["nocc"]-1]-2) & (levels <= ref[frame["nocc"]]+2))
    frontier = np.zeros(len(levels), dtype=bool)
    frontier[frame["nocc"]-1:frame["nocc"]+1] = True
    frontier = complete_degenerate_mask(levels, frontier)
    ref_projection = frame["vectors"][:, complete_degenerate_mask(ref, frame["window"])]
    reference_window_residual = ref_projection.T@T@ref_projection
    nocc = frame["nocc"]
    # A collapsed gap need not put a near-null vector at the HOMO/LUMO: extra
    # low-S states can enter below them and change which levels are occupied.
    leakage = (float(np.sum((frame["vectors"][:, nocc:].T@vectors[:, :nocc])**2))
               if baseline["gap_eV"] > 1e-8 else None)
    cancellation = (P-H0)-(H-H0)-(P-H)
    np.testing.assert_allclose(cancellation, 0., atol=1e-12, rtol=0)
    # Compare both independent reconstructions; no new model inference is involved.
    np.testing.assert_allclose((P-H0)+H0, P, atol=1e-12, rtol=0)
    support = (P != 0) | (H != 0) | (S != 0)
    return dict(gauge_c_eV=c, baseline=baseline, aligned_levels_eV=levels.tolist(),
                H_MAE_gauge_meV=float(np.abs(raw_R[support]).mean()*1000),
                raw_frobenius_eV=float(np.sqrt(r2)), whitened_frobenius_eV=float(np.sqrt(w2)),
                raw_operator_norm_eV=float(np.max(np.abs(np.linalg.eigvalsh(R)))),
                whitened_operator_norm_eV=float(np.max(np.abs(np.linalg.eigvalsh(T)))),
                h0_cancellation_max_eV=float(np.max(np.abs(cancellation))),
                delta_reference_frobenius_eV=float(np.linalg.norm(H-H0)),
                delta_prediction_raw_frobenius_eV=float(np.linalg.norm(P-H0)),
                site=site, species=pairs, families=orbital_groups, overlap_sectors=sectors,
                atomic_blocks=blocks[:10], atomic_block_signed_share_sum=signed_sum,
                top3_atomic_blocks_joint_oracle=group(top_union)["oracle"],
                atom_signed_shares=(block_score.sum(axis=1)/w2).tolist() if w2 else [0.]*natoms,
                predicted_in_gap_count=int(in_gap.sum()), predicted_near_gap_count=int(near.sum()),
                predicted_in_gap_low_weight=subspace_low_weight(vectors[:, in_gap], low),
                predicted_near_gap_low_weight=subspace_low_weight(vectors[:, near], low),
                predicted_frontier_low_weight=subspace_low_weight(vectors[:, frontier], low),
                predicted_frontier_state_count=int(frontier.sum()),
                occupied_subspace_leakage_states=leakage,
                predicted_occupied_low_trace=float(np.sum(vectors[low, :nocc]**2)) if leakage is not None else None,
                reference_occupied_low_trace=float(np.sum(frame["vectors"][low, :nocc]**2)),
                predicted_levels_below_reference_homo=int((levels < ref[frame["nocc"]-1]-.05).sum()),
                reference_near_gap_low_weight=subspace_low_weight(ref_projection, low),
                reference_window_residual_frobenius_eV=float(np.linalg.norm(reference_window_residual)))


def read_references(manifest):
    import sisl

    references, metadata = {}, {}
    for row in manifest["rows"]:
        if row["split"] not in ("validation", "test"):
            continue
        for path in (row["reference_tshs_path"], row["structure_path"], row["h0"]):
            if path not in manifest["files"]:
                raise ValueError("unfrozen reference input: "+path)
        fdf = sisl.get_sile(row["structure_path"])
        geometry = fdf.read_geometry()
        reference, atomic = (sisl.get_sile(row[k]).read_hamiltonian() for k in ("reference_tshs_path", "h0"))
        if float(fdf.get("NetCharge", default=0.)) != 0. or row["net_charge"] != 0.:
            raise ValueError("neutral occupation requires zero NetCharge")
        for matrix in (reference, atomic):
            if not matrix.spin.is_unpolarized or list(matrix.nsc) != [1, 1, 1]:
                raise ValueError("only unpolarized isolated Gamma matrices supported")
            np.testing.assert_array_equal(matrix.geometry.orbitals, geometry.orbitals)
            np.testing.assert_allclose(matrix.geometry.xyz, geometry.xyz, atol=1e-6, rtol=0)
            np.testing.assert_allclose(matrix.geometry.cell, geometry.cell, atol=1e-6, rtol=0)
        H, S = reference.tocsr(0).toarray(), reference.tocsr(reference.S_idx).toarray()
        H0, S0 = atomic.tocsr(0).toarray(), atomic.tocsr(atomic.S_idx).toarray()
        np.testing.assert_allclose(S0, S, atol=1e-12, rtol=0)
        delta_error = None
        if row["delta"] is not None:
            if row["delta"] not in manifest["files"]:
                raise ValueError("unfrozen delta label")
            delta = sisl.get_sile(row["delta"]).read_hamiltonian()
            delta_error = float(np.max(np.abs(delta.tocsr(0).toarray()-(H-H0))))
            np.testing.assert_allclose(delta.tocsr(0).toarray(), H-H0, atol=1e-8, rtol=0)
        species = np.asarray(geometry.atoms.Z)
        if not np.isin(species, [1, 8]).all():
            raise ValueError("FDF species are not H/O")
        electrons = int(np.where(species == 8, 6, 1).sum())
        if electrons % 2:
            raise ValueError("odd neutral electron count")
        families, orbital_order = [], []
        for atom in geometry.atoms:
            order = []
            for orbital in atom.orbitals:
                n, l, m, zeta = (int(getattr(orbital, k)) for k in ("n", "l", "m", "zeta"))
                name = f"{atom.symbol}:{n}{'spdf'[l]}:z{zeta}"
                families.append(name)
                order.append([n, l, m, zeta])
            orbital_order.append(order)
        frame = reference_frame(H, S, H0, electrons//2)
        owners = geometry.o2a(np.arange(geometry.no))
        references[row["sample_id"]] = (frame, owners, species, families)
        eigenvalues = frame["overlap_values"]
        low_orbital_diagonal = np.sum((frame["D"][frame["low"]].T*np.sqrt(eigenvalues[frame["low"]]))**2, axis=1)
        metadata[row["sample_id"]] = dict(split=row["split"], atoms=len(species), orbitals=len(H),
            species_Z=species.tolist(), xyz_angstrom=geometry.xyz.tolist(), orbital_n_l_m_zeta_by_atom=orbital_order,
            electrons=electrons, occupied_levels=electrons//2, reference_gap_eV=frame["gap"],
            reference_homo_eV=float(frame["levels"][electrons//2-1]), reference_lumo_eV=float(frame["levels"][electrons//2]),
            overlap_min=float(eigenvalues[0]), overlap_condition=float(eigenvalues[-1]/eigenvalues[0]),
            low_overlap_count=int(frame["low"].sum()), low_boundary_margin=float(np.min(abs(eigenvalues-LOW))),
            low_overlap_family_weights={name: float(low_orbital_diagonal[np.asarray(families) == name].sum()/max(int(frame["low"].sum()), 1))
                                        for name in sorted(set(families))},
            h0=frame["h0_metrics"], delta_label_error_eV=delta_error)
    if len(references) != 7:
        raise ValueError("expected four ID and three OOD references")
    return references, metadata


def run():
    started = time.monotonic()
    manifest_path = ROOT/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    script_path, script_sha = Path(__file__).resolve(), sha256(Path(__file__))
    verify_files(manifest["files"])
    for name, version in manifest["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("frozen software version changed: "+name)
    if {(j["arm"], j["seed"]) for j in manifest["jobs"]} != {(a, s) for a in ("fixed", "cost") for s in range(3)}:
        raise ValueError("six paired jobs required")
    references, metadata = read_references(manifest)
    records, inputs = [], {str(manifest_path): manifest_sha, str(script_path): script_sha}
    for job in manifest["jobs"]:
        directory = Path(job["root"])
        if directory != ROOT/job["key"]:
            raise ValueError("unexpected job root")
        path = directory/"result.json"
        proof_path = directory/"evaluation_receipt.json"
        proof = json.loads(proof_path.read_text())
        if sha256(path) != proof["result_sha256"] or sha256(proof["checkpoint"]) != proof["checkpoint_sha256"]:
            raise ValueError("result/checkpoint provenance changed")
        result = json.loads(path.read_text())
        if (result["arm"] != job["arm"] or result["seed"] != job["seed"]
                or result["evaluation"]["checkpoint"] != proof["checkpoint"]
                or result["evaluation"]["checkpoint_sha256"] != proof["checkpoint_sha256"]):
            raise ValueError("result identity mismatch")
        inputs[str(path)], inputs[str(proof_path)] = sha256(path), sha256(proof_path)
        inputs[proof["checkpoint"]] = proof["checkpoint_sha256"]
        rows = result["evaluation"]["rows"]
        if len(rows) != 7 or {r["sample_id"] for r in rows} != set(references):
            raise ValueError("incomplete evaluation coverage")
        for row in rows:
            path = Path(row["matrix"])
            if not path.resolve().is_relative_to(directory/"evaluation") or sha256(path) != row["matrix_sha256"]:
                raise ValueError("saved physical matrix provenance changed")
            inputs[str(path)] = row["matrix_sha256"]
            frame, owners, species, families = references[row["sample_id"]]
            if (row["split"] != metadata[row["sample_id"]]["split"] or not row["spectral_valid"]
                    or row["occupied_levels"] != frame["nocc"]):
                raise ValueError("saved population/occupation/spectral gate changed")
            P = load_npz(path).toarray().astype(np.float64)
            out = diagnose(P, frame, owners, species, families)
            np.testing.assert_allclose(out["aligned_levels_eV"], row["predicted_aligned_levels_eV"], atol=1e-8, rtol=0)
            np.testing.assert_allclose(frame["levels"], row["reference_levels_eV"], atol=1e-8, rtol=0)
            np.testing.assert_allclose(out["H_MAE_gauge_meV"], row["H_MAE_gauge_meV"], atol=1e-8, rtol=0)
            np.testing.assert_allclose([out["baseline"]["gap_error_eV"], out["baseline"]["levels_rmse_eV"]],
                [row["gap_error_meV"]/1000, row["levels_rmse_aligned_meV"]/1000], atol=1e-8, rtol=0)
            out.update(job=job["key"], arm=job["arm"], seed=job["seed"], sample_id=row["sample_id"], split=row["split"])
            records.append(out)
            print(f"{job['key']} {row['sample_id']}: gap={out['baseline']['gap_eV']:.6f}eV", file=sys.stderr, flush=True)
    verify_files({**manifest["files"], **inputs})
    return dict(scope="CPU saved-matrix attribution, reference S, full basis; diagnostic-only, oracle controls not deployable",
                overlap_partition=LOW, signed_share_definition="<T,T_group>/||T||F^2; additive, signed, not spectral causality",
                counterfactual_gauge="original common c retained for every intervention; no refitting",
                source_sha256=sha256(Path(__file__)), frozen_hashes=len(manifest["files"]), inputs=inputs,
                reference_metadata=metadata, records=records, seconds=time.monotonic()-started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    if (output.exists() or output.is_symlink() or not output.resolve().is_relative_to(REPO/"auditorias_mejoras")
            or output.resolve().is_relative_to(ROOT)):
        raise ValueError("new diagnostic output inside auditorias_mejoras, outside original campaign, required")
    result = run()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Saved {len(result['records'])} records to {output}", file=sys.stderr)
