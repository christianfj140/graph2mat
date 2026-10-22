"""How density-matrix errors reach H = H_KS[DM] (roadmap 7.7 pilot; CPU, no training).

The reference DM of a converged calculation gets a controlled error and SIESTA builds H from
it in one step (the H0 keys of Delta-learning, but reading the DM). The gauge-aligned error of
that H against the converged H is compared with the DM error, both relative to their own
Delta (DM - DM_atomic, H - H0): if a model that learns Delta-DM to some relative accuracy
would give a smaller relative H error than learning Delta-H to that accuracy, predicting the
DM and building H with SIESTA is worth training. (On graphene 6x6 it was not: 7.7 NO_GO.)

Errors: ``rel`` (Gaussian noise proportional to each element of Delta-DM), ``abs`` (same rms
on every stored element, also where Delta-DM is ~0), ``relq``/``absq`` (the same keeping every
atom's Mulliken charge, what a model trained with the 4.5 Mulliken loss could enforce) and
``t`` (DM_atomic + t Delta-DM: an under-learned correction). DM_atomic is SIESTA's initial DM
(each orbital with its atomic occupation).
"""
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .delta import H0_KEYS, _aligned_csr, _read, combine_hamiltonians
from .runner import _validate_hamiltonian, find_pseudos, replace_fdf_keys, run_siesta_into, system_label

__all__ = [
    "DM_KEYS",
    "LEVELS",
    "run_from_dm",
    "aligned_error",
    "symmetric",
    "keep_charges",
    "pilot_structure",
]

# One SCF step from a saved DM: H of that DM (as H0, but DM.UseSaveDM T).
DM_KEYS = {**H0_KEYS, "DM.UseSaveDM": "T", "DM.Tolerance": "1e6", "SCF.H.Tolerance": "1e6 eV"}
LEVELS: Dict[str, Sequence[float]] = {
    "rel": [0.01, 0.03, 0.1, 0.3],
    "abs": [0.01, 0.03, 0.1, 0.3],
    "t": [0.7, 0.9, 0.97, 0.99],
}


def run_from_dm(reference_dir: Path, work_dir: Path, dm=None, siesta: Optional[str] = None) -> Path:
    """SIESTA in ``work_dir`` with the ``RUN.fdf`` of ``reference_dir`` and ``DM_KEYS``;
    ``dm`` (sisl DensityMatrix) is the initial density. Reuses TSHS only with a matching
    FDF/pseudopotential/DM/output receipt; validates geometry and known orbital/spin layout.
    """
    import sisl

    reference_dir, work_dir = Path(reference_dir), Path(work_dir)
    label = system_label(reference_dir / "RUN.fdf")
    fdf = replace_fdf_keys((reference_dir / "RUN.fdf").read_text(), DM_KEYS)
    geometry = dm.geometry if dm is not None else sisl.get_sile(str(reference_dir / "RUN.fdf")).read_geometry()

    def validate(path):
        _validate_hamiltonian(path, geometry, orbital_counts=None if dm is None else geometry.orbitals,
                              spin=None if dm is None else dm.spin)

    with tempfile.TemporaryDirectory() as tmp:
        extra = {}
        if dm is not None:
            dm.write(str(Path(tmp) / f"{label}.DM"))
            extra[f"{label}.DM"] = Path(tmp) / f"{label}.DM"
        return run_siesta_into(work_dir, fdf, find_pseudos(reference_dir), f"{label}.TSHS",
                               siesta=siesta, extra_files=extra, validate=validate)


def aligned_error(H, ref) -> float:
    """Gauge-aligned RMS on the union of stored H, H_ref and reference S positions.

    Supports unpolarized matrices; the gauge uses the full reference overlap, including
    positions with an exactly zero H error. Periodic images are matched physically.
    """
    if not H.spin.is_unpolarized or not ref.spin.is_unpolarized:
        raise NotImplementedError("DM pilot aligned_error supports unpolarized matrices only")
    difference = combine_hamiltonians(ref, H, -1.)
    diff = difference.tocsr(0)
    s = (sp.eye(difference.no, difference.geometry.no_s, format="csr") if ref.orthogonal
         else difference.tocsr(difference.S_idx))
    denominator = float(np.real(s.conjugate().multiply(s).sum()))
    if denominator <= 0:
        raise ValueError("Cannot align with zero overlap norm")
    c = float(np.real(s.conjugate().multiply(diff).sum())) / denominator
    full = diff - c * s
    support = sp.csr_matrix(diff.shape)
    for values in (_aligned_csr(ref, difference.geometry, 0), _aligned_csr(H, difference.geometry, 0), s):
        pattern = values.copy()
        pattern.data = np.ones(pattern.nnz)
        support = support + pattern
    return float(np.sqrt(np.abs(full).power(2).sum() / support.nnz))


def _rms(csr) -> float:
    return float(np.sqrt((csr.data**2).mean())) if csr.nnz else 0.0


def symmetric(geometry, noise_csr):
    """Symmetric part of a DM error: DM_ij(R) = DM_ji(-R); the antisymmetric part leaves the
    density unchanged and would only inflate the DM error."""
    import sisl

    noise = sisl.DensityMatrix.fromsp(geometry, noise_csr)
    return ((noise + noise.transpose()) * 0.5).tocsr(0)


def keep_charges(noise_csr, overlap_csr, geometry):
    """DM error with every atom's Mulliken charge change removed from its own diagonal
    (the charge of atom a is the sum over its rows of (DM * S))."""
    first = np.asarray(geometry.firsto)
    rows = np.asarray(noise_csr.multiply(overlap_csr).sum(axis=1)).ravel()
    change = np.add.reduceat(rows, first[:-1])
    diagonal = np.zeros(geometry.no)
    for atom in range(geometry.na):
        diagonal[first[atom] : first[atom + 1]] = change[atom] / (first[atom + 1] - first[atom])
    return noise_csr - sp.csr_matrix(
        (diagonal, (np.arange(geometry.no), np.arange(geometry.no))), shape=noise_csr.shape
    )


def pilot_structure(
    reference_dir: Path,
    h0: Path,
    work_dir: Path,
    *,
    seed: int = 0,
    levels: Dict[str, Sequence[float]] = LEVELS,
    siesta: Optional[str] = None,
    sample_id: Optional[str] = None,
) -> Dict:
    """Run the pilot for the converged calculation in ``reference_dir`` (needs its
    ``<label>.TSHS`` and ``<label>.DM``) with its ``h0`` (see ``delta.compute_h0``). Every
    SIESTA run goes to a subdirectory of ``work_dir`` (named with the seed for the random
    errors) and is reused if it finished. Returns the scales and one row per
    error kind and level (relative DM and H errors, H error in meV)."""
    import sisl

    reference_dir, work_dir = Path(reference_dir), Path(work_dir)
    label = system_label(reference_dir / "RUN.fdf")
    geometry = sisl.get_sile(str(reference_dir / f"{label}.TSHS")).read_geometry()
    H_ref = sisl.get_sile(str(reference_dir / f"{label}.TSHS")).read_hamiltonian()
    DM_ref = sisl.get_sile(str(reference_dir / f"{label}.DM")).read_density_matrix(geometry=geometry)
    H0 = _read(h0)

    # SIESTA's initial DM: diagonal, each orbital with its atomic occupation (q0 of the ion;
    # the TSHS geometry has no occupations, the fdf reads them from the ion files).
    fdf_atoms = sisl.get_sile(str(reference_dir / "RUN.fdf")).read_geometry().atoms
    q0 = np.concatenate([[orbital.q0 for orbital in atom] for atom in fdf_atoms])
    if len(q0) != geometry.no:
        raise ValueError(f"{len(q0)} atomic occupations for {geometry.no} orbitals")
    atomic_csr = sp.csr_matrix(
        (q0, (np.arange(geometry.no), np.arange(geometry.no))), shape=DM_ref.tocsr(0).shape
    )
    delta_dm = DM_ref.tocsr(0) - atomic_csr
    scale_dm, scale_h = _rms(delta_dm), aligned_error(H0, H_ref)
    overlap_csr = DM_ref.tocsr(DM_ref.S_idx)
    rows = []

    def measure(kind, level, dm_csr):
        dm = sisl.DensityMatrix.fromsp(geometry, dm_csr, S=overlap_csr)
        # Random errors depend on the seed: it names their run, so another seed never reuses them.
        name = f"{kind}_{level}" if kind in ("exact", "atomic", "t") else f"{kind}_{level}_seed{seed}"
        tshs = run_from_dm(reference_dir, work_dir / name, dm, siesta=siesta)
        err_h = aligned_error(sisl.get_sile(str(tshs)).read_hamiltonian(), H_ref)
        err_dm = _rms(dm_csr - DM_ref.tocsr(0))
        rows.append(
            dict(kind=kind, level=level, dm_rel=err_dm / scale_dm, h_rel=err_h / scale_h,
                 h_err_meV=1000 * err_h, dm_err=err_dm)
        )

    measure("exact", 0.0, DM_ref.tocsr(0))
    # Sanity: H of the atomic DM must be H0 (h_rel = 1 by definition; report its distance to H0).
    measure("atomic", 0.0, atomic_csr)
    tshs = work_dir / "atomic_0.0" / f"{label}.TSHS"
    rows[-1]["vs_h0_meV"] = 1000 * aligned_error(sisl.get_sile(str(tshs)).read_hamiltonian(), H0)
    rng = np.random.default_rng(seed)

    for level in levels["rel"]:
        noise = delta_dm.copy()
        noise.data = noise.data * rng.normal(size=noise.nnz) * level
        measure("rel", level, DM_ref.tocsr(0) + symmetric(geometry, noise))
    for level in levels["abs"]:
        noise = DM_ref.tocsr(0).copy()
        noise.data = rng.normal(size=noise.nnz) * level * scale_dm
        measure("abs", level, DM_ref.tocsr(0) + symmetric(geometry, noise))
    for level in levels["rel"]:
        noise = delta_dm.copy()
        noise.data = noise.data * rng.normal(size=noise.nnz) * level
        measure("relq", level, DM_ref.tocsr(0) + keep_charges(symmetric(geometry, noise), overlap_csr, geometry))
    for level in levels["abs"]:
        noise = DM_ref.tocsr(0).copy()
        noise.data = rng.normal(size=noise.nnz) * level * scale_dm
        measure("absq", level, DM_ref.tocsr(0) + keep_charges(symmetric(geometry, noise), overlap_csr, geometry))
    for t in levels["t"]:
        measure("t", t, atomic_csr + t * delta_dm)
    return dict(sample=sample_id or reference_dir.name, delta_dm_rms=scale_dm,
                delta_h_rms_meV=1000 * scale_h, rows=rows)
