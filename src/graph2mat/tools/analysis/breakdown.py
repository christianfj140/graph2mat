"""Where the Gamma-point H error of a prediction lives (roadmap 4.3).

``|H_pred - H_ref|`` at Gamma, after removing the per-structure gauge shift c*S (roadmap
4.2; for models trained with the gauge-invariant loss the raw error is dominated by their
free energy zero), is split by:

- on-site vs off-site blocks and neighbour shell (minimum-image atom distance);
- orbital l pair (s-s, s-p, ..., d-d);
- elements beyond the orbital ranges (d > R_i + R_j): exactly zero in SIESTA, so any error
  there is spurious (roadmap 4.6, orbital mask);
- elements that are zero in the reference.

``summarize_breakdown`` aggregates structures: each category's share of the error and of the
elements, and its mean error per element.
"""
import glob
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple, Union

import numpy as np

from .metrics import _hk, _same_channels, gauge_shift

__all__ = [
    "SHELLS_ANG",
    "orbital_info",
    "min_image_distances",
    "structure_breakdown",
    "summarize_breakdown",
]

# Graphene: on-site, 1st..6th neighbour shells, the rest.
SHELLS_ANG = (0.0, 0.1, 1.6, 2.6, 3.0, 3.9, 4.5, 5.2, 6.0, 99.0)
L_NAMES = "spdf"

BasisFiles = Union[str, Path, Iterable[Any]]


def _basis_orbitals(basis_files: BasisFiles) -> Dict[int, list]:
    """Orbitals per Z from basis files (glob or list) or from sisl atoms in the list."""
    import sisl

    items = sorted(glob.glob(str(basis_files))) if isinstance(basis_files, (str, Path)) else list(basis_files)
    basis = {}
    for item in items:
        atom = item if hasattr(item, "orbitals") else sisl.get_sile(str(item)).read_basis()
        basis[atom.Z] = atom.orbitals
    return basis


def orbital_info(geometry, species_z: np.ndarray, basis_files: BasisFiles) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Atom, l and cutoff radius of every orbital. TSHS files carry neither species nor l,
    so Z comes from the structure (``species_z``, per atom) and the orbitals from the basis
    files (e.g. ``*.ion.xml``, in SIESTA's order); ``basis_files`` is a glob or a list of
    files or of ``sisl.Atom`` objects."""
    basis = _basis_orbitals(basis_files)
    atoms = geometry.o2a(np.arange(geometry.no))
    local = np.arange(geometry.no) - geometry.a2o(atoms)
    orbs = [basis[species_z[a]][i] for a, i in zip(atoms, local)]
    return atoms, np.array([o.l for o in orbs]), np.array([o.R for o in orbs])


def min_image_distances(geometry) -> np.ndarray:
    """Atom-atom distance to the nearest periodic image, over the periodic directions."""
    from ase.geometry.geometry import general_find_mic

    xyz = geometry.xyz
    diff = xyz[None, :, :] - xyz[:, None, :]
    pbc = np.asarray(geometry.lattice.pbc, dtype=bool)
    if not pbc.any():
        return np.linalg.norm(diff, axis=-1)
    # The public shortcut may assume a reduced cell even for equivalent skewed bases.
    _, distances = general_find_mic(diff.reshape(-1, 3), geometry.cell, pbc=pbc)
    return distances.reshape(len(xyz), len(xyz))


def structure_breakdown(
    predicted: Path,
    reference: Path,
    structure_fdf: Path,
    basis_files: BasisFiles,
    shells: Sequence[float] = SHELLS_ANG,
) -> Dict[str, float]:
    """Per-category sums of the aligned |error| (``<cat>_sum``, eV) and element counts
    (``<cat>_n``) of one structure, plus ``gauge_c_meV`` and the raw error sum."""
    import sisl

    h_ref = sisl.get_sile(str(reference)).read_hamiltonian()
    h_pred = sisl.get_sile(str(predicted)).read_hamiltonian()
    geometry = h_ref.geometry
    channels = _same_channels(h_pred, h_ref)
    # (n_spin, no, no): every spin channel, the category masks broadcast over it.
    ref = _hk(h_ref, [0.0, 0.0, 0.0], channels)
    delta = _hk(h_pred, [0.0, 0.0, 0.0], channels) - ref
    overlap = np.asarray(h_ref.Sk(format="array"))
    c_star = gauge_shift(delta, overlap)
    err, aligned = np.abs(delta), np.abs(delta - c_star * overlap)

    species_z = sisl.get_sile(str(structure_fdf)).read_geometry().atoms.Z
    atoms, l, radius = orbital_info(geometry, species_z, basis_files)
    dist = min_image_distances(geometry)[atoms[:, None], atoms[None, :]]
    masks = {
        "total": np.ones(err.shape[1:], dtype=bool),
        "onsite": atoms[:, None] == atoms[None, :],
        "beyond_range": dist > radius[:, None] + radius[None, :],
        "ref_zero": (ref == 0).all(axis=0),
    }
    for lo, hi in zip(shells[:-1], shells[1:]):
        masks[f"shell_{lo:.1f}-{hi:.1f}"] = (dist >= lo) & (dist < hi)
    for li in range(l.max() + 1):
        for lj in range(li, l.max() + 1):
            masks[f"l_{L_NAMES[li]}{L_NAMES[lj]}"] = ((l[:, None] == li) & (l[None, :] == lj)) | (
                (l[:, None] == lj) & (l[None, :] == li)
            )
    out: Dict[str, float] = {f"{k}_sum": float(aligned[:, m].sum()) for k, m in masks.items()}
    out.update({f"{k}_n": int(m.sum()) * len(channels) for k, m in masks.items()})
    out.update({"gauge_c_meV": 1e3 * c_star, "raw_sum": float(err.sum())})
    return out


def summarize_breakdown(rows: Sequence[Dict[str, float]]) -> Dict:
    """Aggregate ``structure_breakdown`` rows of several structures."""
    total = sum(p["total_sum"] for p in rows)  # aligned
    n_total = sum(p["total_n"] for p in rows)
    cats = [k[:-4] for k in rows[0] if k.endswith("_sum") and k != "raw_sum"]
    return {
        "n_structures": len(rows),
        "H_MAE_meV": 1e3 * sum(p["raw_sum"] for p in rows) / n_total,
        "H_MAE_gauge_aligned_meV": 1e3 * total / n_total,
        "gauge_c_meV_mean_abs": float(np.mean([abs(p["gauge_c_meV"]) for p in rows])),
        "categories": {
            k: {
                "share_of_error": round(sum(p[f"{k}_sum"] for p in rows) / total, 4) if total else 0.0,
                "share_of_elements": round(sum(p[f"{k}_n"] for p in rows) / n_total, 4),
                "mean_error_meV": round(
                    1e3 * sum(p[f"{k}_sum"] for p in rows) / max(1, sum(p[f"{k}_n"] for p in rows)), 4
                ),
            }
            for k in cats
        },
    }
