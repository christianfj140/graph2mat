"""Evaluation of predicted matrices against SIESTA references, from files.

Every function takes the predicted and the reference matrix files (TSHS/HSX). sisl refers
H to the Fermi level (H - E_F S), so E_F is the energy zero. Predictions of models that
only output H (``h_only``) are written with S = 1: their eigenvalues are always solved with
the reference S, which depends only on geometry and basis.

Spin: unpolarized and collinear spin-polarized matrices are supported. For polarized ones
every metric covers both spin channels (H errors over both, eigenvalues of both channels
merged and sorted); other spin configurations raise ``NotImplementedError``.

Gauge: H + c S is the same physics for any constant c, and a local model cannot know the
energy zero of a new structure. ``gauge_shift`` is the least-squares c* of an error dH
(``dH ~ c* S``, one c for every spin channel); "aligned"/"gauge" metrics remove it. Band
metrics "aligned" remove the mean offset in their window instead.

The default k path is G-K-M-G of hexagonal lattices (graphene, hBN, TMDs); pass ``kpath``
(fractional coordinates) for other lattices.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "HEXAGONAL_PATH",
    "INTERLAYER_DZ_ANG",
    "CANON_THRESHOLD",
    "spin_channels",
    "gauge_shift",
    "gamma_hk",
    "gamma_eigenvalues",
    "gamma_metrics",
    "layer_masks",
    "path_bands",
    "layer_metrics",
    "band_dos_metrics",
    "canonical_bands",
    "band_window_metrics",
    "species_pair_errors",
]

HEXAGONAL_PATH = [[0, 0, 0], [1 / 3, 2 / 3, 0], [0.5, 0.5, 0], [0, 0, 0]]
INTERLAYER_DZ_ANG = 1.5  # orbital pairs farther apart than this in z are interlayer
CANON_THRESHOLD = 1e-2  # S eigenvalues below this are dropped (canonical orthogonalization)


def _read(path):
    import sisl

    return sisl.get_sile(str(path)).read_hamiltonian()


def spin_channels(h) -> List[int]:
    """H spin components of a sisl Hamiltonian that the metrics evaluate."""
    if h.spin.is_unpolarized:
        return [0]
    if h.spin.is_polarized:
        return [0, 1]
    raise NotImplementedError(
        f"{h.spin} matrices are not supported: only unpolarized and collinear spin-polarized ones."
    )


def _same_channels(h_pred, h_ref) -> List[int]:
    channels = spin_channels(h_ref)
    if spin_channels(h_pred) != channels:
        raise ValueError(f"prediction spin {h_pred.spin} differs from reference spin {h_ref.spin}")
    return channels


def _spin(h, s) -> Dict[str, int]:
    # sisl only takes ``spin=`` for spin-polarized matrices.
    return {"spin": s} if h.spin.is_polarized else {}


def _hk(h, k, channels) -> np.ndarray:
    """(n_spin, no, no) dense H at k."""
    return np.stack([np.asarray(h.Hk(k=k, format="array", **_spin(h, s))) for s in channels])


def _eig(h, k, channels, overlap=None) -> np.ndarray:
    """Eigenvalues of every spin channel at k, merged and sorted; with the given S if any."""
    import scipy.linalg

    if overlap is None:
        return np.sort(np.concatenate([h.eigh(k=k, **_spin(h, s)) for s in channels]))
    return np.sort(np.concatenate([
        scipy.linalg.eigh(h.Hk(k=k, format="array", **_spin(h, s)), overlap, eigvals_only=True) for s in channels
    ]))


def gauge_shift(delta: np.ndarray, overlap: np.ndarray) -> float:
    """Least-squares c of ``delta ~ c * overlap`` (``overlap`` broadcast over spin channels)."""
    overlap = np.broadcast_to(overlap, np.shape(delta))
    return float(np.real(np.vdot(overlap, delta)) / np.real(np.vdot(overlap, overlap)))


def gamma_hk(matrix: Path) -> np.ndarray:
    """Dense H at Gamma, shape (n_spin, no, no)."""
    h = _read(matrix)
    return _hk(h, [0.0, 0.0, 0.0], spin_channels(h))


def gamma_eigenvalues(matrix: Path, overlap_from: Optional[Path] = None) -> np.ndarray:
    """Gamma eigenvalues of ``matrix`` (all spin channels, sorted), solved with the S of
    ``overlap_from`` if given."""
    hamiltonian = _read(matrix)
    channels = spin_channels(hamiltonian)
    if overlap_from is None:
        return _eig(hamiltonian, [0.0, 0.0, 0.0], channels)
    overlap = _read(overlap_from).Sk(k=[0.0, 0.0, 0.0], format="array")
    return _eig(hamiltonian, [0.0, 0.0, 0.0], channels, overlap)


def _gamma_overlap(reference: Path) -> np.ndarray:
    import sisl

    return np.asarray(
        sisl.get_sile(str(reference)).read_overlap().Sk(k=[0, 0, 0], format="array")
    )


def _gamma_pair(predicted: Path, reference: Path) -> Tuple[np.ndarray, np.ndarray]:
    h_pred, h_ref = _read(predicted), _read(reference)
    channels = _same_channels(h_pred, h_ref)
    return _hk(h_pred, [0.0, 0.0, 0.0], channels), _hk(h_ref, [0.0, 0.0, 0.0], channels)


def gamma_metrics(predicted: Path, reference: Path, window_ev: float = 2.0) -> Dict[str, float]:
    """Gamma-point H errors (meV), relative Frobenius error, hermiticity of the prediction
    (eV) and RMSE of the Gamma eigenvalues within ``window_ev`` of E_F (meV)."""
    h_pred, h_ref = _gamma_pair(predicted, reference)
    error = h_pred - h_ref
    ref_norm = float(np.linalg.norm(h_ref))
    overlap = _gamma_overlap(reference)
    c_star = gauge_shift(error, overlap)
    eig_pred = gamma_eigenvalues(predicted, reference)
    eig_ref = gamma_eigenvalues(reference)
    mask = np.abs(eig_ref) <= window_ev
    return {
        "H_MAE_meV": 1e3 * float(np.mean(np.abs(error))),
        "H_RMSE_meV": 1e3 * float(np.sqrt(np.mean(np.abs(error) ** 2))),
        "H_MAE_gauge_meV": 1e3 * float(np.mean(np.abs(error - c_star * overlap))),
        "rel_Frob": float(np.linalg.norm(error) / ref_norm) if ref_norm else float("nan"),
        "hermiticity_eV": float(np.max(np.abs(h_pred - np.swapaxes(h_pred, -1, -2).conj()))),
        "spectral_err_meV": 1e3 * float(np.sqrt(np.mean((eig_pred[mask] - eig_ref[mask]) ** 2)))
        if mask.any()
        else float("nan"),
    }


def layer_masks(reference: Path, interlayer_dz: float = INTERLAYER_DZ_ANG) -> Tuple[np.ndarray, np.ndarray]:
    """(intralayer, interlayer) masks of the orbital pairs of a layered structure."""
    import sisl

    geometry = sisl.get_sile(str(reference)).read_geometry()
    z = geometry.xyz[geometry.o2a(np.arange(geometry.no)), 2]
    inter = np.abs(z[:, None] - z[None, :]) > interlayer_dz
    return ~inter, inter


def path_bands(
    predicted: Path, reference: Path, kpath: Sequence = HEXAGONAL_PATH, n_per_segment: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues along ``kpath`` of the prediction (with the reference S) and of the reference."""
    import sisl

    h_pred, h_ref = _read(predicted), _read(reference)
    channels = _same_channels(h_pred, h_ref)
    path = sisl.BandStructure(h_ref.geometry, kpath, n_per_segment)
    eig_pred = np.array([_eig(h_pred, k, channels, h_ref.Sk(k=k, format="array")) for k in path.k])
    return eig_pred, np.array([_eig(h_ref, k, channels) for k in path.k])


def layer_metrics(
    predicted: Path,
    reference: Path,
    *,
    kpath: Sequence = HEXAGONAL_PATH,
    n_per_segment: int = 40,
    window_ev: float = 2.0,
    interlayer_dz: float = INTERLAYER_DZ_ANG,
) -> Dict[str, float]:
    """Gamma H errors as is and gauge-aligned, split into intra- and interlayer orbital pairs,
    and band RMSE along ``kpath`` within ``window_ev`` of E_F, as is and aligned (meV).

    c* is only meaningful when the error is mostly a rigid shift of the energy zero.
    """
    h_pred, h_ref = _gamma_pair(predicted, reference)
    delta = h_pred - h_ref
    overlap = _gamma_overlap(reference)
    c_star = gauge_shift(delta, overlap)
    aligned = np.abs(delta - c_star * overlap)
    error = np.abs(delta)
    intra, inter = layer_masks(reference, interlayer_dz)
    eig_pred, eig_ref = path_bands(predicted, reference, kpath, n_per_segment)
    window = np.abs(eig_ref) <= window_ev

    def rmse(d):
        return 1e3 * float(np.sqrt(np.mean(d[window] ** 2))) if window.any() else float("nan")

    diff = eig_pred - eig_ref
    return {
        "gauge_c_meV": 1e3 * c_star,
        "H_MAE_meV": 1e3 * float(error.mean()),
        "H_MAE_intra_meV": 1e3 * float(error[:, intra].mean()),
        "H_MAE_inter_meV": 1e3 * float(error[:, inter].mean()),
        "H_MAE_gauge_meV": 1e3 * float(aligned.mean()),
        "H_MAE_gauge_intra_meV": 1e3 * float(aligned[:, intra].mean()),
        "H_MAE_gauge_inter_meV": 1e3 * float(aligned[:, inter].mean()),
        "band_rmse_meV": rmse(diff),
        "band_rmse_aligned_meV": rmse(diff - diff[window].mean()) if window.any() else float("nan"),
    }


def band_dos_metrics(
    predicted: Path,
    reference: Path,
    *,
    kpath: Sequence = HEXAGONAL_PATH,
    n_per_segment: int = 60,
    grid: Sequence[int] = (24, 24, 1),
    window_ev: float = 2.0,
    dos_range_ev: float = 3.0,
    dos_points: int = 601,
    sigma_ev: float = 0.1,
) -> Dict[str, float]:
    """Band RMSE along ``kpath`` within ``window_ev`` of E_F (meV) and relative L1 difference
    of the Gaussian-broadened DOS (``sigma_ev``) in E_F +- ``dos_range_ev`` on a
    Monkhorst-Pack ``grid`` (the graphene benchmarks used 24x24x1 for the 2-atom cell and
    4x4x1 for the 6x6 supercell)."""
    import sisl

    h_pred, h_ref = _read(predicted), _read(reference)
    channels = _same_channels(h_pred, h_ref)

    def eig_pred_at(k: Any) -> np.ndarray:
        return _eig(h_pred, k, channels, h_ref.Sk(k=k, format="array"))

    def eig_ref_at(k: Any) -> np.ndarray:
        return _eig(h_ref, k, channels)

    path = sisl.BandStructure(h_ref.geometry, kpath, n_per_segment)
    eig_ref = np.array([eig_ref_at(k) for k in path.k])
    eig_pred = np.array([eig_pred_at(k) for k in path.k])
    mask = np.abs(eig_ref) <= window_ev
    mp = sisl.MonkhorstPack(h_ref.geometry, list(grid))
    energies = np.linspace(-dos_range_ev, dos_range_ev, dos_points)

    def dos(eig_at: Any) -> np.ndarray:
        eig = np.concatenate([eig_at(k) for k in mp.k])
        return np.exp(-((energies[:, None] - eig[None, :]) ** 2) / (2 * sigma_ev**2)).sum(axis=1)

    d_ref, d_pred = dos(eig_ref_at), dos(eig_pred_at)
    return {
        "band_rmse_meV": 1e3 * float(np.sqrt(np.mean((eig_pred[mask] - eig_ref[mask]) ** 2))),
        "dos_rel_L1": float(np.abs(d_pred - d_ref).sum() / d_ref.sum()),
    }


def canonical_bands(h, s, path_k: Sequence, threshold: float) -> np.ndarray:
    """Eigenvalues of ``h`` along ``path_k`` with the overlap of ``s`` (sisl Hamiltonians),
    every spin channel of ``h`` merged and sorted.

    ``threshold > 0`` solves in the subspace of the S eigenvectors with the largest
    eigenvalues (canonical orthogonalization), the same for reference and prediction. The
    subspace has the same size at every k (the smallest count of S eigenvalues above the
    threshold on the path), so that bands can be compared by index.
    """
    channels = spin_channels(h)
    if threshold <= 0:
        return np.array([_eig(h, k, channels, s.Sk(k=k, format="array")) for k in path_k])
    decomposed = [np.linalg.eigh(s.Sk(k=k, format="array")) for k in path_k]
    n_keep = min(int((w > threshold).sum()) for w, _ in decomposed)
    if n_keep == 0:
        raise ValueError("Canonical overlap subspace is empty at this threshold")
    out = []
    for k, (w, u) in zip(path_k, decomposed):
        keep = np.argsort(w)[-n_keep:]
        x = u[:, keep] / np.sqrt(w[keep])
        out.append(np.sort(np.concatenate([
            np.linalg.eigvalsh(x.conj().T @ h.Hk(k=k, format="array", **_spin(h, c)) @ x) for c in channels
        ])))
    return np.array(out)


def band_window_metrics(
    predicted: Path,
    reference: Path,
    *,
    kpath: Sequence = HEXAGONAL_PATH,
    n_per_segment: int = 20,
    canon_threshold: float = CANON_THRESHOLD,
) -> Dict[str, float]:
    """Bands and gap of insulators (eV/meV).

    A window around E_F can be empty for insulators (E_F sits mid-gap): here the window is
    [VBM - 2 eV, CBM + 2 eV] of the reference, bands are compared by index (the reference's
    occupation) and the gap error is reported; "aligned" removes the mean offset in the
    window. With diffuse bases S is nearly singular and a few meV of H error in its
    near-null directions make spurious in-gap states: the ``canon_`` metrics solve in the
    subspace of S eigenvalues above ``canon_threshold``.
    """
    import sisl

    hp, hr = _read(predicted), _read(reference)
    _same_channels(hp, hr)
    path = sisl.BandStructure(hr.geometry, kpath, n_per_segment).k
    out: Dict[str, float] = {}
    for tag, threshold in (("", 0.0), ("canon_", canon_threshold)):
        eig_ref, eig_pred = canonical_bands(hr, hr, path, threshold), canonical_bands(hp, hr, path, threshold)
        n_occ = int((eig_ref[0] < 0).sum())  # E_F = 0; insulators: same count at every k
        if not 0 < n_occ < eig_ref.shape[1]:
            raise ValueError(f"no occupied/unoccupied boundary in the bands of {reference}")
        vbm, cbm = float(eig_ref[:, n_occ - 1].max()), float(eig_ref[:, n_occ].min())
        window = (eig_ref >= vbm - 2.0) & (eig_ref <= cbm + 2.0)
        diff = eig_pred - eig_ref
        aligned = diff - diff[window].mean()
        gap_pred = float(eig_pred[:, n_occ].min() - eig_pred[:, n_occ - 1].max())
        out.update(
            {
                f"{tag}gap_ref_eV": cbm - vbm,
                f"{tag}gap_err_meV": 1e3 * (gap_pred - (cbm - vbm)),
                f"{tag}band_edges_rmse_aligned_meV": 1e3 * float(np.sqrt(np.mean(aligned[window] ** 2))),
            }
        )
    return out


def species_pair_errors(predicted: Path, reference: Path, structure_fdf: Path) -> Dict[str, float]:
    """Gauge-aligned Gamma |dH| averaged per species pair (meV), over every spin channel.
    Species come from the structure's fdf (TSHS files do not carry them)."""
    import sisl

    ref = _read(reference)
    h_pred, h_ref = _gamma_pair(predicted, reference)
    delta = h_pred - h_ref
    overlap = np.asarray(ref.Sk(format="array"))
    aligned = np.abs(delta - gauge_shift(delta, overlap) * overlap)
    geom = sisl.get_sile(str(structure_fdf)).read_geometry()
    sym = np.array([geom.atoms[i].symbol for i in range(geom.na)])[ref.geometry.o2a(np.arange(ref.no))]
    out = {}
    for a in sorted(set(sym)):
        for b in sorted(set(sym)):
            if a <= b:
                m = np.outer(sym == a, sym == b) | np.outer(sym == b, sym == a)
                out[f"{a}-{b}"] = 1e3 * float(aligned[:, m].mean())
    return out
