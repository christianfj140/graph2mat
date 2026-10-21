"""Projection files of the spectral losses (roadmap 4.4 and 4.7).

``MatrixDataModule(spectral_window=True)`` reads ``spectral_window.npz`` next to every
label, and ``spectral_window="spectral_null.npz"`` the null-space file. Each holds ``C``,
S-orthonormal columns at Gamma; the loss penalizes ``C^T dH C``. The files of one dataset
are zero-padded to the same width so that they batch together.

- Window (4.4): eigenvectors of the full label H (not H - H0) with ``|e - E_F| <= window_ev``
  (sisl puts E_F at 0).
- Null space (4.7): directions of S with eigenvalue ``lambda < threshold``, scaled by
  ``1 / sqrt(max(lambda, floor))`` (canonical orthogonalization). ``C^T dH C`` is then dH in
  the Lowdin metric on that subspace, where an H error is amplified by ``1 / lambda`` and
  makes the spurious in-gap states of diffuse bases. The ``floor`` keeps a few tiny
  eigenvalues from dominating (those columns have norm < 1 in S).
"""
import os
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "SPECTRAL_WINDOW_EV",
    "WINDOW_FILE",
    "NULL_FILE",
    "spectral_window_columns",
    "null_space_columns",
    "write_projection_files",
]

SPECTRAL_WINDOW_EV = 3.0  # 6x6 graphene at Gamma: +-2 eV holds only the 4 Dirac states, +-3 eV 34
WINDOW_FILE = "spectral_window.npz"
NULL_FILE = "spectral_null.npz"


def spectral_window_columns(matrix: Path, window_ev: float = SPECTRAL_WINDOW_EV) -> np.ndarray:
    """S-orthonormal Gamma eigenvectors of the H in ``matrix`` within ``window_ev`` of E_F."""
    import scipy.linalg
    import sisl

    h = sisl.get_sile(str(matrix)).read_hamiltonian()
    e, v = scipy.linalg.eigh(h.Hk(format="array"), h.Sk(format="array"))
    return v[:, np.abs(e) <= window_ev]


def null_space_columns(matrix: Path, threshold: float = 1e-2, floor: float = 1e-3) -> np.ndarray:
    """Near-null directions of the Gamma overlap of ``matrix``, scaled as in the module docstring."""
    import sisl

    lam, u = np.linalg.eigh(sisl.get_sile(str(matrix)).read_hamiltonian().Sk(format="array"))
    keep = lam < threshold
    return u[:, keep] / np.sqrt(np.maximum(lam[keep], floor))


def write_projection_files(
    columns: Sequence[np.ndarray], out_dirs: Sequence[Path], name: str
) -> int:
    """Write ``columns[i]`` as ``out_dirs[i] / name`` (key ``C``, float32), zero-padded to the
    widest. Writes are atomic (concurrent readers never see a partial file). Returns the width."""
    if len(columns) != len(out_dirs):
        raise ValueError(f"{len(columns)} column sets for {len(out_dirs)} directories")
    width = max(c.shape[1] for c in columns)
    for c, out_dir in zip(columns, out_dirs):
        tmp = Path(out_dir) / f".{Path(name).stem}.{os.getpid()}.npz"
        np.savez(tmp, C=np.pad(c, ((0, 0), (0, width - c.shape[1]))).astype(np.float32))
        os.replace(tmp, Path(out_dir) / name)
    return width
