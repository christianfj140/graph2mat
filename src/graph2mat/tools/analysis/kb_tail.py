"""Spectral floor of the H elements outside Graph2Mat's orbital graph (roadmap 8.2 control).

Graph2Mat only predicts blocks of atom pairs whose orbitals overlap (d < R_i + R_j). SIESTA's
H also has elements beyond that, from the Kleinman-Bylander projectors, which no model on
that graph can produce. ``without_kb_tail`` zeroes exactly those elements of a reference H;
the error of the truncated H against the full one is the floor any such model inherits.
With ``baseline`` (H0 of Delta-learning) the tail of H - H0 is cut instead and H0 is added
back whole, which is what a Delta model predicts.
"""
from typing import Any, Dict, Sequence, Tuple

import numpy as np

__all__ = ["without_kb_tail"]


def without_kb_tail(hamiltonian, radii: Sequence[float], baseline=None) -> Tuple[Any, Dict[str, float]]:
    """Copy of ``hamiltonian`` (sisl) without the elements of atom pairs at distance
    ``>= radii[i] + radii[j]`` (``radii``: orbital range per atom, Ang), and a report:
    number of removed non-zero elements (over all spin components), their largest magnitude
    (meV) and their sum of squares (eV^2). S and the inputs are left untouched."""
    if baseline is not None:
        from graph2mat.tools.siesta.delta import combine_hamiltonians

        truncated, report = without_kb_tail(combine_hamiltonians(hamiltonian, baseline, -1.0), radii)
        return combine_hamiltonians(truncated, baseline, 1.0), report
    truncated = hamiltonian.copy()
    csr = truncated._csr
    rows = np.repeat(np.arange(truncated.no), csr.ncol)
    slots = np.concatenate([np.arange(start, start + count) for start, count in zip(csr.ptr, csr.ncol)])
    cols = csr.col[slots]
    geometry = truncated.geometry
    ia, ja_sc = geometry.o2a(rows), geometry.o2a(cols)
    ja, image = ja_sc % geometry.na, cols // truncated.no
    distances = np.linalg.norm(
        geometry.xyz[ja] + geometry.sc_off[image] @ geometry.cell - geometry.xyz[ia], axis=1
    )
    radii = np.asarray(radii)
    outside = slots[distances >= radii[ia] + radii[ja]]
    # Every H spin component (the overlap, if any, is the last column and stays).
    n_h = truncated.spin.size(truncated.dtype)
    removed = csr._D[outside, :n_h].copy()
    csr._D[outside, :n_h] = 0
    return truncated, {
        "n_removed_elements": int(np.count_nonzero(removed)),
        "removed_max_meV": 1e3 * float(np.abs(removed).max()) if removed.size else 0.0,
        "removed_H2_eV2": float(np.sum(removed**2)),
    }
