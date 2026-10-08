"""Shared physical-format checks and atomic matrix output for server callers."""

import os
from pathlib import Path
import tempfile

import numpy as np
import sisl


OUTPUT_SUFFIXES = {
    "density_matrix": (".DM", ".nc"),
    "hamiltonian": (".nc", ".TSHS"),
    "energy_density_matrix": (".nc",),
}


def prediction_suffix(matrix_type, requested=None):
    supported = OUTPUT_SUFFIXES.get(matrix_type, (".mtx",))
    if requested is not None and requested not in supported:
        raise ValueError(f"{matrix_type} output supports only {', '.join(supported)}.")
    return requested or supported[0]


def write_prediction(matrix, path, *, fermi_level=None, orthogonal_basis=False, allow_overwrite=True):
    """Validate before writing; publish only a successfully serialized matrix.

    H is supplied in the same energy gauge as fermi_level. The nc reader in
    sisl subtracts Ef*S; Graph2Mat does not apply an additional gauge shift.
    A standalone EDM stays in its supplied gauge (no DM/Ef correction invented).
    """
    path = Path(path)
    kwargs = {}
    if isinstance(matrix, sisl.SparseOrbital):
        if not np.isfinite(matrix._csr.data).all() or matrix.nnz == 0:
            raise ValueError("Physical matrix output requires finite observed entries.")
    if isinstance(matrix, sisl.Hamiltonian):
        prediction_suffix("hamiltonian", path.suffix)
        if fermi_level is None or not np.isfinite(fermi_level):
            raise ValueError("Hamiltonian output requires an explicit finite fermi_level (eV).")
        if matrix.orthogonal:
            if not orthogonal_basis:
                raise ValueError("Hamiltonian output requires explicit overlap S or orthogonal_basis=True.")
        elif np.any(matrix.tocsr(matrix.S_idx).diagonal() <= 0):
            raise ValueError("Hamiltonian overlap S must have a positive finite diagonal.")
        if path.suffix == ".TSHS" and fermi_level != 0:
            raise ValueError("sisl's TSHS writer cannot preserve nonzero fermi_level; use .nc.")
        kwargs["Ef"] = float(fermi_level)
    elif isinstance(matrix, sisl.EnergyDensityMatrix):
        prediction_suffix("energy_density_matrix", path.suffix)
        if fermi_level is not None and fermi_level != 0:
            raise ValueError("Standalone EDM cannot apply nonzero fermi_level without a density matrix.")
        kwargs["Ef"] = 0.
    elif isinstance(matrix, sisl.DensityMatrix):
        prediction_suffix("density_matrix", path.suffix)

    with tempfile.NamedTemporaryFile(prefix=".g2m-", suffix=path.suffix, dir=path.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        matrix.write(temporary, **kwargs)
        if allow_overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication, including races after prediction.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
