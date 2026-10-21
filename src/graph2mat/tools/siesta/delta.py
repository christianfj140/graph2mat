"""Delta-learning on the non-self-consistent SIESTA Hamiltonian H0 (roadmap 5.1).

H0 is the H that SIESTA builds from the superposition of atomic densities: a single SCF
iteration that counts as converged (the result does not depend on the mixing weight, so it is
the H of the initial density). A model learns H - H0 from copies of the samples whose label is
that difference, and H0 is added back to its predictions. The aligned H error of H - H0 is the
H error itself (same difference, same S), and a different Fermi level of H0 is a c*S shift that
the gauge-invariant loss ignores.

Workflow::

    h0 = compute_h0(sample_dir, h0_dir)                         # one SIESTA step
    write_delta_sample(sample_dir, sample_dir / "x.TSHS", h0, delta_dir)
    # ... train on the delta samples, predict ...
    add_h0(delta_dir_prediction / "ML_prediction.HSX", h0)      # H = (H - H0) + H0

``stage_h0_overlap`` puts H0 next to a structure without labels: a ``long_range`` model
(roadmap 8.4) needs S at inference, and S depends only on geometry and basis.
"""
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from .runner import find_pseudos, replace_fdf_keys, run_siesta, system_label

__all__ = [
    "H0_KEYS",
    "h0_fdf",
    "compute_h0",
    "combine_hamiltonians",
    "write_delta_sample",
    "add_h0",
    "stage_h0_overlap",
]

# One SCF iteration that counts as converged, from the atomic densities, H written as TSHS.
H0_KEYS = {
    "MaxSCFIterations": "1",
    "DM.Tolerance": "100",
    "SCF.H.Tolerance": "100 eV",
    "DM.Require.Energy.Convergence": "F",
    "SCF.MustConverge": "F",
    "DM.UseSaveDM": "F",
    "UseSaveData": "F",
    "MD.UseSaveXV": "F",
    "MD.UseSaveCG": "F",
    # Single point: MD inputs (AB MD frames) would otherwise move the atoms.
    "MD.TypeOfRun": "CG",
    "MD.NumCGsteps": "0",
    "MD.Steps": "0",
    "TS.HS.Save": "T",
}


def h0_fdf(text: str) -> str:
    """The fdf of a converged calculation turned into the fdf of its H0."""
    return replace_fdf_keys(text, H0_KEYS)


def _read(path):
    import sisl

    return sisl.get_sile(str(path)).read_hamiltonian()


def compute_h0(
    reference_dir: Path,
    out_dir: Path,
    *,
    reference_matrix: Optional[Path] = None,
    siesta: Optional[str] = None,
    geometry_tol: float = 1e-4,
) -> Path:
    """H0 of the calculation in ``reference_dir`` (its ``RUN.fdf`` and pseudopotentials).

    SIESTA runs in ``out_dir`` (emptied first) and the result, ``<SystemLabel>.TSHS``, is
    returned; an existing one is reused without running SIESTA. The geometry of H0 is
    checked against ``reference_matrix`` (default: ``<SystemLabel>.TSHS`` of the reference,
    if it exists): if any atom moved more than ``geometry_tol`` Ang the file is removed and
    a ``RuntimeError`` raised.
    """
    import sisl

    reference_dir, out_dir = Path(reference_dir), Path(out_dir)
    label = system_label(reference_dir / "RUN.fdf")
    out = out_dir / f"{label}.TSHS"
    if out.exists():
        return out
    if reference_matrix is None and (reference_dir / f"{label}.TSHS").exists():
        reference_matrix = reference_dir / f"{label}.TSHS"
    run_siesta(
        out_dir,
        h0_fdf((reference_dir / "RUN.fdf").read_text()),
        find_pseudos(reference_dir),
        siesta=siesta,
    )
    if not out.exists():
        raise RuntimeError(f"SIESTA wrote no {out.name} in {out_dir}")
    if reference_matrix is not None:
        moved = np.abs(
            sisl.get_sile(str(out)).read_geometry().xyz
            - sisl.get_sile(str(reference_matrix)).read_geometry().xyz
        ).max()
        if moved > geometry_tol:
            out.unlink()
            raise RuntimeError(
                f"H0 geometry differs from the label's by {moved:.4f} Ang in {out_dir}"
            )
    return out


def combine_hamiltonians(a, b, sign: float):
    """``a + sign * b`` for every spin component, with the overlap of ``a``.

    When the supercells differ both are brought to the larger one (elementwise maximum of
    ``nsc``), so no coupling of either is dropped: SIESTA's H0 usually reaches more images
    than a model's prediction.
    """
    import sisl

    if a.spin.kind != b.spin.kind:
        raise ValueError(f"cannot combine Hamiltonians with spin {a.spin} and {b.spin}")
    if not np.array_equal(b.nsc, a.nsc):
        nsc = np.maximum(a.nsc, b.nsc)
        a, b = a.copy(), b.copy()
        a.set_nsc(nsc)
        b.set_nsc(nsc)
    n_spin = a.spin.size(a.dtype)
    h = [a.tocsr(i) + sign * b.tocsr(i) for i in range(n_spin)]
    S = None if a.orthogonal else a.tocsr(a.S_idx)
    if n_spin == 1:
        return sisl.Hamiltonian.fromsp(a.geometry, h[0], S=S)
    return sisl.Hamiltonian.fromsp(a.geometry, h, S=S, spin=a.spin)


def write_delta_sample(
    reference_dir: Path, reference_matrix: Path, h0: Path, out_dir: Path
) -> Path:
    """Copy of a sample whose label is H - H0: ``RUN.fdf`` of ``reference_dir`` and the
    difference, with the overlap of the reference, under the reference matrix's name."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(reference_dir) / "RUN.fdf", out_dir / "RUN.fdf")
    label = out_dir / Path(reference_matrix).name
    combine_hamiltonians(_read(reference_matrix), _read(h0), -1.0).write(str(label))
    return label


def add_h0(predicted: Path, h0: Path, out: Optional[Path] = None) -> Path:
    """Prediction of H - H0 to prediction of H (in place unless ``out`` is given). The
    overlap stays the prediction's."""
    out = Path(predicted) if out is None else Path(out)
    combine_hamiltonians(_read(predicted), _read(h0), 1.0).write(str(out))
    return out


def stage_h0_overlap(h0: Path, structure_fdf: Path, overwrite: bool = False) -> Path:
    """Copy H0 next to ``structure_fdf`` as ``<SystemLabel>.TSHS``, the file a ``long_range``
    model reads S from at inference. Refuses to replace an existing file (it could be the
    reference) unless ``overwrite``."""
    structure_fdf = Path(structure_fdf)
    dest = structure_fdf.parent / f"{system_label(structure_fdf)}.TSHS"
    if dest.exists() and not overwrite:
        raise FileExistsError(f"{dest} exists; pass overwrite=True to replace it")
    shutil.copy2(h0, dest)
    return dest
