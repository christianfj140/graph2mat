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
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from .runner import _fdf_label, _validate_hamiltonian, find_pseudos, replace_fdf_keys, run_siesta_into, system_label

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


def _fdf_value(text: str, key: str) -> Optional[str]:
    """Value of the first fdf line that sets ``key`` (fdf label rules), if any."""
    wanted = re.sub(r"[._-]", "", key).lower()
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and re.sub(r"[._-]", "", parts[0]).lower() == wanted:
            return re.split(r"[#!]", parts[1], maxsplit=1)[0].strip()
    return None


def _non_collinear(text: str) -> bool:
    """Whether the fdf asks for non-collinear or spin-orbit spin."""
    spin = re.sub(r"[\s._-]", "", (_fdf_value(text, "Spin") or "").lower())
    if spin.startswith(("spinorbit", "so", "noncolinear", "noncollinear", "nc")):
        return True
    legacy = [(_fdf_value(text, key) or "").lower() for key in ("SpinOrbit", "NonCollinearSpin")]
    return any(value in ("t", "true", ".true.", "yes") for value in legacy)


def h0_fdf(text: str, n_atoms: Optional[int] = None) -> str:
    """The fdf of a converged calculation turned into the fdf of its H0.

    Non-collinear and spin-orbit runs start from a spin-polarized density in SIESTA
    (ferromagnetic by default): their H0 gets ``DM.InitSpin`` with zero moment on all
    ``n_atoms`` atoms (any previous ``DM.InitSpin`` block is replaced), so that H0 is the
    H of the superposition of non-magnetic atomic densities (roadmap 8.1, F0).
    """
    out = replace_fdf_keys(text, H0_KEYS)
    if _non_collinear(text):
        if n_atoms is None:
            raise ValueError("h0_fdf needs n_atoms for a non-collinear or spin-orbit fdf.")
        kept, inside = [], False
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].lower() == "%block" and _fdf_label(parts[1]) == "dminitspin":
                inside = True
            elif inside and parts and parts[0].lower() == "%endblock":
                inside = False
            elif not inside:
                kept.append(line)
        out = "\n".join(kept) + "\n"
        out += "%block DM.InitSpin\n" + "".join(f" {i + 1} 0.0\n" for i in range(n_atoms)) + "%endblock DM.InitSpin\n"
    return out


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

    The result, ``out_dir / <SystemLabel>.TSHS``, is returned; an existing one is reused
    without running SIESTA if its input/output receipt matches. ``out_dir`` only appears
    when SIESTA succeeds and validation passes (see ``runner.run_siesta_into``).
    The geometry of H0, also a reused one, is checked against
    ``reference_matrix`` (default: ``<SystemLabel>.TSHS`` of the reference, if it exists):
    if any atom moved more than ``geometry_tol`` Ang a ``RuntimeError`` is raised.
    """
    import sisl

    reference_dir, out_dir = Path(reference_dir), Path(out_dir)
    label = system_label(reference_dir / "RUN.fdf")
    if reference_matrix is None and (reference_dir / f"{label}.TSHS").exists():
        reference_matrix = reference_dir / f"{label}.TSHS"
    geometry = sisl.get_sile(str(reference_dir / "RUN.fdf")).read_geometry()
    reference = _read(reference_matrix) if reference_matrix is not None else None

    def validate(path):
        _validate_hamiltonian(path, geometry if reference is None else reference.geometry,
                              orbital_counts=None if reference is None else reference.geometry.orbitals,
                              spin=None if reference is None else reference.spin, geometry_tol=geometry_tol)

    return run_siesta_into(
        out_dir,
        h0_fdf((reference_dir / "RUN.fdf").read_text(), geometry.na),
        find_pseudos(reference_dir),
        f"{label}.TSHS",
        siesta=siesta,
        validate=validate,
    )


def combine_hamiltonians(a, b, sign: float):
    """``a + sign * b`` for every spin component, with the overlap of ``a``.

    When the supercells differ both are brought to the larger one (elementwise maximum of
    ``nsc``), so no coupling of either is dropped: SIESTA's H0 usually reaches more images
    than a model's prediction.
    """
    import sisl

    if a.spin.kind != b.spin.kind:
        raise ValueError(f"cannot combine Hamiltonians with spin {a.spin} and {b.spin}")
    # sisl packs SOC's 8 real components into 4 complex ones. Match that storage
    # convention before adding components; casting each CSR separately loses it.
    dtype = np.result_type(a.dtype, b.dtype)
    a, b = a.astype(dtype, copy=False), b.astype(dtype, copy=False)
    if (a.geometry.na != b.geometry.na or a.no != b.no
            or not np.array_equal(a.geometry.orbitals, b.geometry.orbitals)
            or not np.allclose(a.geometry.xyz, b.geometry.xyz, rtol=0, atol=1e-6)
            or not np.allclose(a.geometry.cell, b.geometry.cell, rtol=0, atol=1e-6)):
        raise ValueError("Hamiltonian geometry or orbital layout differs")
    if not np.array_equal(b.nsc, a.nsc):
        nsc = np.maximum(a.nsc, b.nsc)
        a, b = a.copy(), b.copy()
        a.set_nsc(nsc)
        b.set_nsc(nsc)
    n_spin = a.spin.size(a.dtype)
    h = [a.tocsr(i) + sign * _aligned_csr(b, a.geometry, i) for i in range(n_spin)]
    S = None if a.orthogonal else a.tocsr(a.S_idx)
    if n_spin == 1:
        return sisl.Hamiltonian.fromsp(a.geometry, h[0], S=S)
    return sisl.Hamiltonian.fromsp(a.geometry, h, S=S, spin=a.spin)


def _aligned_csr(matrix, geometry, component):
    """One component with columns indexed by the target's physical periodic images."""
    from scipy.sparse import csr_matrix

    csr = matrix.tocsr(component)
    if np.array_equal(matrix.geometry.sc_off, geometry.sc_off):
        return csr
    coo = csr.tocoo()
    images = geometry.sc_index(matrix.geometry.sc_off[coo.col // matrix.no])
    return csr_matrix((coo.data, (coo.row, images * matrix.no + coo.col % matrix.no)),
                      shape=(matrix.no, geometry.no_s))


def project_time_reversal(h) -> Tuple[Any, float]:
    """Spin-orbit ``h`` projected onto its time-reversal-symmetric channels ``[h0, gx, gy, gz]``
    (overlap kept), and the relative TRS residual that the projection removed.

    SCF noise makes SIESTA's spin-orbit H break time reversal by ~1e-5 eV, which is of the
    order of ``SOC_TRS_RTOL`` relative to the much smaller H - H0: the data loader would
    reject such labels. The residual is returned so that it is recorded, not hidden.
    """
    import sisl
    from scipy.sparse import csr_array

    from graph2mat.core.data.spin_orbit import siesta_to_soc_full_channels, soc_channels_to_siesta, soc_trs_residual

    if not h.spin.is_spinorbit:
        raise ValueError(f"time-reversal projection needs a spin-orbit H, not {h.spin}")
    if np.issubdtype(h.dtype, np.complexfloating):
        if not h.orthogonal and np.any(h.tocsr(h.S_idx).data.imag != 0):
            raise ValueError("Time-reversal projection requires a real overlap")
        h = h.astype(np.float64)
    stacked = np.stack([h.tocsr(i).toarray() for i in range(8)], -1)
    residual = float(soc_trs_residual(stacked))
    projected = soc_channels_to_siesta(siesta_to_soc_full_channels(stacked)[..., :4])
    support = np.abs(stacked).sum(-1) != 0
    components = [csr_array(projected[..., i] * support) for i in range(8)]
    S = None if h.orthogonal else h.tocsr(h.S_idx)
    return sisl.Hamiltonian.fromsp(h.geometry, components, S=S, spin=sisl.Spin("spin-orbit")), residual


def write_delta_sample(
    reference_dir: Path, reference_matrix: Path, h0: Path, out_dir: Path, project_trs: bool = False
) -> Path:
    """Copy of a sample whose label is H - H0: ``RUN.fdf`` of ``reference_dir`` and the
    difference, with the overlap of the reference, under the reference matrix's name.

    ``project_trs``: for time-reversal-symmetric spin-orbit labels (policy ``spin_orbit``),
    project H - H0 onto the TRS channels (``project_time_reversal``) and record the removed
    residual in ``trs_residual.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(reference_dir) / "RUN.fdf", out_dir / "RUN.fdf")
    label = out_dir / Path(reference_matrix).name
    delta = combine_hamiltonians(_read(reference_matrix), _read(h0), -1.0)
    if project_trs:
        delta, residual = project_time_reversal(delta)
        (out_dir / "trs_residual.json").write_text(json.dumps({"relative_trs_residual": residual}))
    delta.write(str(label))
    return label


def add_h0(predicted: Path, h0: Path, out: Optional[Path] = None) -> Path:
    """Prediction of H - H0 to prediction of H (in place unless ``out`` is given, written to
    a temporary file first). The overlap is the prediction's, or H0's when the prediction
    has none (H-only and spin-orbit predictions are written without S; S depends only on
    geometry and basis, so H0's is the reference S)."""
    import sisl

    out = Path(predicted) if out is None else Path(out)
    prediction, base = _read(predicted), _read(h0)
    result = combine_hamiltonians(prediction, base, 1.0)
    if result.orthogonal and not base.orthogonal:
        n_spin = result.spin.size(result.dtype)
        values = [result.tocsr(i) for i in range(n_spin)]
        overlap = base.copy()
        overlap.set_nsc(result.nsc)
        result = sisl.Hamiltonian.fromsp(
            result.geometry, values[0] if n_spin == 1 else values,
            S=_aligned_csr(overlap, result.geometry, overlap.S_idx),
            **({} if n_spin == 1 else {"spin": result.spin}),
        )
    tmp = out.with_name(f".{out.stem}.tmp-{os.getpid()}{out.suffix}")
    result.write(str(tmp))
    os.replace(tmp, out)
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
