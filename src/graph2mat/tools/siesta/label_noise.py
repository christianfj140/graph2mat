"""Noise floor of SIESTA labels (roadmap 5.4).

A converged calculation is recomputed with physically irrelevant changes, and the Gamma-point
H MAE against the original measures how much the label moves:

- ``tight``: DM.Tolerance 1e-4 -> 1e-6 (SCF convergence noise of the labels);
- ``shift``: rigid translation by a fraction of the real-space mesh spacing (egg-box noise);
- ``shift_tight``: both, i.e. mesh noise alone.

H in the atomic-orbital basis is invariant under a rigid translation, so any difference is
numerical noise that no model can learn: a floor for the H error of any model.
"""
import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from ..analysis.metrics import gamma_hk
from .runner import find_pseudos, run_siesta_into, system_label

__all__ = ["VARIANTS", "SHIFT_ANG", "noise_fdf", "run_label_noise", "gamma_mae_mev"]

SHIFT_ANG = np.array([0.031, 0.017, 0.009])
# variant: (rigid shift, DM.Tolerance)
VARIANTS: Dict[str, Tuple[bool, str]] = {
    "tight": (False, "1.d-6"),
    "shift": (True, "1.d-4"),
    "shift_tight": (True, "1.d-6"),
}


def noise_fdf(text: str, shift: bool, tolerance: str, shift_ang: np.ndarray = SHIFT_ANG) -> str:
    """fdf with every atom of ``AtomicCoordinatesAndAtomicSpecies`` moved by ``shift_ang``
    (Ang coordinates) if ``shift``, and ``DM.Tolerance`` set to ``tolerance`` (appended when
    the fdf does not set it)."""
    lines = text.splitlines()
    inside = False
    tolerance_set = False
    for i, line in enumerate(lines):
        if re.match(r"\s*%block AtomicCoordinatesAndAtomicSpecies", line, re.I):
            inside = True
            continue
        if re.match(r"\s*%endblock AtomicCoordinatesAndAtomicSpecies", line, re.I):
            inside = False
        elif inside and shift:
            parts = line.split()
            xyz = np.array([float(x) for x in parts[:3]]) + shift_ang
            lines[i] = " ".join(f"{x:.12f}" for x in xyz) + " " + " ".join(parts[3:])
        elif re.match(r"\s*DM\.Tolerance", line, re.I):
            lines[i] = f"DM.Tolerance           {tolerance}"
            tolerance_set = True
    if not tolerance_set:
        lines.append(f"DM.Tolerance           {tolerance}")
    return "\n".join(lines) + "\n"


def run_label_noise(
    reference_dir: Path,
    work_dir: Path,
    variants: Mapping[str, Tuple[bool, str]] = VARIANTS,
    siesta: Optional[str] = None,
) -> Dict[str, Path]:
    """Recompute the calculation of ``reference_dir`` once per variant, in
    ``work_dir / <variant>`` (finished runs are reused, failed ones never land there).
    Returns the TSHS of each variant. Coordinates must be in Ang for ``shift``."""
    reference_dir, work_dir = Path(reference_dir), Path(work_dir)
    label = system_label(reference_dir / "RUN.fdf")
    text = (reference_dir / "RUN.fdf").read_text()
    return {
        name: run_siesta_into(work_dir / name, noise_fdf(text, shift, tolerance), find_pseudos(reference_dir),
                              f"{label}.TSHS", siesta=siesta, nice=0)
        for name, (shift, tolerance) in variants.items()
    }


def gamma_mae_mev(a: Path, b: Path) -> float:
    """Gamma-point H MAE between two matrix files (meV)."""
    return 1e3 * float(np.mean(np.abs(gamma_hk(a) - gamma_hk(b))))
