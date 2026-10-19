"""Conversion between SIESTA's 8-component spin-orbit matrices and TRS channels.

SIESTA/sisl store the non-collinear spin-orbit Hamiltonian as 8 real values per
matrix element, in the order::

    c = [Re uu, Re dd, Re ud, Im ud, Im uu, Im dd, Re du, Im du]

With time-reversal symmetry (TRS) the spinor block is
``h0 * s0 + i * (gx * sx + gy * sy + gz * sz)`` with real ``h0, g_a``, so only
4 real channels ``[h0, gx, gy, gz]`` are independent. ``h0`` is symmetric and
``g`` antisymmetric on-site, hence ``g`` changes sign under transposition.

Pure NumPy: no torch dependency.
"""

import numpy as np

__all__ = [
    "SOC_TRS_RTOL",
    "SOC_TRANSPOSE_SIGN",
    "siesta_to_soc_channels",
    "soc_trs_residual",
    "soc_channels_to_siesta",
]

#: Maximum TRS residual, relative to the largest channel, accepted when loading.
SOC_TRS_RTOL = 1e-5

#: Sign of each channel ``[h0, gx, gy, gz]`` under matrix transposition.
SOC_TRANSPOSE_SIGN = np.array([1.0, -1.0, -1.0, -1.0])


def _check_last_dim(values, n, name):
    values = np.asarray(values)
    if values.shape[-1] != n:
        raise ValueError(f"{name} must have {n} components in the last axis, got {values.shape}.")
    return values


def _channels_and_residuals(values):
    c = [values[..., i] for i in range(8)]
    channels = np.stack(
        [(c[0] + c[1]) / 2, (c[3] + c[7]) / 2, (c[2] - c[6]) / 2, (c[4] - c[5]) / 2], -1
    )
    residuals = np.stack(
        [(c[4] + c[5]) / 2, (c[2] + c[6]) / 2, (c[7] - c[3]) / 2, (c[0] - c[1]) / 2], -1
    )
    return channels, residuals


def _relative_residual(channels, residuals):
    """``max|residuals| / max|channels|``; ``nan`` if any value is not finite."""
    if residuals.size == 0:
        return 0.0
    if not (np.isfinite(channels).all() and np.isfinite(residuals).all()):
        return float("nan")
    tiny = np.finfo(np.float64).tiny
    return float(np.abs(residuals).max() / max(np.abs(channels).max(), tiny))


def soc_trs_residual(values: np.ndarray) -> float:
    """Relative time-reversal residual of SIESTA spin-orbit values ``(..., 8)``.

    Returns ``max|residuals| / max(max|channels|, tiny)``, or ``nan`` if any
    value is not finite. The residuals are ``Im h0``, ``Re hx``, ``Re hy`` and
    ``Re hz``, which vanish under TRS.
    """
    return _relative_residual(*_channels_and_residuals(_check_last_dim(values, 8, "values")))


def siesta_to_soc_channels(values: np.ndarray) -> np.ndarray:
    """Convert SIESTA values ``(..., 8)`` into TRS channels ``(..., 4)``.

    Channels are ``[h0, gx, gy, gz]``. Raises ``ValueError`` if the values are
    not finite or if the relative TRS residual exceeds ``SOC_TRS_RTOL``; the
    residual is never symmetrized away silently.
    """
    channels, residuals = _channels_and_residuals(_check_last_dim(values, 8, "values"))
    rel = _relative_residual(channels, residuals)
    if np.isnan(rel):
        raise ValueError("Spin-orbit values must be finite (found nan or inf).")
    if rel > SOC_TRS_RTOL:
        raise ValueError(
            f"Spin-orbit values violate time-reversal symmetry: relative residual "
            f"{rel:.3e} > {SOC_TRS_RTOL:.0e}. Magnetic spin-orbit is not supported."
        )
    return channels


def soc_channels_to_siesta(channels: np.ndarray) -> np.ndarray:
    """Convert TRS channels ``(..., 4)`` into SIESTA values ``(..., 8)``."""
    channels = _check_last_dim(channels, 4, "channels")
    h0, gx, gy, gz = (channels[..., i] for i in range(4))
    return np.stack([h0, h0, gy, gx, gz, -gz, -gy, gx], -1)
