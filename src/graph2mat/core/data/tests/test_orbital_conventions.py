"""Roadmap 3.10: orbital conventions against independent real spherical harmonics.

``basis_convention`` is a rotation of the axes in which e3nn's harmonics are evaluated.
SIESTA's harmonics (sisl's own implementation) must come out identical for every l.
Quantum ESPRESSO interleaves m (0, +1, -1, +2, -2, ...): a rotation reproduces it only
for l <= 1, so ``qe_spherical`` must refuse higher shells instead of mixing orbitals.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
o3 = pytest.importorskip("e3nn.o3")

from graph2mat import PointBasis
from graph2mat.core.data.basis import get_change_of_basis


def _directions():
    v = np.random.default_rng(0).normal(size=(200, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def _e3nn(convention, l, v):
    cob, _ = get_change_of_basis("cartesian", convention)
    return o3.spherical_harmonics(l, torch.tensor(v @ cob.T), normalize=True).numpy()


@pytest.mark.parametrize("l", [1, 2, 3])
def test_siesta_spherical_is_siesta_harmonics(l):
    from sisl._core.orbital import _rspherical_harm

    v = _directions()
    theta = np.arctan2(v[:, 1], v[:, 0])
    ref = np.stack([_rspherical_harm(l=l, m=m, theta=theta, cos_phi=v[:, 2]) for m in range(-l, l + 1)], 1)
    np.testing.assert_allclose(_e3nn("siesta_spherical", l, v), ref, atol=1e-12)


def test_qe_spherical_p_shell_is_quantum_espresso_order():
    # QE ylmr2: m = 0, +1, -1 with the (-1)^m phase: z, -x, -y (up to normalization).
    v = _directions()
    x, y, z = v.T
    np.testing.assert_allclose(_e3nn("qe_spherical", 1, v), np.sqrt(3 / (4 * np.pi)) * np.stack([z, -x, -y], 1), atol=1e-12)


def test_qe_spherical_rejects_shells_above_p():
    PointBasis("A", R=2, basis="2x0e + 1x1o", basis_convention="qe_spherical")
    with pytest.raises(ValueError, match="qe_spherical"):
        PointBasis("A", R=2, basis="1x0e + 1x1o + 1x2e", basis_convention="qe_spherical")
