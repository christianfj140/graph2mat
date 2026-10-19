import numpy as np
import pytest
import sisl
from scipy.sparse import csr_array

from graph2mat.core.data.spin_orbit import (
    SOC_TRANSPOSE_SIGN,
    siesta_to_soc_channels,
    soc_channels_to_siesta,
    soc_trs_residual,
)


@pytest.fixture
def block():
    """Synthetic on-site block, 3 s orbitals: h0 symmetric, g_a antisymmetric."""
    rng = np.random.default_rng(0)
    n = 3
    a = rng.normal(size=(4, n, n))
    h0 = a[0] + a[0].T
    gx, gy, gz = (m - m.T for m in a[1:])
    atom = sisl.Atom(1, [sisl.AtomicOrbital(f"s{i}", R=1.0) for i in range(n)])
    geom = sisl.Geometry([[0, 0, 0]], atom, lattice=sisl.Lattice(20.0))
    return geom, h0, gx, gy, gz


def test_channels_match_sisl_spinor_and_roundtrip(block):
    geom, h0, gx, gy, gz = block
    channels = np.stack([h0, gx, gy, gz], -1)  # (3, 3, 4)
    values = soc_channels_to_siesta(channels)  # (3, 3, 8)
    assert values.shape == (3, 3, 8)
    H = sisl.Hamiltonian.fromsp(
        geom, [csr_array(values[..., i]) for i in range(8)], spin=sisl.Spin("so")
    )
    s0 = np.eye(2)
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.diag([1.0, -1.0])
    spinor = np.kron(h0, s0) + 1j * (
        np.kron(gx, sx) + np.kron(gy, sy) + np.kron(gz, sz)
    )
    np.testing.assert_array_equal(H.Hk(format="array"), spinor)
    np.testing.assert_array_equal(siesta_to_soc_channels(values), channels)
    assert soc_trs_residual(values) == 0


def test_trs_violation_is_reported_not_repaired(block):
    _, h0, gx, gy, gz = block
    channels = np.stack([h0, gx, gy, gz], -1)
    ref = soc_channels_to_siesta(channels)
    values = ref.copy()
    values[..., 0] += 1e-3 * np.abs(channels).max()  # Re hz != 0
    with pytest.raises(ValueError, match="time-reversal"):
        siesta_to_soc_channels(values)
    values[..., 0] = ref[..., 0] + 1e-7 * np.abs(channels).max()
    siesta_to_soc_channels(values)  # SCF-like residual: accepted


def test_transpose_sign_and_leading_dims(block):
    _, h0, gx, gy, gz = block
    channels = np.stack([h0, gx, gy, gz], -1)
    np.testing.assert_array_equal(
        channels.swapaxes(0, 1), channels * SOC_TRANSPOSE_SIGN
    )
    batch = np.stack([channels, 2 * channels])
    np.testing.assert_array_equal(
        siesta_to_soc_channels(soc_channels_to_siesta(batch)), batch
    )
