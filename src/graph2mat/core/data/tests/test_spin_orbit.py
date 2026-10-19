import numpy as np
import pytest
import sisl
from scipy.sparse import csr_array

from graph2mat import AtomicTableWithEdges, BasisMatrixData, MatrixDataProcessor
from graph2mat.core.data.configuration import OrbitalConfiguration
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


# ---------------------------------------------------------------------------
# Residual formulas: each of the four TRS-odd residuals perturbed separately
# ---------------------------------------------------------------------------

# (name, {component: sign}, index of the residual it excites)
_RESIDUAL_PERTURBATIONS = [
    ("Im h0", {4: 1, 5: 1}, 0),
    ("Re hx", {2: 1, 6: 1}, 1),
    ("Re hy", {7: 1, 3: -1}, 2),
    ("Re hz", {0: 1, 1: -1}, 3),
]


@pytest.mark.parametrize("name,pert,k", _RESIDUAL_PERTURBATIONS, ids=[p[0] for p in _RESIDUAL_PERTURBATIONS])
def test_each_trs_residual_is_pinned(block, name, pert, k):
    _, h0, gx, gy, gz = block
    channels = np.stack([h0, gx, gy, gz], -1)
    ref = soc_channels_to_siesta(channels)
    scale = np.abs(channels).max()

    def perturbed(rel):
        values = ref.copy()
        for comp, sign in pert.items():
            values[..., comp] += sign * rel * scale
        return values

    # Perturbation is orthogonal to the channels: only the residual moves.
    big = perturbed(1e-3)
    np.testing.assert_allclose(soc_trs_residual(big), 1e-3, rtol=1e-9)
    with pytest.raises(ValueError, match="time-reversal"):
        siesta_to_soc_channels(big)
    small = perturbed(1e-7)
    np.testing.assert_allclose(soc_trs_residual(small), 1e-7, rtol=1e-7)
    np.testing.assert_allclose(siesta_to_soc_channels(small), channels, atol=1e-14)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("comp", [0, 4])
def test_non_finite_values_raise(block, bad, comp):
    _, h0, gx, gy, gz = block
    values = soc_channels_to_siesta(np.stack([h0, gx, gy, gz], -1))
    values[0, 0, comp] = bad
    assert np.isnan(soc_trs_residual(values))
    with pytest.raises(ValueError, match="finite"):
        siesta_to_soc_channels(values)


# ---------------------------------------------------------------------------
# Label loading (policy "spin_orbit")
# ---------------------------------------------------------------------------

def _orbitals(ls):
    return [sisl.AtomicOrbital(n=l + 1, l=l, m=m, R=2.8) for l in ls for m in range(-l, l + 1)]


def make_soc_system(seed=0, orthogonal=False):
    """Periodic 2-atom (s+p, s) system with TRS spin-orbit blocks.

    Cell is short along x (nsc=[3,1,1]) so atoms interact with their own images.
    Blocks are built on dense supercell arrays in channel space, with
    ``h0_ij(R) = h0_ji(-R)^T`` and ``g_ij(R) = -g_ji(-R)^T`` (and ``S`` symmetric),
    then converted with ``soc_channels_to_siesta``.

    Returns ``(H, atoms, blocks, overlap)``: the sisl SO Hamiltonian, the atoms,
    ``blocks[R] -> (no, no, 4)`` TRS channels and ``overlap[R] -> (no, no)``
    (``None`` if ``orthogonal``). ``R`` is the supercell shift ``-1, 0, 1``.
    """
    atoms = [sisl.Atom("H", orbitals=_orbitals((0, 1))), sisl.Atom("O", orbitals=_orbitals((0,)))]
    geometry = sisl.Geometry([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], atoms=atoms, lattice=[5.0, 20.0, 20.0])
    geometry.set_nsc([3, 1, 1])
    no = geometry.no
    rng = np.random.default_rng(seed)

    def transpose(c):
        return c.swapaxes(0, 1) * SOC_TRANSPOSE_SIGN

    onsite = rng.normal(size=(no, no, 4))
    image = rng.normal(size=(no, no, 4))
    blocks = {0: (onsite + transpose(onsite)) / 2, 1: image, -1: transpose(image)}
    s_on, s_im = rng.normal(size=(no, no)), rng.normal(size=(no, no))
    overlap = {0: np.eye(no) + 0.01 * (s_on + s_on.T), 1: 0.01 * s_im, -1: 0.01 * s_im.T}

    def dense(per_image):
        # sc_off ordering is sisl's: concatenate in that order along columns.
        return np.concatenate([per_image[int(off[0])] for off in geometry.sc_off], axis=1)

    values = dense({r: soc_channels_to_siesta(b) for r, b in blocks.items()})
    comps = [csr_array(values[..., i]) for i in range(8)]
    S = None if orthogonal else csr_array(dense(overlap))
    H = sisl.Hamiltonian.fromsp(geometry, comps, S=S, spin=sisl.Spin("so"))
    return H, atoms, blocks, None if orthogonal else overlap


def soc_processor(atoms, symmetric=False, policy="spin_orbit", n_components=4):
    return MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(atoms),
        out_matrix="hamiltonian",
        symmetric_matrix=symmetric,
        sub_point_matrix=False,
        n_matrix_components=n_components,
        matrix_component_policy=policy,
    )


def _expected_blocks(blocks, H):
    """Channel blocks per (atom_i, atom_j, isc) from the fixture's dense blocks."""
    geom = H.geometry
    first = geom.firsto
    out = {}
    for i in range(geom.na):
        for j in range(geom.na):
            for isc, off in enumerate(H.lattice.sc_off):
                out[(i, j, isc)] = blocks[int(off[0])][first[i] : first[i + 1], first[j] : first[j + 1]]
    return out


@pytest.mark.parametrize("symmetric", [False, True])
def test_spin_orbit_labels_exclude_overlap(symmetric):
    H, atoms, blocks, _ = make_soc_system()
    assert H._csr.data.shape[1] == 9 and H.S_idx == 8  # non-orthogonal: S is the 9th component
    data = BasisMatrixData.new(H, soc_processor(atoms, symmetric))
    assert data.point_labels.shape[1] == data.edge_labels.shape[1] == 4

    # Matrix stored by the configuration: 4 TRS channels per block, never S.
    config = OrbitalConfiguration.new(H, matrix_component_policy="spin_orbit")
    expected = _expected_blocks(blocks, H)
    assert config.matrix._n_components == 4
    got = config.matrix.block_dict
    assert got, "no blocks loaded"
    for key, block in got.items():
        np.testing.assert_allclose(block, expected[key], atol=1e-12, rtol=0)
    # Every block of the fixture is populated (dense supercell), including self-images.
    assert {k for k in expected if k[0] == k[1] and k[2] != 0} <= set(got)

    # Point labels are exactly the on-site channel blocks, flattened, atom by atom.
    onsite = np.concatenate([expected[(i, i, 0)].reshape(-1, 4) for i in range(H.geometry.na)])
    np.testing.assert_allclose(data.point_labels, onsite, atol=1e-12, rtol=0)
    if not symmetric:
        # Edge labels are the full (i, j, isc) channel blocks in edge order.
        edges = np.concatenate(
            [expected[(i, j, isc)].reshape(-1, 4) for (i, j), isc in zip(data.edge_index.T.tolist(), data.neigh_isc)]
        )
        np.testing.assert_allclose(data.edge_labels, edges, atol=1e-12, rtol=0)


def test_spin_orbit_orthogonal_has_same_labels():
    H, atoms, blocks, _ = make_soc_system(orthogonal=True)
    assert H.orthogonal and H._csr.data.shape[1] == 8
    config = OrbitalConfiguration.new(H, matrix_component_policy="spin_orbit")
    for key, block in config.matrix.block_dict.items():
        np.testing.assert_allclose(block, _expected_blocks(blocks, H)[key], atol=1e-12, rtol=0)


def _perturbed(H, rel, comp=0):
    """Copy of ``H`` with a Re hz-type residual ``rel`` (relative to the largest channel)."""
    H2 = H.copy()
    data = H2._csr.data
    h_scale = np.abs(data[:, :8]).max()
    data[:, comp] += rel * h_scale
    return H2


def test_spin_orbit_trs_tolerance():
    H, atoms, _, _ = make_soc_system()
    processor = soc_processor(atoms)
    BasisMatrixData.new(_perturbed(H, 1e-7), processor)  # SCF-like residual: loads
    with pytest.raises(ValueError, match="time-reversal"):
        BasisMatrixData.new(_perturbed(H, 1e-3), processor)


@pytest.mark.parametrize("spin", ["unpolarized", "polarized", "non-colinear"])
def test_spin_orbit_policy_requires_spin_orbit_matrix(spin):
    rng = np.random.default_rng(1)
    atoms = [sisl.Atom("H", orbitals=_orbitals((0,)))]
    geom = sisl.Geometry([[0.0, 0.0, 0.0]], atoms=atoms, lattice=[20.0] * 3)
    n = {"unpolarized": 1, "polarized": 2, "non-colinear": 4}[spin]
    comps = [csr_array(np.array([[rng.normal()]])) for _ in range(n)]
    H = sisl.Hamiltonian.fromsp(geom, comps[0] if n == 1 else comps, spin=spin)
    with pytest.raises(ValueError, match="spin-orbit"):
        BasisMatrixData.new(H, soc_processor(atoms))


def test_spin_orbit_processor_validation():
    atoms = [sisl.Atom("H", orbitals=_orbitals((0,)))]
    table = AtomicTableWithEdges(atoms)
    with pytest.raises(ValueError, match="hamiltonian"):
        MatrixDataProcessor(basis_table=table, out_matrix="density_matrix", matrix_component_policy="spin_orbit")
    with pytest.raises(ValueError, match="n_matrix_components"):
        MatrixDataProcessor(
            basis_table=table, out_matrix="hamiltonian", n_matrix_components=2, matrix_component_policy="spin_orbit"
        )
    for n in (1, 4):
        MatrixDataProcessor(
            basis_table=table, out_matrix="hamiltonian", n_matrix_components=n, matrix_component_policy="spin_orbit"
        )
