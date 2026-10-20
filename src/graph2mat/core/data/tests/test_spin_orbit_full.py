"""Policy ``spin_orbit_full``: the eight real spinor channels ``[a0, bx, by, bz, b0, ax, ay, az]``.

The spinor block is ``H = sum_mu sigma_mu (x) h^mu`` with ``h^mu = a^mu + i b^mu``. The oracles
below build it by hand from complex matrices and the Pauli matrices and read the eight SIESTA
values off the spinor entries (sisl's documented component order); nothing is derived from the
formulas under test.
"""
import numpy as np
import pytest
import sisl
from scipy.sparse import csr_array

from graph2mat import BasisMatrixData, MatrixDataProcessor, AtomicTableWithEdges
from graph2mat.core.data.configuration import OrbitalConfiguration
from graph2mat.core.data.sparse import csr_to_sisl_sparse_orbital
from graph2mat.core.data.spin_orbit import (
    SOC_FULL_TIME_REVERSAL_PARITY,
    SOC_FULL_TRANSPOSE_SIGN,
    attach_overlap,
    siesta_to_soc_channels,
    siesta_to_soc_full_channels,
    soc_channels_to_siesta,
    soc_full_channels_to_siesta,
)
from graph2mat.core.data.tests.test_spin_orbit import (
    _orbitals,
    _overlap_of,
    make_soc_system,
    soc_processor,
)

POLICY = "spin_orbit_full"
PAULI = [
    np.eye(2),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, -1j], [1j, 0]]),
    np.diag([1.0, -1.0]),
]


def full_processor(atoms, symmetric=False, n_components=8):
    return soc_processor(atoms, symmetric, policy=POLICY, n_components=n_components)


def spinor(h):
    """Hand-built spinor ``sum_mu kron(h^mu, sigma_mu)`` (index = 2 * orbital + spin)."""
    return sum(np.kron(h[mu], PAULI[mu]) for mu in range(4))


def spinor_to_sisl_values(H):
    """Eight SIESTA values ``(n, n, 8)`` read off the spinor entries ``H[2i+s, 2j+s']``."""
    uu, ud, du, dd = H[0::2, 0::2], H[0::2, 1::2], H[1::2, 0::2], H[1::2, 1::2]
    return np.stack(
        [uu.real, dd.real, ud.real, ud.imag, uu.imag, dd.imag, du.real, du.imag], -1
    )


def random_hermitian(rng, n, size=4):
    m = rng.normal(size=(size, n, n)) + 1j * rng.normal(size=(size, n, n))
    return (m + m.conj().transpose(0, 2, 1)) / 2


@pytest.fixture
def onsite():
    """One atom, 3 s orbitals, random complex Hermitian ``h^mu``, and its sisl Hamiltonian."""
    rng = np.random.default_rng(3)
    n = 3
    h = random_hermitian(rng, n)
    atom = sisl.Atom(1, [sisl.AtomicOrbital(f"s{i}", R=1.0) for i in range(n)])
    geom = sisl.Geometry([[0, 0, 0]], atom, lattice=sisl.Lattice(20.0))
    values = spinor_to_sisl_values(spinor(h))
    H = sisl.Hamiltonian.fromsp(
        geom, [csr_array(values[..., i]) for i in range(8)], spin=sisl.Spin("so")
    )
    return h, H, values


def sisl_values(H):
    return np.stack([H.tocsr(i).toarray() for i in range(8)], -1)


def test_constants():
    np.testing.assert_array_equal(SOC_FULL_TRANSPOSE_SIGN, [1, -1, -1, -1, -1, 1, 1, 1])
    np.testing.assert_array_equal(SOC_FULL_TIME_REVERSAL_PARITY, [1, 1, 1, 1, -1, -1, -1, -1])


def test_pauli_oracle_roundtrip(onsite):
    h, H, values = onsite
    # The hand-read values are what sisl turns into this spinor (fixes the convention).
    np.testing.assert_allclose(H.Hk(format="array"), spinor(h), atol=1e-15)
    np.testing.assert_array_equal(sisl_values(H), values)

    channels = siesta_to_soc_full_channels(sisl_values(H))
    a, b = h.real, h.imag
    expected = np.stack(
        [a[0], b[1], b[2], b[3], b[0], a[1], a[2], a[3]], -1
    )  # [a0, bx, by, bz, b0, ax, ay, az]
    np.testing.assert_allclose(channels, expected, atol=1e-15, rtol=0)
    np.testing.assert_allclose(soc_full_channels_to_siesta(channels), values, atol=1e-15, rtol=0)

    # Back through sisl: the reconstructed Hamiltonian has the same spinor.
    back = soc_full_channels_to_siesta(channels)
    H2 = sisl.Hamiltonian.fromsp(
        H.geometry, [csr_array(back[..., i]) for i in range(8)], spin=sisl.Spin("so")
    )
    np.testing.assert_allclose(H2.Hk(format="array"), spinor(h), atol=1e-15)


def test_oracle_blocks_are_not_trs(onsite):
    h, _, _ = onsite
    # Guards the oracle: the odd channels (b0, ax, ay, az) are populated.
    assert all(np.abs(x).max() > 0.1 for x in (h[0].imag, h[1].real, h[2].real, h[3].real))


def test_time_reversal_parity_and_transpose_sign_oracle(onsite):
    h, _, _ = onsite
    n = h.shape[1]
    # Theta = i sigma_y K on the spinor index: H -> sigma_y conj(H) sigma_y.
    sy = np.kron(np.eye(n), PAULI[2])
    h_t = sy @ spinor(h).conj() @ sy
    ch = siesta_to_soc_full_channels(spinor_to_sisl_values(spinor(h)))
    ch_t = siesta_to_soc_full_channels(spinor_to_sisl_values(h_t))
    np.testing.assert_allclose(ch_t, ch * SOC_FULL_TIME_REVERSAL_PARITY, atol=1e-14)

    # Hermitian on-site block: H = H^dagger  <=>  channels transpose with the sign.
    np.testing.assert_allclose(
        ch.swapaxes(0, 1), ch * SOC_FULL_TRANSPOSE_SIGN, atol=1e-14
    )


def test_trs_subset_matches_spin_orbit():
    rng = np.random.default_rng(5)
    chans = rng.normal(size=(4, 5, 8))
    chans[..., 4:] = 0  # b0 = a^k = 0
    values = soc_full_channels_to_siesta(chans)
    np.testing.assert_array_equal(values, soc_channels_to_siesta(chans[..., :4]))
    np.testing.assert_array_equal(
        siesta_to_soc_full_channels(values)[..., :4], siesta_to_soc_channels(values)
    )
    np.testing.assert_allclose(siesta_to_soc_full_channels(values)[..., 4:], 0, atol=1e-16)


def test_full_channels_do_not_check_trs_but_reject_non_finite_and_bad_shape():
    rng = np.random.default_rng(6)
    values = rng.normal(size=(4, 4, 8))  # grossly non-TRS
    siesta_to_soc_full_channels(values)
    for bad in (np.nan, np.inf, -np.inf):
        v = values.copy()
        v[0, 0, 4] = bad
        with pytest.raises(ValueError, match="finite"):
            siesta_to_soc_full_channels(v)
    with pytest.raises(ValueError, match="8"):
        siesta_to_soc_full_channels(values[..., :4])
    with pytest.raises(ValueError, match="8"):
        soc_full_channels_to_siesta(values[..., :4])
    batch = rng.normal(size=(2, 3, 3, 8))
    np.testing.assert_allclose(
        siesta_to_soc_full_channels(soc_full_channels_to_siesta(batch)), batch, atol=1e-15
    )


# ---------------------------------------------------------------------------
# Periodic non-TRS system
# ---------------------------------------------------------------------------

def make_magnetic_system(seed=0, orthogonal=False):
    """``make_soc_system`` plus a non-TRS part built from complex spinor blocks.

    The extra part has ``h^0 = i b0`` and ``h^k = a^k`` (the channels ``spin_orbit`` rejects),
    with periodic Hermiticity ``h(-R) = h(R)^dagger`` imposed on the complex blocks.
    """
    H, atoms, blocks, overlap = make_soc_system(seed, orthogonal=orthogonal)
    geometry = H.geometry
    no = geometry.no
    rng = np.random.default_rng(100 + seed)

    def odd_part(m):  # (4, no, no) complex: i b0 and real a^k
        return np.stack([1j * m[0].imag, m[1].real, m[2].real, m[3].real])

    on = odd_part(random_hermitian(rng, no))
    im = odd_part(rng.normal(size=(4, no, no)) + 1j * rng.normal(size=(4, no, no)))
    per_image = {0: on, 1: im, -1: im.conj().transpose(0, 2, 1)}
    per_image = {r: spinor_to_sisl_values(spinor(h)) for r, h in per_image.items()}

    def dense(i):
        return np.concatenate([per_image[int(off[0])][..., i] for off in geometry.sc_off], axis=1)

    comps = [csr_array(H.tocsr(i).toarray() + dense(i)) for i in range(8)]
    S = None if orthogonal else csr_array(H.tocsr(H.S_idx).toarray())
    H_full = sisl.Hamiltonian.fromsp(geometry, comps, S=S, spin=sisl.Spin("so"))
    return H_full, H, atoms


def test_non_trs_system_differs_from_trs_system():
    H_full, H_trs, _ = make_magnetic_system()
    assert np.abs(sisl_values(H_full) - sisl_values(H_trs)).max() > 0.1


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("orthogonal", [False, True])
def test_full_policy_roundtrips_non_trs_system(symmetric, orthogonal):
    H_full, _, atoms = make_magnetic_system(orthogonal=orthogonal)
    processor = full_processor(atoms, symmetric)
    data = BasisMatrixData.new(H_full, processor)
    assert data.point_labels.shape[1] == data.edge_labels.shape[1] == 8
    assert np.abs(data.edge_labels[:, 4:]).max() > 0.1  # odd channels loaded, not dropped

    H_orth = processor.matrix_from_data(data, out_format="sisl_H", threshold=None)
    assert H_orth.orthogonal and H_orth.spin.is_spinorbit
    for i in range(8):
        np.testing.assert_allclose(H_orth.tocsr(i).toarray(), H_full.tocsr(i).toarray(), atol=1e-12)

    H = H_orth
    if not orthogonal:
        H = attach_overlap(H_orth, _overlap_of(H_full))
    for k in (np.zeros(3), np.array([0.17, 0, 0]), np.array([-0.31, 0, 0])):
        Hk = H.Hk(k, format="array")
        np.testing.assert_allclose(Hk, Hk.conj().T, atol=1e-10)
        np.testing.assert_allclose(Hk, H_full.Hk(k, format="array"), atol=1e-10)
        np.testing.assert_allclose(H.eigh(k), H_full.eigh(k), atol=1e-10)


def test_spin_orbit_policy_still_rejects_non_trs_system():
    H_full, _, atoms = make_magnetic_system()
    with pytest.raises(ValueError, match="time-reversal"):
        BasisMatrixData.new(H_full, soc_processor(atoms))


def test_full_policy_matches_spin_orbit_on_trs_system():
    H, atoms, blocks, _ = make_soc_system()
    so = BasisMatrixData.new(H, soc_processor(atoms))
    full = BasisMatrixData.new(H, full_processor(atoms))
    np.testing.assert_array_equal(full.point_labels[:, :4], so.point_labels)
    np.testing.assert_array_equal(full.edge_labels[:, :4], so.edge_labels)
    np.testing.assert_allclose(full.point_labels[:, 4:], 0, atol=1e-14)
    np.testing.assert_allclose(full.edge_labels[:, 4:], 0, atol=1e-14)
    # And the reconstruction is the same Hamiltonian.
    rec = full_processor(atoms).matrix_from_data(full, out_format="sisl_H", threshold=None)
    for i in range(8):
        np.testing.assert_allclose(rec.tocsr(i).toarray(), H.tocsr(i).toarray(), atol=1e-12)


def test_transposed_blocks_carry_the_sign_in_labels():
    H_full, _, atoms = make_magnetic_system()
    config = OrbitalConfiguration.new(H_full, matrix_component_policy=POLICY)
    assert config.matrix._n_components == 8
    sign = np.array([1, -1, -1, -1, -1, 1, 1, 1])
    lattice = H_full.lattice
    got = config.matrix.block_dict
    checked = 0
    for (i, j, isc), block in got.items():
        opposite = int(lattice.sc_index(-lattice.sc_off[isc]))
        np.testing.assert_allclose(
            got[(j, i, opposite)], block.swapaxes(0, 1) * sign, atol=1e-12, rtol=0
        )
        checked += 1
    assert checked > 8


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spin", ["unpolarized", "polarized", "non-colinear"])
def test_full_policy_requires_spin_orbit_matrix(spin):
    rng = np.random.default_rng(1)
    atoms = [sisl.Atom("H", orbitals=_orbitals((0,)))]
    geom = sisl.Geometry([[0.0, 0.0, 0.0]], atoms=atoms, lattice=[20.0] * 3)
    n = {"unpolarized": 1, "polarized": 2, "non-colinear": 4}[spin]
    comps = [csr_array(np.array([[rng.normal()]])) for _ in range(n)]
    H = sisl.Hamiltonian.fromsp(geom, comps[0] if n == 1 else comps, spin=spin)
    with pytest.raises(ValueError, match="spin_orbit_full"):
        BasisMatrixData.new(H, full_processor(atoms))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_full_policy_rejects_non_finite_values(bad):
    H_full, _, atoms = make_magnetic_system()
    H_full._csr.data[0, 4] = bad
    with pytest.raises(ValueError, match="finite"):
        BasisMatrixData.new(H_full, full_processor(atoms))


def test_full_policy_processor_validation():
    atoms = [sisl.Atom("H", orbitals=_orbitals((0,)))]
    table = AtomicTableWithEdges(atoms)
    with pytest.raises(ValueError, match="hamiltonian"):
        MatrixDataProcessor(basis_table=table, out_matrix="density_matrix", matrix_component_policy=POLICY)
    for n in (2, 4):
        with pytest.raises(ValueError, match="n_matrix_components"):
            MatrixDataProcessor(
                basis_table=table, out_matrix="hamiltonian", n_matrix_components=n, matrix_component_policy=POLICY
            )
    for n in (1, 8):
        MatrixDataProcessor(
            basis_table=table, out_matrix="hamiltonian", n_matrix_components=n, matrix_component_policy=POLICY
        )
    # spin_orbit keeps its own count.
    with pytest.raises(ValueError, match="n_matrix_components"):
        MatrixDataProcessor(
            basis_table=table, out_matrix="hamiltonian", n_matrix_components=8, matrix_component_policy="spin_orbit"
        )


def test_full_policy_reconstructs_only_to_sisl():
    H_full, _, atoms = make_magnetic_system()
    processor = full_processor(atoms)
    data = BasisMatrixData.new(H_full, processor)
    with pytest.raises(ValueError, match="sisl"):
        processor.matrix_from_data(data, out_format="scipy_csr")
    with pytest.raises(ValueError, match="sisl"):
        processor.labels_to("scipy_csr", data)


@pytest.mark.parametrize("n", [1, 2, 4, 7, 9])
def test_full_policy_needs_eight_channels(n):
    H, _, _, _ = make_soc_system(orthogonal=True)
    csr = [csr_array(H.tocsr(0)) for _ in range(n)]
    with pytest.raises(ValueError, match="8"):
        csr_to_sisl_sparse_orbital(
            csr if n > 1 else csr[0],
            geometry=H.geometry,
            sp_class=sisl.Hamiltonian,
            matrix_component_policy=POLICY,
        )


def test_full_policy_needs_hamiltonian_class():
    H, _, _, _ = make_soc_system(orthogonal=True)
    csr = [csr_array(H.tocsr(0)) for _ in range(8)]
    with pytest.raises(ValueError, match="Hamiltonian"):
        csr_to_sisl_sparse_orbital(
            csr, geometry=H.geometry, sp_class=sisl.SparseOrbital, matrix_component_policy=POLICY
        )
