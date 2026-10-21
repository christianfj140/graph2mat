"""Full spin-orbit readout block: the eight real spinor channels ``[a0, bx, by, bz, b0, ax, ay, az]``.

The spinor block is ``H = sum_mu sigma_mu (x) h^mu`` with ``h^mu = a^mu + i b^mu``. Hermiticity
is checked on spinors built by hand from the channels with the Pauli matrices, and equivariance
against Wigner D matrices of the orbitals and the Cartesian rotation of the axial vector channels.
"""
import numpy as np
import pytest
import sisl
import torch
from e3nn import o3

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor, OrbitalConfiguration, PointBasis
from graph2mat.bindings.e3nn import E3nnGraph2Mat, E3nnSpinOrbitFullMatrixBlock
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.core.data.basis import get_change_of_basis
from graph2mat.core.data.spin_orbit import siesta_to_soc_full_channels

PAULI = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]
VECTOR = [slice(1, 4), slice(5, 8)]  # (bx, by, bz) and (ax, ay, az): Cartesian axial vectors


@pytest.fixture(autouse=True)
def float64():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)


class _Coefficients(torch.nn.Module):
    """Operation returning the coefficients it is given."""

    def __init__(self, irreps_out):
        super().__init__()
        self.irreps_out = o3.Irreps(irreps_out)

    def forward(self, coefficients):
        # Edge blocks receive (forward, reverse) pairs: the forward direction's coefficients.
        return coefficients[0] if isinstance(coefficients, tuple) else coefficients


def _block(i_basis, j_basis=None, symmetry="ij=ji", **kwargs):
    return E3nnSpinOrbitFullMatrixBlock(
        i_basis=i_basis, j_basis=j_basis or i_basis, symmetry=symmetry, operation_cls=_Coefficients,
        irreps={"node_feats_irreps": o3.Irreps("0e+1o")}, n_matrix_components=8, **kwargs,
    )


def spinor(channels):
    """``sum_mu kron(a^mu + i b^mu, sigma_mu)`` from channels ``(n_i, n_j, 8)``."""
    c = np.asarray(channels)
    a = [c[..., 0], c[..., 5], c[..., 6], c[..., 7]]
    b = [c[..., 4], c[..., 1], c[..., 2], c[..., 3]]
    return sum(np.kron(a[mu] + 1j * b[mu], PAULI[mu]) for mu in range(4))


@pytest.mark.parametrize("basis", ["0e", "2x0e+1o", "1o+2e"], ids=["s", "sp", "pd"])
def test_onsite_blocks_are_hermitian_spinors(basis):
    torch.manual_seed(0)
    block = _block(PointBasis("A", R=3, basis=basis, basis_convention="siesta_spherical"))
    B = block(coefficients=torch.randn(3, block._irreps_out.dim)).numpy()
    n = block.i_irreps.dim
    assert B.shape == (3, n, n, 8)
    for b in B:
        H = spinor(b)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)
    # Every channel is free, except the antisymmetric b and b0 of a single orbital (H = a0 + a.sigma).
    nonzero = np.abs(B).max(axis=(0, 1, 2)) > 1e-3
    assert nonzero.all() if n > 1 else nonzero.tolist() == [True] + [False] * 4 + [True] * 3


def test_homonuclear_edge_pair_is_hermitian_conjugate():
    # H_ji = H_ij^dagger: the block of the reversed edge is the conjugate transpose spinor.
    torch.manual_seed(0)
    block = _block(PointBasis("A", R=3, basis="2x0e+1o", basis_convention="siesta_spherical"),
                   symmetry="ij", symm_transpose=True)
    a, b = torch.randn(2, 3, block._irreps_out.dim)
    ab, ba = block(coefficients=(a, b)).numpy(), block(coefficients=(b, a)).numpy()
    for x, y in zip(ab, ba):
        np.testing.assert_allclose(spinor(y), spinor(x).conj().T, atol=1e-12)
        assert not np.allclose(spinor(x), spinor(x).conj().T, atol=1e-3)


def _rotations():
    """Six random proper rotations and four improper ones (three -R and the inversion)."""
    rng = np.random.default_rng(0)
    proper = []
    for _ in range(6):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))
        proper.append(q if np.linalg.det(q) > 0 else -q)
    return proper + [-q for q in proper[:3]] + [-np.eye(3)]


@pytest.mark.parametrize("convention", ["cartesian", "siesta_spherical"])
@pytest.mark.parametrize("i_basis,j_basis,symmetry", [
    ("0e", "0e", "ij=ji"),
    ("2x0e+1o", "2x0e+1o", "ij=ji"),
    ("1o+2e", "1o+2e", "ij=ji"),
    ("1o+2e", "2x0e+1o", "ij"),
], ids=["s-onsite", "sp-onsite", "pd-onsite", "pd-sp-edge"])
def test_block_equivariance(i_basis, j_basis, symmetry, convention):
    """Rotating the coefficients with D(Q) of the output irreps rotates the block: each channel as
    ``D_i X D_j^T`` and the vector channels also as Cartesian axial vectors ``det(R) R v``."""
    i_basis = PointBasis("A", R=3, basis=i_basis, basis_convention=convention)
    j_basis = PointBasis("B", R=3, basis=j_basis, basis_convention=convention)
    block = _block(i_basis, j_basis, symmetry=symmetry)
    torch.manual_seed(0)
    c = torch.randn(4, block._irreps_out.dim)
    B = block(coefficients=c).numpy()
    assert np.abs(B).max() > 1e-3
    cob, cob_inv = get_change_of_basis("cartesian", convention)
    for Q in _rotations():
        Qt = torch.tensor(Q)
        rotated = block(coefficients=c @ block._irreps_out.D_from_matrix(Qt).T).numpy()
        Di, Dj = (b.e3nn_irreps.D_from_matrix(Qt).numpy() for b in (i_basis, j_basis))
        expected = np.einsum("ix,nxyc,jy->nijc", Di, B, Dj)
        R = cob_inv @ Q @ cob  # the data-frame rotation in Cartesian axes
        for v in VECTOR:
            expected[..., v] = expected[..., v] @ (np.linalg.det(R) * R).T
        np.testing.assert_allclose(rotated, expected, atol=1e-10)


def test_s_only_species_irreps():
    s = PointBasis("A", R=3, basis="0e", basis_convention="cartesian")
    onsite, edge = _block(s), _block(s, symmetry="ij")
    assert onsite._b0_irreps_out.dim == 0 and onsite._a_irreps_out == o3.Irreps("1x1e")
    assert edge._b0_irreps_out == o3.Irreps("1x0e") and edge._a_irreps_out == o3.Irreps("1x1e")
    assert edge._irreps_out == o3.Irreps("1x0e+1x1e+1x0e+1x1e")


def test_rejected_options():
    basis = PointBasis("A", R=3, basis="0e+1o", basis_convention="cartesian")
    for n in (1, 4):
        with pytest.raises(ValueError, match="n_matrix_components=8"):
            E3nnSpinOrbitFullMatrixBlock(i_basis=basis, j_basis=basis, symmetry="ij", operation_cls=_Coefficients,
                                         irreps={"node_feats_irreps": o3.Irreps("0e")}, n_matrix_components=n)
    with pytest.raises(ValueError, match="output_scaleshift"):
        _block(basis, output_scaleshift=True)
    with pytest.raises(ValueError, match="fuse_homonuclear"):
        _block(basis, fuse_homonuclear=True)
    block = _block(basis)
    for method in (block.forward_coefficients, block.forward_with_coefficients):
        with pytest.raises(NotImplementedError, match="coefficient outputs"):
            method(coefficients=torch.zeros(1, block._irreps_out.dim))


def _shell(l):
    return [sisl.AtomicOrbital(n=l + 1, l=l, m=m, R=2.1) for m in range(-l, l + 1)]


def test_graph2mat_reconstructed_hk_is_hermitian():
    """Periodic two-species chain, symmetric_matrix=True: the predicted blocks, reconstructed into a
    sisl spin-orbit Hamiltonian by the spin_orbit_full processor, give Hermitian H(k), with
    non-zero time-reversal-odd channels."""
    atoms = [sisl.Atom("H", orbitals=_shell(0) + _shell(1)), sisl.Atom("O", orbitals=_shell(0) + _shell(2))]
    geometry = sisl.Geometry([[0.0, 0.0, 0.0], [1.3, 0.4, 0.2]], atoms=atoms, lattice=[3.0, 20.0, 20.0])
    geometry.set_nsc([3, 1, 1])
    table = AtomicTableWithEdges(atoms)
    processor = MatrixDataProcessor(basis_table=table, out_matrix="hamiltonian", symmetric_matrix=True,
                                    sub_point_matrix=False, n_matrix_components=8,
                                    matrix_component_policy="spin_orbit_full")
    data = TorchBasisMatrixData.from_config(OrbitalConfiguration.from_geometry(geometry), data_processor=processor)
    torch.manual_seed(0)
    g2m = E3nnGraph2Mat(preprocessing_edges=None, unique_basis=table, symmetric=True,
                        irreps={"node_feats_irreps": o3.Irreps("0e+1o+1e")},
                        matrix_block_cls=E3nnSpinOrbitFullMatrixBlock, n_matrix_components=8)
    x = data.positions
    feats = torch.cat([data.point_types.reshape(-1, 1) + 1.0, x, torch.linalg.cross(x, x.mean(0).expand_as(x) + 1.0)], 1)
    with torch.no_grad():
        nodes, edges = g2m(data, node_feats=feats)
    assert nodes.shape[1] == edges.shape[1] == 8
    H = processor.matrix_from_data(data, {"node_labels": nodes, "edge_labels": edges}, out_format="sisl_H")
    channels = siesta_to_soc_full_channels(np.stack([H.tocsr(i).toarray() for i in range(8)], -1))
    assert np.abs(channels[..., 4:]).max() > 1e-3
    for k in np.random.default_rng(0).uniform(-0.5, 0.5, size=(4, 3)):
        Hk = H.Hk(k, format="array")
        assert np.abs(Hk).max() > 1e-3
        np.testing.assert_allclose(Hk, Hk.conj().T, atol=1e-10)
