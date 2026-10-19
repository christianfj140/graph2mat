"""Spin-orbit (TRS) readout block: h0 + axial-vector g channels."""
import copy
import warnings

import numpy as np
import pytest
import torch
from e3nn import o3

from graph2mat import (
    BasisConfiguration,
    BasisTableWithEdges,
    MatrixDataProcessor,
    PointBasis,
    conversions,
)
from graph2mat.bindings.e3nn import E3nnGraph2Mat, E3nnSpinOrbitMatrixBlock
from graph2mat.core.data.basis import get_change_of_basis

PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


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


def _block(basis, symmetry="ij=ji", **kwargs):
    return E3nnSpinOrbitMatrixBlock(
        i_basis=basis, j_basis=basis, symmetry=symmetry, operation_cls=_Coefficients,
        irreps={"node_feats_irreps": o3.Irreps("0e+1o")}, n_matrix_components=4, **kwargs,
    )


def test_onsite_symmetry_by_construction():
    torch.manual_seed(0)
    block = _block(PointBasis("A", R=3, basis="2x0e+1o", basis_convention="cartesian"))
    B = block(coefficients=torch.randn(5, block._irreps_out.dim))
    assert B.shape == (5, 5, 5, 4)
    assert torch.allclose(B[..., 0], B[..., 0].transpose(1, 2))
    assert torch.allclose(B[..., 1:], -B[..., 1:].transpose(1, 2))
    assert B[..., 1:].abs().max() > 1e-3


def test_homonuclear_edge_signed_transpose():
    torch.manual_seed(0)
    block = _block(PointBasis("A", R=3, basis="2x0e+1o", basis_convention="cartesian"),
                   symmetry="ij", symm_transpose=True)
    a, b = torch.randn(2, 3, block._irreps_out.dim)
    ab, ba = block(coefficients=(a, b)), block(coefficients=(b, a))
    assert torch.allclose(ba[..., 0], ab[..., 0].transpose(1, 2))
    assert torch.allclose(ba[..., 1:], -ab[..., 1:].transpose(1, 2))
    assert not torch.allclose(ab[..., 1:], -ab[..., 1:].transpose(1, 2))


@pytest.mark.parametrize("convention", ["cartesian", "siesta_spherical"])
def test_ls_coupling_spectrum(convention):
    block = _block(PointBasis("A", R=3, basis="1o", basis_convention=convention))
    c = torch.zeros(1, block._irreps_out.dim)
    offset = block._single_irreps_out.dim
    (scalar,) = [s for (mul, ir), s in zip(block._g_irreps_out, block._g_irreps_out.slices()) if ir == o3.Irrep("0e")]
    c[0, offset + scalar.start] = 0.7
    B = block(coefficients=c)[0].numpy()
    h0, g = B[..., 0], np.moveaxis(B[..., 1:], -1, 0)
    spinor = np.kron(h0, np.eye(2)) + 1j * sum(np.kron(g[a], PAULI[a]) for a in range(3))
    e = np.sort(np.linalg.eigvalsh(spinor))
    # lambda L.S on p: a doublet (j=1/2) at -2a and a quartet (j=3/2) at +a, or its mirror.
    if np.allclose(e[:4], e[0]):  # mirror: quartet below the doublet
        e = np.sort(-e)
    assert abs(e[2]) > 1e-3
    assert np.allclose(e[:2], e[0]) and np.allclose(e[2:], e[2]) and np.isclose(e[0], -2 * e[2])


def test_rejected_options():
    basis = PointBasis("A", R=3, basis="0e+1o", basis_convention="cartesian")
    with pytest.raises(ValueError, match="n_matrix_components"):
        E3nnSpinOrbitMatrixBlock(i_basis=basis, j_basis=basis, symmetry="ij", operation_cls=_Coefficients,
                                 irreps={"node_feats_irreps": o3.Irreps("0e")})
    with pytest.raises(ValueError, match="output_scaleshift"):
        _block(basis, output_scaleshift=True)
    with pytest.raises(ValueError, match="fuse_homonuclear"):
        _block(basis, fuse_homonuclear=True)
    block = _block(basis)
    with pytest.raises(NotImplementedError, match="coefficient outputs"):
        block.forward_coefficients(coefficients=torch.zeros(1, block._irreps_out.dim))
    with pytest.raises(NotImplementedError, match="coefficient outputs"):
        block.forward_with_coefficients(coefficients=torch.zeros(1, block._irreps_out.dim))


def _model_and_data(basis, point_types, positions, symmetric):
    table = BasisTableWithEdges(basis)
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=symmetric, sub_point_matrix=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # unreachable 3e irreps of p x p x 1e from 0e+1o features
        g2m = E3nnGraph2Mat(
            preprocessing_edges=None, unique_basis=table,
            irreps={"node_feats_irreps": o3.Irreps("0e + 1o")}, symmetric=symmetric,
            matrix_block_cls=E3nnSpinOrbitMatrixBlock, n_matrix_components=4,
        )
    config = BasisConfiguration(
        point_types=point_types, positions=np.array(positions), basis=basis,
        cell=np.eye(3) * 100, pbc=(False, False, False),
    )
    data = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")(config, data_processor=processor)
    return table, g2m, data


def _blocks(table, data, g2m, symmetric):
    """Node and edge blocks (n_i, n_j, 4) from the flat 4-column labels."""
    feats = torch.concatenate([data.point_types.reshape(-1, 1) + 1.0, data.positions], axis=1)
    nodes, edges = g2m(data, node_feats=feats)
    assert nodes.shape[1] == edges.shape[1] == 4
    sizes = table.basis_size[data.point_types.numpy()]
    out, start = [], 0
    for n in sizes:
        out.append(nodes[start:start + n * n].reshape(n, n, 4).detach().numpy())
        start += n * n
    assert start == len(nodes)
    edge_index = data.edge_index.numpy()
    if symmetric:
        edge_index = edge_index[:, ::2]
    start = 0
    for i, j in edge_index.T:
        ni, nj = sizes[i], sizes[j]
        out.append(edges[start:start + ni * nj].reshape(ni, nj, 4).detach().numpy())
        start += ni * nj
    assert start == len(edges)
    pairs = [(i, i) for i in range(len(sizes))] + [tuple(e) for e in edge_index.T]
    return out, pairs


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


@pytest.mark.parametrize("convention", ["cartesian", "siesta_spherical"])
@pytest.mark.parametrize("symmetric", [True, False])
def test_graph2mat_spin_orbit_equivariance(symmetric, convention):
    basis = [
        PointBasis("A", R=3, basis="2x0e + 1o", basis_convention=convention),
        PointBasis("B", R=3, basis="0e + 1o", basis_convention=convention),
    ]
    positions = [[1, 1, 1], [2.0, 1.3, 0.8], [1.2, 2.1, 1.4]]
    table, g2m, data = _model_and_data(basis, ["A", "B", "A"], positions, symmetric)
    blocks, pairs = _blocks(table, data, g2m, symmetric)
    # g is axial and stored in Cartesian axes; Q acts on the data (basis convention) frame.
    cob, cob_inv = get_change_of_basis("cartesian", convention)
    irreps = [table.basis[t].e3nn_irreps for t in data.point_types.numpy()]

    rng = np.random.default_rng(0)
    rotations = [_random_rotation(rng) for _ in range(16)] + [-np.eye(3)]
    assert max(abs(b[..., 1:]).max() for b in blocks) > 1e-3
    for Q in rotations:
        rotated = copy.copy(data)
        rotated.positions = data.positions @ torch.tensor(Q).T
        blocks_rot, _ = _blocks(table, rotated, g2m, symmetric)
        R = cob_inv @ Q @ cob
        axial = np.linalg.det(R) * R
        D = [ir.D_from_matrix(torch.tensor(Q)).numpy() for ir in irreps]
        for (i, j), b, b_rot in zip(pairs, blocks, blocks_rot):
            h0 = D[i] @ b[..., 0] @ D[j].T
            g = np.einsum("ab,bxy->axy", axial, D[i] @ np.moveaxis(b[..., 1:], -1, 0) @ D[j].T)
            assert np.allclose(b_rot[..., 0], h0, atol=1e-10)
            assert np.allclose(np.moveaxis(b_rot[..., 1:], -1, 0), g, atol=1e-10)


def test_s_only_species_blocks():
    basis = [PointBasis("A", R=3, basis="0e", basis_convention="cartesian")]
    table, g2m, data = _model_and_data(basis, ["A", "A"], [[0, 0, 0], [1.0, 0.5, 0.2]], True)
    assert g2m.self_interactions[0]._g_irreps_out.dim == 0
    (edge_block,) = [b for b in g2m.interactions.values() if b is not None]
    assert edge_block._g_irreps_out == o3.Irreps("1x1e")
    blocks, pairs = _blocks(table, data, g2m, True)
    for (i, j), b in zip(pairs, blocks):
        if i == j:
            assert np.all(b[..., 1:] == 0)
