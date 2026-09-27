"""Regression tests for label association, reverse-edge pairing and model copying."""
import copy
import pickle
import warnings

import numpy as np
import pytest
import torch
from e3nn import o3

from graph2mat import BasisConfiguration, BasisTableWithEdges, MatrixDataProcessor, PointBasis
from graph2mat.bindings.e3nn import E3nnEdgeMessageBlock, E3nnGraph2Mat
from graph2mat.core.data import conversions

NODE_IRREPS = o3.Irreps("0e + 1o")


def _two_species_data(symmetric: bool, point_types, positions):
    basis = [
        PointBasis("A", R=2.5, basis="0e + 1o", basis_convention="cartesian"),
        PointBasis("B", R=2.5, basis="2x0e + 1o", basis_convention="cartesian"),
    ]
    table = BasisTableWithEdges(basis)
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=symmetric, sub_point_matrix=False)
    config = BasisConfiguration(
        point_types=point_types,
        positions=np.asarray(positions, dtype=float),
        basis=basis,
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
    )
    conv = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")
    return table, processor, conv(config, data_processor=processor)


def _dense(g2m, processor, data, node_feats):
    nodes, edges = g2m(data, node_feats=node_feats)
    return processor.matrix_from_data(
        data, predictions={"node_labels": nodes, "edge_labels": edges}, out_format="numpy"
    )


def _offsets(table, data):
    sizes = table.basis_size[data.point_types.numpy()]
    return np.concatenate([[0], np.cumsum(sizes)])


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("basis_grouping", ["point_type", "basis_shape", "max"])
def test_blocks_only_depend_on_their_atoms(symmetric, basis_grouping):
    """Changing atom k's features must leave every block not involving k untouched.

    Detects blocks assigned to the wrong edge (e.g. label resorting that mixes up
    the two directions of heteronuclear edges when symmetric=False).
    """
    torch.manual_seed(0)
    positions = [[0, 0, 0], [1.0, 0.2, 0], [0.1, 1.1, 0.3], [1.2, 1.0, -0.2], [0.5, 0.4, 1.0]]
    table, processor, data = _two_species_data(symmetric, ["A", "B", "A", "B", "B"], positions)
    g2m = E3nnGraph2Mat(
        preprocessing_edges=None,
        unique_basis=table,
        irreps={"node_feats_irreps": NODE_IRREPS},
        symmetric=symmetric,
        basis_grouping=basis_grouping,
    )
    feats = torch.randn(len(positions), NODE_IRREPS.dim)
    before = _dense(g2m, processor, data, feats)
    k = 3
    feats[k] = torch.randn(NODE_IRREPS.dim)
    after = _dense(g2m, processor, data, feats)

    off = _offsets(table, data)
    for i in range(len(positions)):
        for j in range(len(positions)):
            block = (slice(off[i], off[i + 1]), slice(off[j], off[j + 1]))
            if k in (i, j):
                continue
            np.testing.assert_allclose(before[block], after[block], atol=1e-6, err_msg=f"block {i},{j}")


def test_nonsymmetric_homonuclear_edge_sees_the_receiver():
    """With edge messages, block ij of a same-species pair must depend on m_ji (atom j's side)."""
    torch.manual_seed(0)
    basis = [PointBasis("A", R=3.0, basis="0e + 1o", basis_convention="cartesian")]
    table = BasisTableWithEdges(basis)
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=False, sub_point_matrix=False)
    config = BasisConfiguration(
        point_types=["A", "A"], positions=np.array([[0.0, 0, 0], [1.1, 0.3, -0.2]]), basis=basis,
        cell=np.eye(3) * 100, pbc=(False, False, False),
    )
    data = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")(
        config, data_processor=processor
    )
    sender, receiver = data.edge_index
    vectors = data.positions[receiver] - data.positions[sender]
    data.edge_attrs = o3.spherical_harmonics("0e + 1o", vectors, normalize=True)
    data.edge_feats = torch.stack([vectors.norm(dim=-1), vectors.norm(dim=-1) ** 2], dim=-1)

    g2m = E3nnGraph2Mat(
        unique_basis=table,
        irreps={
            "node_feats_irreps": NODE_IRREPS,
            "edge_attrs_irreps": o3.Irreps("0e + 1o"),
            "edge_feats_irreps": o3.Irreps("2x0e"),
            "edge_hidden_irreps": o3.Irreps("2x0e + 2x1o"),
        },
        preprocessing_edges=E3nnEdgeMessageBlock,
        symmetric=False,
    )
    feats = torch.randn(2, NODE_IRREPS.dim)
    before = _dense(g2m, processor, data, feats)
    feats[1] = torch.randn(NODE_IRREPS.dim)
    after = _dense(g2m, processor, data, feats)
    # m_01 only depends on atom 0 (the sender), so block (0, 1) only changes through m_10.
    assert not np.allclose(before[:4, 4:], after[:4, 4:])


def test_graph2mat_can_be_copied_and_pickled():
    """EMA/SWA deep-copy the model; checkpoints of whole models pickle it."""
    table, processor, data = _two_species_data(True, ["A", "B"], [[0, 0, 0], [1.0, 0.2, 0]])
    g2m = E3nnGraph2Mat(
        preprocessing_edges=None, unique_basis=table, irreps={"node_feats_irreps": NODE_IRREPS}, symmetric=True
    )
    feats = torch.randn(2, NODE_IRREPS.dim)
    for clone in (copy.deepcopy(g2m), pickle.loads(pickle.dumps(g2m))):
        for a, b in zip(g2m(data, node_feats=feats), clone(data, node_feats=feats)):
            assert torch.equal(a, b)


def _block_warnings(node_irreps: str) -> list:
    basis = [PointBasis("A", R=2.5, basis="0e + 1o", basis_convention="cartesian")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        E3nnGraph2Mat(
            preprocessing_edges=None, unique_basis=basis,
            irreps={"node_feats_irreps": o3.Irreps(node_irreps)}, symmetric=True,
        )
    return [w for w in caught if "cannot be produced" in str(w.message)]


def test_unreachable_irreps_warn():
    # Scalar-only features (e.g. MACE with num_interactions=1) cannot produce the 2e part
    # of a p-p block.
    assert _block_warnings("4x0e")
    assert not _block_warnings("2x0e + 2x1o")


@pytest.mark.parametrize("symmetric", [True, False])
def test_return_coefficients_does_not_change_blocks(symmetric):
    table, processor, data = _two_species_data(symmetric, ["A", "B", "A"], [[0, 0, 0], [1.0, 0.2, 0], [0.2, 1.0, 0.1]])
    g2m = E3nnGraph2Mat(
        preprocessing_edges=None, unique_basis=table, irreps={"node_feats_irreps": NODE_IRREPS}, symmetric=symmetric
    )
    feats = torch.randn(3, NODE_IRREPS.dim)
    nodes, edges = g2m(data, node_feats=feats)
    nodes_c, edges_c, coefficients = g2m(data, node_feats=feats, return_coefficients=True)
    assert torch.allclose(nodes, nodes_c) and torch.allclose(edges, edges_c)
    for key, block in g2m.interactions.items():
        if block is not None and f"edge:{key}" in coefficients["edge"]:
            assert coefficients["edge"][f"edge:{key}"].shape[-1] == block._irreps_out.dim


@pytest.mark.parametrize("mid", [None, 3, "2x0e + 2x1o"])
def test_edge_message_mid_irreps_projects_before_gather(mid):
    irreps = {
        "node_feats_irreps": o3.Irreps("8x0e + 8x1o"),
        "edge_attrs_irreps": o3.Irreps("0e + 1o"),
        "edge_feats_irreps": o3.Irreps("2x0e"),
        "edge_hidden_irreps": o3.Irreps("2x0e + 2x1o"),
        "edge_message_mid_irreps": mid,
    }
    block = E3nnEdgeMessageBlock(irreps)
    expected = {None: irreps["node_feats_irreps"], 3: o3.Irreps("3x0e + 3x1o"), "2x0e + 2x1o": o3.Irreps("2x0e + 2x1o")}[mid]
    assert block.linear_up.irreps_out == expected
    assert block.conv_tp.irreps_in1 == expected

    data = {
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_attrs": torch.randn(2, 4),
        "edge_feats": torch.randn(2, 2),
    }
    _, messages = block(data, torch.randn(2, irreps["node_feats_irreps"].dim))
    assert messages.shape == (2, irreps["edge_hidden_irreps"].dim)
    if mid is not None:
        full = E3nnEdgeMessageBlock({**irreps, "edge_message_mid_irreps": None})
        assert block.conv_tp.weight_numel < full.conv_tp.weight_numel


@pytest.mark.parametrize("n", [1, 2])
def test_drop_last_mace_interaction_keeps_equivariant_features(n):
    """MACE's last layer outputs scalars only; building one extra layer and dropping it
    keeps l>0 features in every layer (with n=1 MACE would give only scalars)."""
    mace_modules = pytest.importorskip("mace.modules")
    from graph2mat.models.mace import drop_last_mace_interaction

    def build(num_interactions):
        return mace_modules.MACE(
            r_max=5.0, num_bessel=4, num_polynomial_cutoff=5, max_ell=2,
            interaction_cls=mace_modules.RealAgnosticResidualInteractionBlock,
            interaction_cls_first=mace_modules.RealAgnosticResidualInteractionBlock,
            num_interactions=num_interactions, num_elements=2, hidden_irreps=o3.Irreps("2x0e+2x1o"),
            MLP_irreps=o3.Irreps("1x0e"), atomic_energies=torch.zeros(2), avg_num_neighbors=4.0,
            atomic_numbers=torch.arange(2), correlation=2, gate=None,
        )

    assert o3.Irreps(build(n).interactions[-1].hidden_irreps).lmax == 0
    kept = drop_last_mace_interaction(build(n + 1))
    assert len(kept.interactions) == len(kept.products) == len(kept.readouts) == n
    assert int(kept.num_interactions) == n
    assert all(o3.Irreps(i.hidden_irreps).lmax == 1 for i in kept.interactions)
