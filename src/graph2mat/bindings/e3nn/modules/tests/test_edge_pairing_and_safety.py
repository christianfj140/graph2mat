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


@pytest.mark.parametrize("options", [{"lean_mace": True}, {"gradient_checkpointing": True}])
def test_lean_mace_matches_full_forward_and_gradients(options):
    """The lean MACE path (no energies, no position gradients) and gradient
    checkpointing must not change the matrix or the parameter gradients."""
    mace_modules = pytest.importorskip("mace.modules")
    from torch_geometric.data import Batch
    from graph2mat.models.mace import MatrixMACE

    positions = [[0, 0, 0], [1.0, 0.2, 0], [0.1, 1.1, 0.3], [1.2, 1.0, -0.2]]
    table, _, data = _two_species_data(True, ["A", "B", "A", "B"], positions)
    batch = Batch.from_data_list([data])
    batch.node_attrs = torch.eye(2)[batch.point_types]

    torch.manual_seed(0)
    mace = mace_modules.MACE(
        r_max=5.0, num_bessel=4, num_polynomial_cutoff=6, max_ell=2,
        interaction_cls=mace_modules.RealAgnosticResidualInteractionBlock,
        interaction_cls_first=mace_modules.RealAgnosticResidualInteractionBlock,
        num_interactions=2, num_elements=2, hidden_irreps=o3.Irreps("4x0e+4x1o"),
        MLP_irreps=o3.Irreps("1x0e"), atomic_energies=torch.zeros(2), avg_num_neighbors=3.0,
        atomic_numbers=torch.arange(2), correlation=2, gate=None,
    )

    def run(**kwargs):
        torch.manual_seed(1)
        model = MatrixMACE(mace=copy.deepcopy(mace), unique_basis=table, symmetric=True,
                           edge_hidden_irreps=o3.Irreps("2x0e+2x1o"), **kwargs)
        out = model(batch.clone())
        loss = (out["node_labels"] ** 2).sum() + (out["edge_labels"] ** 2).sum()
        loss.backward()
        grads = {k: p.grad for k, p in model.named_parameters() if p.grad is not None}
        return out, grads

    full, full_grads = run()
    lean, lean_grads = run(**options)
    assert torch.allclose(full["node_labels"], lean["node_labels"], atol=1e-6)
    assert torch.allclose(full["edge_labels"], lean["edge_labels"], atol=1e-6)
    assert full_grads.keys() == lean_grads.keys()
    for key in full_grads:
        assert torch.allclose(full_grads[key], lean_grads[key], atol=1e-5), key


def test_symm_transpose_single_call_matches_two_calls():
    table, _, _ = _two_species_data(True, ["A", "A"], [[0, 0, 0], [1.0, 0.2, 0]])
    g2m = E3nnGraph2Mat(
        preprocessing_edges=None, unique_basis=table, irreps={"node_feats_irreps": NODE_IRREPS}, symmetric=True
    )
    block = next(b for b in g2m.interactions.values() if b is not None and b.symm_transpose)
    a, b = torch.randn(5, NODE_IRREPS.dim), torch.randn(5, NODE_IRREPS.dim)
    two_calls = (
        block._compute_block(node_feats=(a, b))
        + block._transpose_block(block._compute_block(node_feats=(b, a)))
    ) / 2
    assert torch.allclose(block(node_feats=(a, b)), two_calls, atol=1e-6)


def test_simplified_irreps_index_groups_equal_irreps():
    pytest.importorskip("mace")
    from graph2mat.models.mace import simplified_irreps_index

    irreps, index = simplified_irreps_index(o3.Irreps("2x0e+1x1o+1x0e"))
    assert irreps == o3.Irreps("3x0e+1x1o")
    assert index.tolist() == [0, 1, 5, 2, 3, 4]


def test_simplify_node_irreps_runs_with_fewer_instructions():
    mace_modules = pytest.importorskip("mace.modules")
    from torch_geometric.data import Batch
    from graph2mat.models.mace import MatrixMACE

    positions = [[0, 0, 0], [1.0, 0.2, 0], [0.1, 1.1, 0.3]]
    table, _, data = _two_species_data(True, ["A", "B", "A"], positions)
    batch = Batch.from_data_list([data])
    batch.node_attrs = torch.eye(2)[batch.point_types]
    mace = mace_modules.MACE(
        r_max=5.0, num_bessel=4, num_polynomial_cutoff=6, max_ell=2,
        interaction_cls=mace_modules.RealAgnosticResidualInteractionBlock,
        interaction_cls_first=mace_modules.RealAgnosticResidualInteractionBlock,
        num_interactions=3, num_elements=2, hidden_irreps=o3.Irreps("4x0e+4x1o"),
        MLP_irreps=o3.Irreps("1x0e"), atomic_energies=torch.zeros(2), avg_num_neighbors=3.0,
        atomic_numbers=torch.arange(2), correlation=2, gate=None,
    )
    models = {flag: MatrixMACE(mace=copy.deepcopy(mace), unique_basis=table, symmetric=True,
                               edge_hidden_irreps=o3.Irreps("2x0e+2x1o"), simplify_node_irreps=flag)
              for flag in (False, True)}
    n_instr = {flag: sum(len(m.instructions) for m in model.matrix_readouts.modules()
                         if isinstance(m, o3.TensorSquare)) for flag, model in models.items()}
    assert n_instr[True] < n_instr[False]
    out = models[True](batch.clone())
    assert torch.isfinite(out["node_labels"]).all() and torch.isfinite(out["edge_labels"]).all()


@pytest.mark.parametrize("edge_block", ["E3nnSimpleEdgeBlock", "E3nnLinearEdgeBlock"])
def test_edge_blocks_with_reused_node_features(edge_block):
    """With preprocessing_edges_reuse_nodes the edge operation gets node features and edge
    messages (different irreps): each tensor product must get its own input, by name."""
    import graph2mat.bindings.e3nn as g2m_e3nn

    basis = [PointBasis("A", R=3.0, basis="0e + 1o", basis_convention="cartesian")]
    table = BasisTableWithEdges(basis)
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    config = BasisConfiguration(point_types=["A", "A", "A"], positions=np.array([[0.0, 0, 0], [1.1, .3, -.2], [.2, 1.2, .4]]),
                                basis=basis, cell=np.eye(3) * 100, pbc=(False, False, False))
    data = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")(config, data_processor=processor)
    sender, receiver = data.edge_index
    vectors = data.positions[receiver] - data.positions[sender]
    data.edge_attrs = o3.spherical_harmonics("0e + 1o", vectors, normalize=True)
    data.edge_feats = torch.stack([vectors.norm(dim=-1), vectors.norm(dim=-1) ** 2], dim=-1)
    g2m = E3nnGraph2Mat(
        unique_basis=table,
        irreps={"node_feats_irreps": o3.Irreps("4x0e+4x1o"), "edge_attrs_irreps": o3.Irreps("0e+1o"),
                "edge_feats_irreps": o3.Irreps("2x0e"), "edge_hidden_irreps": o3.Irreps("2x0e+2x1o")},
        preprocessing_edges=E3nnEdgeMessageBlock, preprocessing_edges_reuse_nodes=True,
        edge_operation=getattr(g2m_e3nn, edge_block), symmetric=True,
    )
    nodes, edges = g2m(data, node_feats=torch.randn(3, 16))
    assert torch.isfinite(edges).all() and edges.shape[0] > 0


@pytest.mark.parametrize("edge_block", ["E3nnSimpleEdgeBlock", "E3nnLinearEdgeBlock"])
def test_symmetric_model_predicts_symmetric_matrix(edge_block):
    """symmetric=True must give M = M^T for H, DM and EDM (audit ENG_QUALITY-20)."""
    import graph2mat.bindings.e3nn as g2m_e3nn

    torch.manual_seed(0)
    positions = [[0, 0, 0], [1.0, 0.2, 0], [0.1, 1.1, 0.3], [1.2, 1.0, -0.2]]
    table, processor, data = _two_species_data(True, ["A", "B", "A", "B"], positions)
    sender, receiver = data.edge_index
    vectors = data.positions[receiver] - data.positions[sender]
    data.edge_attrs = o3.spherical_harmonics("0e + 1o", vectors, normalize=True)
    data.edge_feats = torch.stack([vectors.norm(dim=-1), vectors.norm(dim=-1) ** 2], dim=-1)
    g2m = E3nnGraph2Mat(
        unique_basis=table,
        irreps={"node_feats_irreps": NODE_IRREPS, "edge_attrs_irreps": o3.Irreps("0e+1o"),
                "edge_feats_irreps": o3.Irreps("2x0e"), "edge_hidden_irreps": o3.Irreps("2x0e+2x1o")},
        preprocessing_edges=E3nnEdgeMessageBlock, edge_operation=getattr(g2m_e3nn, edge_block), symmetric=True,
    )
    matrix = _dense(g2m, processor, data, torch.randn(len(positions), NODE_IRREPS.dim))
    assert abs(matrix).max() > 1e-4
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile (inductor) test needs a GPU")
def test_compiled_submodules_match_eager():
    """compile_submodules must not change the matrix nor the gradients."""
    mace_modules = pytest.importorskip("mace.modules")
    from mace.tools.compile import disable_e3nn_codegen
    from torch_geometric.data import Batch
    from graph2mat.models.mace import MatrixMACE

    positions = [[0, 0, 0], [1.0, 0.2, 0], [0.1, 1.1, 0.3], [1.2, 1.0, -0.2]]
    table, _, data = _two_species_data(True, ["A", "B", "A", "B"], positions)
    batch = Batch.from_data_list([data])
    batch.node_attrs = torch.eye(2)[batch.point_types]
    batch = batch.to("cuda")
    torch.manual_seed(0)
    with disable_e3nn_codegen():
        mace = mace_modules.MACE(
            r_max=5.0, num_bessel=4, num_polynomial_cutoff=6, max_ell=2,
            interaction_cls=mace_modules.RealAgnosticResidualInteractionBlock,
            interaction_cls_first=mace_modules.RealAgnosticResidualInteractionBlock,
            num_interactions=2, num_elements=2, hidden_irreps=o3.Irreps("4x0e+4x1o"),
            MLP_irreps=o3.Irreps("1x0e"), atomic_energies=torch.zeros(2), avg_num_neighbors=3.0,
            atomic_numbers=torch.arange(2), correlation=2, gate=None,
        )
        model = MatrixMACE(mace=mace, unique_basis=table, symmetric=True, lean_mace=True,
                           edge_hidden_irreps=o3.Irreps("2x0e+2x1o")).cuda()

    def run():
        model.zero_grad()
        out = model(batch.clone())
        ((out["node_labels"] ** 2).sum() + (out["edge_labels"] ** 2).sum()).backward()
        return out, {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}

    eager, eager_grads = run()
    keys = set(model.state_dict())
    model.compile_submodules()
    compiled, compiled_grads = run()
    assert set(model.state_dict()) == keys  # checkpoints unchanged
    assert torch.allclose(eager["edge_labels"], compiled["edge_labels"], atol=1e-5)
    assert torch.allclose(eager["node_labels"], compiled["node_labels"], atol=1e-5)
    for key, grad in eager_grads.items():
        assert torch.allclose(grad, compiled_grads[key], atol=1e-4), key


@pytest.mark.parametrize("layers", [1, 2])
def test_edge_update_layers(layers):
    """Roadmap 7.2: with edge updates the message of j -> i also depends on the receiver i,
    stays equivariant, vanishes with the radial features (cutoff) and the model's matrix
    stays symmetric."""
    torch.manual_seed(0)
    irreps = {
        "node_feats_irreps": o3.Irreps("4x0e + 4x1o + 2x2e"),
        "edge_attrs_irreps": o3.Irreps("0e + 1o + 2e"),
        "edge_feats_irreps": o3.Irreps("2x0e"),
        "edge_hidden_irreps": o3.Irreps("2x0e + 2x1o + 2x2e"),
        "edge_update_layers": layers,
        "edge_update_node_irreps": 3,
    }
    block = E3nnEdgeMessageBlock(irreps)
    base = E3nnEdgeMessageBlock({**irreps, "edge_update_layers": 0})
    base.load_state_dict(block.state_dict(), strict=False)
    vectors = torch.randn(3, 3)
    data = {"edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
            "edge_attrs": o3.spherical_harmonics(irreps["edge_attrs_irreps"], vectors, normalize=True),
            "edge_feats": torch.rand(3, 2)}
    feats = torch.randn(3, irreps["node_feats_irreps"].dim)
    _, messages = block(data, feats)

    # Receiver dependence: changing node 1 (receiver of edge 0 only, sender of edge 1).
    other = feats.clone()
    other[1] += torch.randn_like(other[1])
    _, changed = block(data, other)
    _, changed_base = base(data, other)
    _, messages_base = base(data, feats)
    assert torch.allclose(changed_base[0], messages_base[0])  # without updates, edge 0 ignores its receiver
    assert not torch.allclose(changed[0], messages[0])

    # Equivariance.
    rot = o3.rand_matrix()
    rotated = {**data, "edge_attrs": o3.spherical_harmonics(irreps["edge_attrs_irreps"], vectors @ rot.T, normalize=True)}
    _, messages_rot = block(rotated, feats @ irreps["node_feats_irreps"].D_from_matrix(rot).T)
    expected = messages @ irreps["edge_hidden_irreps"].D_from_matrix(rot).T
    assert torch.allclose(messages_rot, expected, atol=1e-5)

    # Zero radial features (an edge at the cutoff): zero message.
    _, at_cutoff = block({**data, "edge_feats": torch.zeros(3, 2)}, feats)
    assert at_cutoff.abs().max() == 0

    # The full symmetric model still predicts a symmetric matrix.
    positions = [[0, 0, 0], [1.0, 0.2, 0], [0.1, 1.1, 0.3], [1.2, 1.0, -0.2]]
    table, processor, data = _two_species_data(True, ["A", "B", "A", "B"], positions)
    sender, receiver = data.edge_index
    vectors = data.positions[receiver] - data.positions[sender]
    data.edge_attrs = o3.spherical_harmonics("0e + 1o", vectors, normalize=True)
    data.edge_feats = torch.stack([vectors.norm(dim=-1), vectors.norm(dim=-1) ** 2], dim=-1)
    g2m = E3nnGraph2Mat(
        unique_basis=table,
        irreps={"node_feats_irreps": NODE_IRREPS, "edge_attrs_irreps": o3.Irreps("0e+1o"),
                "edge_feats_irreps": o3.Irreps("2x0e"), "edge_hidden_irreps": o3.Irreps("2x0e+2x1o"),
                "edge_update_layers": layers},
        preprocessing_edges=E3nnEdgeMessageBlock, symmetric=True,
    )
    matrix = _dense(g2m, processor, data, torch.randn(len(positions), NODE_IRREPS.dim))
    assert abs(matrix).max() > 1e-4
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
