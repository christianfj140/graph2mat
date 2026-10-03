import copy

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
from graph2mat.bindings.e3nn import E3nnGraph2Mat


@pytest.fixture(scope="module", params=["point_type", "basis_shape", "max"])
def basis_grouping(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def symmetric(request):
    return request.param


def get_rotation_matrix(data, table, alpha, beta, gamma):
    R_0 = 1
    R_1 = o3.wigner_D(1, alpha, beta, gamma)

    dim = table.basis_size[data.point_types].sum()
    big_R = np.zeros((dim, dim))

    for inv_els in (0, 1, 5, 9, 10):
        big_R[inv_els, inv_els] = R_0

    for start, end in ((2, 5), (6, 9), (11, 14)):
        big_R[start:end, start:end] = R_1

    return R_1, big_R


@pytest.fixture(scope="module", params=[False, True])
def with_no_basis(request):
    """Whether the test should include a point with no basis."""
    return request.param


def test_equivariance(basis_grouping, symmetric, with_no_basis):
    # The basis
    point_1 = PointBasis("A", R=3, basis="2x0e + 1o", basis_convention="cartesian")
    point_2 = PointBasis("B", R=3, basis="0e + 1o", basis_convention="cartesian")
    point_3 = PointBasis("C", R=3, basis="2x0e + 1o", basis_convention="cartesian")

    basis = [point_1, point_2, point_3]

    # Add an extra point with no basis
    if with_no_basis:
        point_4 = PointBasis("F", R=3, basis_convention="cartesian")
        basis.append(point_4)

    # The basis table.
    table = BasisTableWithEdges(basis)
    # The data processor.
    processor = MatrixDataProcessor(
        basis_table=table, symmetric_matrix=symmetric, sub_point_matrix=False
    )

    g2m = E3nnGraph2Mat(
        preprocessing_edges=None,
        unique_basis=table,
        irreps={"node_feats_irreps": o3.Irreps("0e + 1o")},
        symmetric=symmetric,
        basis_grouping=basis_grouping,
    )

    point_types = ["A", "B", "C"]
    positions = [[1, 1, 1], [2.0, 1, 1], [4.0, 1, 1]]
    # Add an extra point with no basis. To make sure that things work
    # in the most general case, we add it in the middle of the list.
    if with_no_basis:
        point_types.insert(1, "F")
        positions.insert(1, [6.0, 1, 1])

    config = BasisConfiguration(
        point_types=point_types,
        positions=np.array(positions),
        basis=basis,
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
    )

    conv = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")
    data = conv(config, data_processor=processor)

    R1, big_R = get_rotation_matrix(
        data, table, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(90 * np.pi / 180)
    )

    data_rot = copy.copy(data)

    data_rot.positions = data.positions @ R1.T

    def get_mat(data):
        node_feats = torch.concatenate(
            [data.point_types.reshape(-1, 1) + 1, data.positions], axis=1
        )
        nodes, edges = g2m(data, node_feats=node_feats)

        return processor.matrix_from_data(
            data,
            predictions={"node_labels": nodes, "edge_labels": edges},
            out_format="numpy",
        )

    pred = get_mat(data)
    pred_rot = get_mat(data_rot)

    post_rotated = big_R @ pred @ big_R.T

    max_diff = abs(post_rotated - pred_rot).max()

    assert max_diff < 1e-5, f"Equivariance error is too high: {max_diff:.2e}.\n"


def test_output_scaleshift_fit_keeps_equivariance():
    torch.manual_seed(0)
    basis = [
        PointBasis("A", R=3, basis="2x0e + 1o", basis_convention="cartesian"),
        PointBasis("B", R=3, basis="0e + 1o", basis_convention="cartesian"),
    ]
    table = BasisTableWithEdges(basis)
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    config = BasisConfiguration(
        point_types=["A", "B", "A"],
        positions=np.array([[1, 1, 1], [2.0, 1.3, 0.8], [1.2, 2.1, 1.4]]),
        basis=basis, cell=np.eye(3) * 100, pbc=(False, False, False),
    )
    data = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")(
        config, data_processor=processor
    )
    kwargs = dict(preprocessing_edges=None, unique_basis=table,
                  irreps={"node_feats_irreps": o3.Irreps("0e + 1o")}, symmetric=True)

    def node_feats(d):
        return torch.concatenate([d.point_types.reshape(-1, 1) + 1.0, d.positions], axis=1)

    # Labels from a "teacher" with an on-site offset, as in a Hamiltonian.
    teacher = E3nnGraph2Mat(**kwargs)
    nodes, edges = teacher(data, node_feats=node_feats(data))
    data.point_labels = (nodes - 7.0).detach()
    data.edge_labels = edges.detach()

    student = E3nnGraph2Mat(**kwargs, output_scaleshift=True)
    student.fit_output_scaleshift([data], table)
    assert bool(student.output_scaleshift_fitted)

    # (H, S) labels with an H-only model (gauge loss, S as a label only): fitted on H alone.
    with_overlap = copy.copy(data)
    with_overlap.point_labels = torch.stack([data.point_labels, torch.rand_like(data.point_labels)], dim=1)
    with_overlap.edge_labels = torch.stack([data.edge_labels, torch.rand_like(data.edge_labels)], dim=1)
    h_only = E3nnGraph2Mat(**kwargs, output_scaleshift=True)
    h_only.fit_output_scaleshift([with_overlap], table)
    assert torch.equal(h_only.state_dict()["self_interactions.0.output_shift"],
                       student.state_dict()["self_interactions.0.output_shift"])

    for block in student.self_interactions:
        ids, scalar = block._irrep_instances()
        assert torch.all(block.output_shift[~scalar] == 0) and torch.any(block.output_shift[scalar] != 0)
        for i in ids.unique():
            assert block.output_scale[ids == i].unique().numel() == 1
    for block in student.interactions.values():
        assert torch.all(block.output_shift == 0)

    R1, big_R = get_rotation_matrix(data, table, torch.tensor(0.3), torch.tensor(1.1), torch.tensor(-0.7))
    data_rot = copy.copy(data)
    data_rot.positions = data.positions @ R1.T

    def get_mat(d):
        n, e = student(d, node_feats=node_feats(d))
        return processor.matrix_from_data(d, predictions={"node_labels": n, "edge_labels": e}, out_format="numpy")

    # big_R from get_rotation_matrix is built for basis (2x0e+1o, 0e+1o, 2x0e+1o) = A, B, A.
    assert abs(big_R @ get_mat(data) @ big_R.T - get_mat(data_rot)).max() < 1e-4


@pytest.mark.parametrize("node_block", ["E3nnLinearTSQNodeBlock", "E3nnProjectedLinearTSQNodeBlock"])
def test_linear_blocks_keep_equivariance(node_block):
    """Linear + tensor product readouts, with random (nonzero) 0e biases."""
    import graph2mat.bindings.e3nn as g2m_e3nn
    from graph2mat.bindings.e3nn import E3nnLinearEdgeBlock

    E3nnLinearTSQNodeBlock = getattr(g2m_e3nn, node_block)

    torch.manual_seed(0)
    basis = [
        PointBasis("A", R=3, basis="2x0e + 1o", basis_convention="cartesian"),
        PointBasis("B", R=3, basis="0e + 1o", basis_convention="cartesian"),
    ]
    table = BasisTableWithEdges(basis)
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    config = BasisConfiguration(
        point_types=["A", "B", "A"],
        positions=np.array([[1, 1, 1], [2.0, 1.3, 0.8], [1.2, 2.1, 1.4]]),
        basis=basis, cell=np.eye(3) * 100, pbc=(False, False, False),
    )
    data = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")(
        config, data_processor=processor
    )
    g2m = E3nnGraph2Mat(
        preprocessing_edges=None, unique_basis=table, irreps={"node_feats_irreps": o3.Irreps("0e + 1o")},
        symmetric=True, node_operation=E3nnLinearTSQNodeBlock, edge_operation=E3nnLinearEdgeBlock,
    )
    with torch.no_grad():
        for block in g2m.self_interactions:
            block.operation.linear.bias.normal_()

    def get_mat(d):
        feats = torch.concatenate([d.point_types.reshape(-1, 1) + 1.0, d.positions], axis=1)
        n, e = g2m(d, node_feats=feats)
        return processor.matrix_from_data(d, predictions={"node_labels": n, "edge_labels": e}, out_format="numpy")

    R1, big_R = get_rotation_matrix(data, table, torch.tensor(0.3), torch.tensor(1.1), torch.tensor(-0.7))
    data_rot = copy.copy(data)
    data_rot.positions = data.positions @ R1.T
    pred, pred_rot = get_mat(data), get_mat(data_rot)
    assert abs(pred).max() > 1e-3
    assert abs(big_R @ pred @ big_R.T - pred_rot).max() < 1e-5


@pytest.mark.parametrize("n_components", [1, 2])
def test_type_layout_matches_the_dynamic_path(with_no_basis, n_components):
    """Roadmap 6.3: the precomputed per-type layout gives exactly the dynamic path's output
    (homonuclear and heteronuclear edges, periodic images, a batch of two structures)."""
    from torch_geometric.data import Batch

    from graph2mat.bindings.torch.modules.graph2mat import torch_type_layout
    from graph2mat.core.modules.graph2mat import LAYOUT_ATTR

    basis = [
        PointBasis("A", R=2.5, basis="2x0e + 1o", basis_convention="cartesian"),
        PointBasis("B", R=2.0, basis="0e + 1o", basis_convention="cartesian"),
    ]
    if with_no_basis:
        basis.append(PointBasis("F", R=2.0, basis_convention="cartesian"))
    table = BasisTableWithEdges(basis)
    processor = MatrixDataProcessor(
        basis_table=table, symmetric_matrix=True, sub_point_matrix=False,
        n_matrix_components=n_components,
    )
    g2m = E3nnGraph2Mat(
        preprocessing_edges=None, unique_basis=table,
        irreps={"node_feats_irreps": o3.Irreps("2x0e + 1o")},
        symmetric=True, basis_grouping="point_type", n_matrix_components=n_components,
    )
    conv = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")
    rng = np.random.default_rng(0)
    structures = []
    for types in (["A", "B", "A", "B"], ["B", "A", "A"] + (["F"] if with_no_basis else [])):
        config = BasisConfiguration(
            point_types=types, positions=rng.uniform(0, 3.0, (len(types), 3)), basis=basis,
            cell=np.eye(3) * 3.0, pbc=(True, True, False),
        )
        structures.append(conv(config, data_processor=processor))
    batch = Batch.from_data_list(structures)
    node_feats = torch.tensor(rng.normal(size=(batch.num_nodes, 5)), dtype=torch.float32)

    dynamic = g2m(batch, node_feats=node_feats)
    object.__setattr__(
        batch, LAYOUT_ATTR,
        torch_type_layout(batch.point_types.numpy(), batch.edge_types.numpy(), table),
    )
    def dynamic_path_used(*args, **kwargs):
        raise AssertionError("the layout was ignored: dynamic path used")

    g2m._get_nodelabels_resort_index = g2m._get_edgelabels_resort_index = dynamic_path_used
    fast = g2m(batch, node_feats=node_feats)

    assert set(getattr(batch, LAYOUT_ATTR)["edge_index"]) >= {0, 1}  # homo- and heteronuclear
    for d, f in zip(dynamic, fast):
        assert d.shape == f.shape
        assert torch.equal(d, f)


def test_type_layout_collate_survives_worker_processes():
    """The layout is a plain attribute of the batch: it must reach the main process from
    DataLoader workers and be ignored by get_example."""
    from torch_geometric.loader import DataLoader

    from graph2mat.core.modules.graph2mat import LAYOUT_ATTR
    from graph2mat.tools.lightning.data import TypeLayoutCollate

    basis = [PointBasis("A", R=2.5, basis="0e + 1o", basis_convention="cartesian")]
    table = BasisTableWithEdges(basis)
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    conv = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")
    rng = np.random.default_rng(1)
    data = [conv(BasisConfiguration(point_types=["A"] * 3, positions=rng.uniform(0, 3.0, (3, 3)),
                                    basis=basis, cell=np.eye(3) * 3.0, pbc=(True, True, False)),
                 data_processor=processor) for _ in range(4)]
    for d in data:  # labels of the right length (symmetric: one direction per edge)
        d.point_labels = torch.zeros(int(table.point_block_pointer(d.point_types.numpy())[-1]))
        d.edge_labels = torch.zeros(int(table.edge_block_size[d.edge_types.numpy()[::2]].sum()))
        d.spectral_C = torch.ones(3 * 4, 2)  # 3 atoms x 4 orbitals, a window of 2 (roadmap 4.4)
    loader = DataLoader(data, batch_size=2, num_workers=2)
    loader.collate_fn = TypeLayoutCollate(loader.collate_fn, table)
    from graph2mat.core.data.metrics import _per_element_structure_ids, gauge_invariant_mse

    for batch in loader:
        layout = getattr(batch, LAYOUT_ATTR)
        assert set(layout) >= {"node_index", "edge_index", "node_sort", "edge_sort", "node_struct"}
        assert batch.get_example(1).num_nodes == 3
        # The structure ids it carries are those computed from the batch itself.
        plain = batch.clone()
        plain.__dict__.pop(LAYOUT_ATTR, None)
        ref = _per_element_structure_ids(batch.point_labels, batch.edge_labels, plain, table)
        for a, b in zip(ref, (layout["node_struct"], layout["edge_struct"], layout["n_structures"])):
            assert torch.equal(torch.as_tensor(a), torch.as_tensor(b))
        # With spectral_C it also carries the orbital of every label element.
        from graph2mat.core.data.metrics import spectral_window_layout

        spectral = spectral_window_layout(batch.edge_labels, plain, table)
        assert batch.spectral_C.shape == (24, 2) and layout["no_max"] == 12 and layout["symmetric"]
        for key in ("node_row", "node_col", "edge_row", "edge_col", "orb_struct", "orb_local"):
            assert torch.equal(layout[key], spectral[key])
