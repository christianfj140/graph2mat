import pytest
import numpy as np
import torch

from graph2mat.core.data.metrics import (
    block_normalized_huber,
    block_type_huber,
    block_type_mse,
    coefficient_space_mse,
    coefficients_to_labels,
    hamiltonian_composite_loss,
    target_coefficients_from_labels,
)


@pytest.mark.parametrize(
    "loss_fn",
    [
        block_type_mse(),
        block_type_huber(beta=0.25),
        block_type_huber(delta=0.25),
    ],
    ids=["mse", "huber_beta", "huber_delta"],
)
def test_block_type_mse_and_huber_have_gradients(loss_fn):
    nodes_ref = torch.tensor([0.5, -1.0, 2.0])
    edges_ref = torch.tensor([1.5, -0.25, 0.75, -2.0])
    nodes_pred = torch.nn.Parameter(torch.zeros_like(nodes_ref))
    edges_pred = torch.nn.Parameter(torch.zeros_like(edges_ref))

    loss, stats = loss_fn(
        nodes_pred=nodes_pred,
        nodes_ref=nodes_ref,
        edges_pred=edges_pred,
        edges_ref=edges_ref,
    )
    loss.backward()

    assert loss.item() > 0
    assert nodes_pred.grad is not None
    assert edges_pred.grad is not None
    assert torch.any(nodes_pred.grad != 0)
    assert torch.any(edges_pred.grad != 0)
    assert "node_rmse" in stats
    assert "edge_rmse" in stats


@pytest.mark.parametrize(
    "loss_fn",
    [block_type_mse(), block_type_huber(beta=0.25)],
    ids=["mse", "huber"],
)
def test_block_type_mse_and_huber_reduce_in_mini_overfit(loss_fn):
    torch.manual_seed(0)
    nodes_ref = torch.randn(8)
    edges_ref = torch.randn(12)
    nodes_pred = torch.nn.Parameter(torch.zeros_like(nodes_ref))
    edges_pred = torch.nn.Parameter(torch.zeros_like(edges_ref))
    optimizer = torch.optim.Adam([nodes_pred, edges_pred], lr=0.1)

    with torch.no_grad():
        initial_loss, _ = loss_fn(
            nodes_pred=nodes_pred,
            nodes_ref=nodes_ref,
            edges_pred=edges_pred,
            edges_ref=edges_ref,
        )

    for _step in range(200):
        optimizer.zero_grad()
        loss, _ = loss_fn(
            nodes_pred=nodes_pred,
            nodes_ref=nodes_ref,
            edges_pred=edges_pred,
            edges_ref=edges_ref,
        )
        loss.backward()
        optimizer.step()

    final_loss, _ = loss_fn(
        nodes_pred=nodes_pred,
        nodes_ref=nodes_ref,
        edges_pred=edges_pred,
        edges_ref=edges_ref,
    )

    assert final_loss.item() < initial_loss.item() * 1e-3


@pytest.mark.parametrize("kwargs", [{"beta": 0.0}, {"beta": -0.01}])
def test_block_type_huber_validates_beta(kwargs):
    with pytest.raises(ValueError, match="beta"):
        block_type_huber(**kwargs)


class _FakeBasisTable:
    def point_block_pointer(self, point_types):
        sizes = {0: 2, 1: 4}
        pointers = [0]
        for point_type in point_types:
            pointers.append(pointers[-1] + sizes[int(point_type)])
        return pointers

    def edge_block_pointer(self, edge_types):
        sizes = {1: 1, 2: 3}
        pointers = [0]
        for edge_type in edge_types:
            pointers.append(pointers[-1] + sizes[int(abs(edge_type))])
        return pointers


class _FakeReadout:
    symmetric = True


class _FakeModel:
    matrix_readouts = _FakeReadout()


class _FakeBatch:
    point_types = torch.tensor([0, 1])
    edge_types = torch.tensor([1, -1, 2, -2])


class _IdentityCoefficientOperation:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.change_of_basis = torch.eye(int(np.prod(shape))).reshape(-1, *shape)

    def coefficients_to_block(self, coefficients):
        return coefficients.reshape(coefficients.shape[0], *self.shape)


class _CoefficientBasisTable:
    point_block_shape = np.array([[1, 2], [1, 2]])
    edge_block_shape = np.array([[1, 1, 2], [1, 2, 2]])
    edge_block_size = np.array([1, 2, 4])

    def point_block_pointer(self, point_types):
        pointers = [0]
        for point_type in point_types:
            shape = self.point_block_shape[:, int(point_type)]
            pointers.append(pointers[-1] + int(np.prod(shape)))
        return np.asarray(pointers)

    def edge_block_pointer(self, edge_types):
        pointers = [0]
        for edge_type in edge_types:
            pointers.append(
                pointers[-1] + int(self.edge_block_size[abs(int(edge_type))])
            )
        return np.asarray(pointers)


class _CoefficientReadout:
    symmetric = True
    types_to_graph2mat = np.array([0, 1])
    edge_types_to_graph2mat = np.array([0, 1, 2])
    self_interactions = [
        _IdentityCoefficientOperation((1, 1)),
        _IdentityCoefficientOperation((2, 2)),
    ]
    interactions = {
        "(0, 0, 0)": _IdentityCoefficientOperation((1, 1)),
        "(0, 1, 1)": _IdentityCoefficientOperation((1, 2)),
        "(1, 1, 2)": _IdentityCoefficientOperation((2, 2)),
    }


class _CoefficientModel:
    matrix_readouts = _CoefficientReadout()


class _CoefficientBatch:
    point_types = torch.tensor([1, 0, 0])
    edge_types = torch.tensor([-1, 0, 1])


def test_block_normalized_huber_is_zero_for_exact_prediction():
    nodes_ref = torch.tensor([1.0, -1.0, 2.0, 3.0, 4.0, 5.0])
    edges_ref = torch.tensor([0.5, -2.0, 3.0, 4.0])
    loss_fn = block_normalized_huber(beta=0.5)

    loss, stats = loss_fn(
        nodes_pred=nodes_ref.clone(),
        nodes_ref=nodes_ref,
        edges_pred=edges_ref.clone(),
        edges_ref=edges_ref,
        batch=_FakeBatch(),
        basis_table=_FakeBasisTable(),
        model=_FakeModel(),
    )

    assert loss.item() == pytest.approx(0.0)
    assert stats["node_block_huber"].item() == pytest.approx(0.0)
    assert stats["edge_block_huber"].item() == pytest.approx(0.0)


def test_block_normalized_huber_averages_physical_blocks_and_has_gradients():
    nodes_ref = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    edges_ref = torch.tensor([3.0, 4.0, 4.0, 4.0])
    nodes_pred = torch.nn.Parameter(torch.zeros_like(nodes_ref))
    edges_pred = torch.nn.Parameter(torch.zeros_like(edges_ref))
    loss_fn = block_normalized_huber(beta=0.5)

    loss, stats = loss_fn(
        nodes_pred=nodes_pred,
        nodes_ref=nodes_ref,
        edges_pred=edges_pred,
        edges_ref=edges_ref,
        batch=_FakeBatch(),
        basis_table=_FakeBasisTable(),
        model=_FakeModel(),
    )
    loss.backward()

    assert loss.item() == pytest.approx(4.5)
    assert stats["node_block_huber"].item() == pytest.approx(1.25)
    assert stats["edge_block_huber"].item() == pytest.approx(3.25)
    assert stats["node_blocks"].item() == 2
    assert stats["edge_blocks"].item() == 2
    assert "node_type0_block_huber" in stats
    assert "edge_type2_block_huber" in stats
    assert nodes_pred.grad is not None
    assert edges_pred.grad is not None
    assert torch.any(nodes_pred.grad != 0)
    assert torch.any(edges_pred.grad != 0)


def test_block_normalized_huber_differs_from_raw_element_mean_for_unequal_blocks():
    nodes_ref = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    edges_ref = torch.tensor([3.0, 4.0, 4.0, 4.0])
    nodes_pred = torch.zeros_like(nodes_ref)
    edges_pred = torch.zeros_like(edges_ref)
    beta = 0.5

    block_loss, block_stats = block_normalized_huber(beta=beta)(
        nodes_pred=nodes_pred,
        nodes_ref=nodes_ref,
        edges_pred=edges_pred,
        edges_ref=edges_ref,
        batch=_FakeBatch(),
        basis_table=_FakeBasisTable(),
        model=_FakeModel(),
    )
    raw_loss, raw_stats = block_type_huber(beta=beta)(
        nodes_pred=nodes_pred,
        nodes_ref=nodes_ref,
        edges_pred=edges_pred,
        edges_ref=edges_ref,
    )

    assert block_stats["node_block_huber"].item() == pytest.approx(1.25)
    assert block_stats["edge_block_huber"].item() == pytest.approx(3.25)
    assert raw_stats["node_smooth_l1"].item() == pytest.approx(17 / 12)
    assert raw_stats["edge_smooth_l1"].item() == pytest.approx(3.5)
    assert block_loss.item() == pytest.approx(4.5)
    assert raw_loss.item() == pytest.approx(59 / 12)
    assert block_loss.item() != pytest.approx(raw_loss.item())


def test_block_normalized_huber_uses_edge_label_length_for_symmetric_ordering():
    nodes_ref = torch.ones(6)
    edges_ref = torch.ones(3)
    nodes_pred = torch.nn.Parameter(torch.zeros_like(nodes_ref))
    edges_pred = torch.nn.Parameter(torch.zeros_like(edges_ref))
    loss_fn = block_normalized_huber(beta=0.5)

    loss, stats = loss_fn(
        nodes_pred=nodes_pred,
        nodes_ref=nodes_ref,
        edges_pred=edges_pred,
        edges_ref=edges_ref,
        batch=_CoefficientBatch(),
        basis_table=_CoefficientBasisTable(),
        model=_CoefficientModel(),
    )
    loss.backward()

    assert loss.item() > 0
    assert stats["edge_blocks"].item() == 2
    assert "edge_type0_block_huber" in stats
    assert "edge_type1_block_huber" in stats
    assert edges_pred.grad is not None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"beta": 0.0}, "beta"),
        ({"beta": -0.01}, "beta"),
        ({"node_weight": -1.0}, "node_weight"),
        ({"edge_weight": -1.0}, "edge_weight"),
        ({"node_weight": 0.0, "edge_weight": 0.0}, "node_weight"),
    ],
)
def test_block_normalized_huber_validates_kwargs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        block_normalized_huber(**kwargs)


def test_coefficient_space_mse_aligns_targets_with_symmetric_label_reconstruction():
    basis_table = _CoefficientBasisTable()
    readout = _CoefficientReadout()
    batch = _CoefficientBatch()
    nodes_ref = torch.arange(1, 7, dtype=torch.float32)
    edges_ref = torch.arange(10, 13, dtype=torch.float32)

    target = target_coefficients_from_labels(
        readout=readout,
        batch=batch,
        basis_table=basis_table,
        nodes_ref=nodes_ref,
        edges_ref=edges_ref,
    )
    reconstructed = coefficients_to_labels(
        readout=readout,
        batch=batch,
        basis_table=basis_table,
        coefficients=target,
    )

    assert torch.allclose(reconstructed["node_labels"], nodes_ref)
    assert torch.allclose(reconstructed["edge_labels"], edges_ref)

    out = {
        "node_coefficients": target["node"],
        "edge_coefficients": target["edge"],
    }
    loss, stats = coefficient_space_mse(
        nodes_pred=nodes_ref,
        nodes_ref=nodes_ref,
        edges_pred=edges_ref,
        edges_ref=edges_ref,
        batch=batch,
        basis_table=basis_table,
        out=out,
        model=_CoefficientModel(),
    )

    assert loss.item() == pytest.approx(0.0)
    assert stats["coefficient_blocks"].item() == 4


def test_block_normalized_huber_rejects_multicomponent_labels():
    loss_fn = block_normalized_huber(beta=0.5)

    with pytest.raises(ValueError, match="single-component H-only"):
        loss_fn(
            nodes_pred=torch.zeros(2, 2),
            nodes_ref=torch.zeros(2, 2),
            edges_pred=torch.zeros(4, 2),
            edges_ref=torch.zeros(4, 2),
            batch=_FakeBatch(),
            basis_table=_FakeBasisTable(),
            model=_FakeModel(),
        )


def test_hamiltonian_composite_loss_has_gradients():
    nodes_ref = torch.tensor([0.5, -1.0, 2.0])
    edges_ref = torch.tensor([1.5, -0.25, 0.75, -2.0])
    nodes_pred = torch.nn.Parameter(torch.zeros_like(nodes_ref))
    edges_pred = torch.nn.Parameter(torch.zeros_like(edges_ref))
    loss_fn = hamiltonian_composite_loss(
        beta=0.01,
        coefficient_mse_weight=0.0,
        node_weight=2.0,
        edge_weight=0.5,
    )

    loss, stats = loss_fn(
        nodes_pred=nodes_pred,
        nodes_ref=nodes_ref,
        edges_pred=edges_pred,
        edges_ref=edges_ref,
    )
    loss.backward()

    assert loss.item() > 0
    assert nodes_pred.grad is not None
    assert edges_pred.grad is not None
    assert torch.any(nodes_pred.grad != 0)
    assert torch.any(edges_pred.grad != 0)
    assert "label_huber" in stats
    assert "coefficient_mse_weight" in stats


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"label_loss": "mae"}, "label_loss"),
        ({"beta": 0.0}, "beta"),
        ({"coefficient_mse_weight": -1.0}, "coefficient_mse_weight"),
        ({"node_weight": 0.0, "edge_weight": 0.0}, "node_weight"),
    ],
)
def test_hamiltonian_composite_loss_validates_kwargs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        hamiltonian_composite_loss(**kwargs)
