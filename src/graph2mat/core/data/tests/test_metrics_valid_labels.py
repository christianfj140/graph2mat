"""Missing labels and empty branches must not poison matrix losses or gradients."""

import numpy as np
import pytest
import torch

from graph2mat.core.data import metrics


# Literal losses for node errors [1,1] and edge errors [2,2], or the two-channel
# errors [1,2,1,2] and [2,4,2,4]. Relative errors are [0.5] and [1].
BASIC_LOSSES = [
    ("block_type_mse", 5., 12.5, 1.),
    ("elementwise_mse", 2.5, 6.25, 1.),
    ("block_type_mae", 3., 4.5, 1.),
    ("elementwise_mae", 1.5, 2.25, 1.),
    ("node_mse", 1., 2.5, 1.),
    ("edge_mse", 4., 10., 1.),
    ("block_type_mape", 1.5, 1.5, 1.),
    ("block_type_mapemaemix", 301.5, 451.5, 101.),
    ("block_type_mapemsemix", 50001.5, 125001.5, 10001.),
    ("block_type_mapestdmix", 1.5, 1.5, 1.),
    ("block_type_mse_threshold", 10., 50., 1.),
    ("block_type_mse_sigmoid_thresh", 10., 50., 1.),
    ("block_type_mae_sigmoid_thresh", 6., 18., 1.),
    ("block_type_huber", 2., 3.5, 0.5),
    ("block_type_smooth_l1", 2., 3.5, 0.5),
]


def _array(values, backend, prediction=False):
    values = np.asarray(values, dtype=np.float64)
    return (torch.tensor(values, requires_grad=prediction)
            if backend == "torch" else values)


def _check_gradients(loss, nodes, edges, name, missing_row=None):
    loss.backward()
    for prediction, unused in ((nodes, name == "edge_mse"), (edges, name == "node_mse")):
        if unused:
            assert prediction.grad is None  # The one-branch objectives are intentional.
            continue
        assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
        if missing_row is not None:
            assert torch.count_nonzero(prediction.grad[missing_row]) == 0


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("name,want_scalar,want_spin,one_branch", BASIC_LOSSES)
def test_basic_losses_ignore_nan_label_rows(backend, components, name, want_scalar, want_spin, one_branch):
    if components == 1:
        nr, er = [0., 2., np.nan], [0., -2., np.nan]
        npred, epred = [1., 3., 99.], [2., 0., 88.]
    else:
        nr, er = [[0., 0.], [2., 4.], [np.nan, 1.]], [[0., 0.], [-2., -4.], [1., np.nan]]
        npred, epred = [[1., 2.], [3., 6.], [99., 88.]], [[2., 4.], [0., 0.], [77., 66.]]
    nodes, edges = _array(npred, backend, True), _array(epred, backend, True)
    loss, stats = getattr(metrics, name)()(
        nodes_pred=nodes, nodes_ref=_array(nr, backend),
        edges_pred=edges, edges_ref=_array(er, backend), log_verbose=True,
    )
    value = loss.detach().item() if backend == "torch" else loss
    assert value == pytest.approx(want_scalar if components == 1 else want_spin)
    assert all(np.isfinite(float(value)) for value in stats.values())
    for key in ("node_above_threshold_frac", "edge_above_threshold_frac"):
        if key in stats:
            assert float(stats[key]) == 1.
    if backend == "torch":
        _check_gradients(loss, nodes, edges, name, missing_row=2)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("missing", ["nodes", "edges", "both", "masked"])
@pytest.mark.parametrize("name,want_scalar,want_spin,one_branch", BASIC_LOSSES)
def test_basic_losses_have_finite_empty_branch_reductions(backend, missing, name, want_scalar, want_spin, one_branch):
    nr = [] if missing in ("nodes", "both") else [np.nan] if missing == "masked" else [1.]
    er = [] if missing in ("edges", "both") else [np.nan] if missing == "masked" else [1.]
    nodes, edges = _array([2.] * len(nr), backend, True), _array([2.] * len(er), backend, True)
    loss, stats = getattr(metrics, name)()(
        nodes_pred=nodes, nodes_ref=_array(nr, backend),
        edges_pred=edges, edges_ref=_array(er, backend), log_verbose=True,
    )
    want = (0. if missing in ("both", "masked")
            or (missing == "nodes" and name == "node_mse")
            or (missing == "edges" and name == "edge_mse") else one_branch)
    value = loss.detach().item() if backend == "torch" else loss
    assert value == pytest.approx(want)
    assert all(np.isfinite(float(value)) for value in stats.values())
    if backend == "torch":
        _check_gradients(loss, nodes, edges, name)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("objective", ["normalized", "composite", "composite_blocks"])
@pytest.mark.parametrize("missing", ["mixed", "nodes", "edges", "both", "no_edges"])
def test_huber_objectives_align_missing_labels_with_physical_blocks(backend, objective, missing):
    from graph2mat import BasisConfiguration, BasisTableWithEdges, MatrixDataProcessor, PointBasis
    from graph2mat.bindings.torch import TorchBasisMatrixData

    table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1]), PointBasis("B", R=2, basis=[2])])
    config = BasisConfiguration(
        point_types=["A", "B"], positions=np.array([[0., 0., 0.], [8. if missing == "no_edges" else 0.7, 0., 0.]]),
        basis=table, cell=np.eye(3) * 20, pbc=(False, False, False), matrix=np.ones((3, 3)),
    )
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    data = TorchBasisMatrixData.from_config(config, data_processor=processor)
    # Five node elements in blocks of size 1 and 4; A is wholly absent, B has
    # three valid errors [1,2,3]. The A-B edge has one valid error [4].
    nr, npred = [np.nan, 1., 1., np.nan, 1.], [99., 2., 3., 88., 4.]
    er, epred = ([], []) if missing == "no_edges" else ([1., np.nan], [5., 77.])
    if missing in ("nodes", "both"):
        nr = [np.nan] * 5
    if missing in ("edges", "both"):
        er = [np.nan] * len(er)
    nodes, edges = _array(npred, backend, True), _array(epred, backend, True)
    if objective == "normalized":
        loss_fn = metrics.block_normalized_huber(beta=1)
    else:
        loss_fn = metrics.hamiltonian_composite_loss(beta=1, per_block_normalization=objective == "composite_blocks")
    loss, stats = loss_fn(nodes_pred=nodes, nodes_ref=_array(nr, backend),
                          edges_pred=edges, edges_ref=_array(er, backend),
                          batch=data, basis_table=table, log_verbose=True)
    # Smooth L1 errors: [.5,1.5,2.5] for B, [3.5] for A-B.
    want_node = 0. if missing in ("nodes", "both") else 1.5
    want_edge = 0. if missing in ("edges", "both", "no_edges") else 3.5
    value = loss.detach().item() if backend == "torch" else loss
    assert value == pytest.approx(want_node + want_edge)
    assert all(np.isfinite(float(value)) for value in stats.values())
    if objective == "normalized":
        assert float(stats["node_blocks"]) == (0 if missing in ("nodes", "both") else 1)
        assert "node_type0_block_huber" not in stats
        if missing not in ("nodes", "both"):
            assert float(stats["node_type1_block_huber"]) == pytest.approx(1.5)
    if backend == "torch":
        _check_gradients(loss, nodes, edges, objective)
        assert torch.count_nonzero(nodes.grad[[0, 3]]) == 0
        if len(er):
            assert edges.grad[1] == 0
