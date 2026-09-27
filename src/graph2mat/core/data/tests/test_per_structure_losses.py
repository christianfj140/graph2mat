"""training_weighting_policy (audit Fase 9): legacy bias vs per-structure/per-domain."""

import numpy as np
import pytest
import torch

from graph2mat.core.data.metrics import (
    block_type_mse,
    block_type_mse_per_domain,
    block_type_mse_per_structure,
)


class FakeBasisTable:
    """One label per node and per edge: ids map 1:1 to elements."""

    edge_block_size = np.array([1])

    @staticmethod
    def point_block_pointer(point_types):
        return np.arange(len(point_types) + 1)


def _batch(n_small=2, n_large=8):
    n = n_small + n_large
    point_types = np.zeros(n, dtype=int)
    node_graph = np.array([0] * n_small + [1] * n_large)
    # one intra-structure edge per node
    edge_index = np.stack([np.arange(n), np.roll(np.arange(n), 1)])
    edge_index[:, 0] = [0, 1]  # keep the wrap-around edge inside structure 0
    edge_types = np.zeros(n, dtype=int)
    return {
        "point_types": point_types,
        "batch": node_graph,
        "edge_types": edge_types,
        "edge_index": edge_index,
    }


def _labels(error_small, error_large, n_small=2, n_large=8):
    ref_nodes = torch.zeros(n_small + n_large)
    pred_nodes = torch.cat(
        [torch.full((n_small,), float(error_small)), torch.full((n_large,), float(error_large))]
    )
    ref_edges = torch.zeros(n_small + n_large)
    pred_edges = pred_nodes.clone()
    return pred_nodes, ref_nodes, pred_edges, ref_edges


def _edge_graph_matches_nodes(batch):
    return np.take(batch["batch"], batch["edge_index"][0])


def test_legacy_elementwise_bias_toward_large_structures():
    """The documented legacy bias: the same unit error costs 4x less on the
    small structure than on the large one, purely because of element counts."""
    pred_n, ref_n, pred_e, ref_e = _labels(1, 0)
    loss_small_err, _ = block_type_mse()(
        nodes_pred=pred_n, nodes_ref=ref_n, edges_pred=pred_e, edges_ref=ref_e
    )
    pred_n, ref_n, pred_e, ref_e = _labels(0, 1)
    loss_large_err, _ = block_type_mse()(
        nodes_pred=pred_n, nodes_ref=ref_n, edges_pred=pred_e, edges_ref=ref_e
    )
    # node + edge terms: unit error on the 2-element structure costs 0.4,
    # the SAME unit error on the 8-element structure costs 1.6 (4x more).
    assert loss_small_err.item() == pytest.approx(0.4)
    assert loss_large_err.item() == pytest.approx(1.6)


def test_per_structure_gives_equal_weight_to_equal_normalized_error():
    batch = _batch()
    table = FakeBasisTable()
    kwargs = dict(batch=batch, basis_table=table)
    pred_n, ref_n, pred_e, ref_e = _labels(1, 0)
    loss_small_err, stats = block_type_mse_per_structure()(
        nodes_pred=pred_n, nodes_ref=ref_n, edges_pred=pred_e, edges_ref=ref_e, **kwargs
    )
    pred_n, ref_n, pred_e, ref_e = _labels(0, 1)
    loss_large_err, _ = block_type_mse_per_structure()(
        nodes_pred=pred_n, nodes_ref=ref_n, edges_pred=pred_e, edges_ref=ref_e, **kwargs
    )
    # Same normalized error -> same loss, regardless of structure size.
    assert loss_small_err.item() == pytest.approx(loss_large_err.item())
    assert loss_small_err.item() == pytest.approx(1.0)  # (1+0)/2 nodes + (1+0)/2 edges
    assert stats["n_structures"] == 2


def test_per_structure_keeps_gradients():
    batch = _batch()
    pred_n, ref_n, _, ref_e = _labels(1, 1)
    pred_n = torch.nn.Parameter(pred_n)
    pred_e = torch.nn.Parameter(torch.ones_like(ref_e))
    loss, _ = block_type_mse_per_structure()(
        nodes_pred=pred_n, nodes_ref=ref_n, edges_pred=pred_e, edges_ref=ref_e,
        batch=batch, basis_table=FakeBasisTable(),
    )
    loss.backward()
    assert pred_n.grad is not None and torch.any(pred_n.grad != 0)
    assert pred_e.grad is not None and torch.any(pred_e.grad != 0)


def test_per_domain_respects_weights_and_threshold():
    batch = _batch()
    pred_n, ref_n, pred_e, ref_e = _labels(1, 0)  # error only on small domain
    loss_fn = block_type_mse_per_domain(
        domain_threshold_atoms=5, small_domain_weight=0.9, large_domain_weight=0.1
    )
    loss, stats = loss_fn(
        nodes_pred=pred_n, nodes_ref=ref_n, edges_pred=pred_e, edges_ref=ref_e,
        batch=batch, basis_table=FakeBasisTable(),
    )
    # small-structure loss = 2 (node 1 + edge 1); large = 0.
    assert loss.item() == pytest.approx(0.9 * 2.0)
    assert stats["n_large_structures"] == 1


def test_per_domain_requires_threshold():
    batch = _batch()
    pred_n, ref_n, pred_e, ref_e = _labels(1, 0)
    with pytest.raises(ValueError, match="domain_threshold_atoms"):
        block_type_mse_per_domain()(
            nodes_pred=pred_n, nodes_ref=ref_n, edges_pred=pred_e, edges_ref=ref_e,
            batch=batch, basis_table=FakeBasisTable(),
        )
