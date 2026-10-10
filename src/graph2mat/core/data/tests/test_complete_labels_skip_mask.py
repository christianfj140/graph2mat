"""Roadmap 2.6: complete labels (H has no NaN) skip the NaN row mask.

On the GPU every boolean index is a device -> host sync. The collate knows on the CPU
whether the batch's labels are complete; the losses must then give exactly the same
values and gradients without masking, and keep masking whenever that is not proven.
"""

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from graph2mat.core.data import metrics
from graph2mat.core.modules.graph2mat import LAYOUT_ATTR


class OneLabelPerSite:
    edge_block_size = np.array([1])

    @staticmethod
    def point_block_pointer(point_types):
        return np.arange(len(point_types) + 1)


def _batch(complete):
    n = 6
    torch.manual_seed(0)
    batch = {"point_types": np.zeros(n, dtype=int), "batch": np.array([0, 0, 0, 1, 1, 1]),
             "edge_types": np.zeros(n, dtype=int),
             "edge_index": np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]]),
             "point_labels": torch.stack([torch.randn(n), 1 + 0.1 * torch.rand(n)], dim=1),
             "edge_labels": torch.stack([torch.randn(n), 0.3 * torch.rand(n)], dim=1)}
    if complete is not None:
        batch[LAYOUT_ATTR] = {"labels_complete": complete}
    return batch


def _loss(name, batch, **extra):
    nodes = (batch["point_labels"] + 0.1 * torch.randn_like(batch["point_labels"])).requires_grad_(True)
    edges = (batch["edge_labels"] + 0.1 * torch.randn_like(batch["edge_labels"])).requires_grad_(True)
    torch.manual_seed(1)
    loss, _ = getattr(metrics, name)()(nodes_pred=nodes, nodes_ref=batch["point_labels"], edges_pred=edges,
                                      edges_ref=batch["edge_labels"], batch=batch,
                                      basis_table=OneLabelPerSite(), **extra)
    return loss, torch.autograd.grad(loss, (nodes, edges))


LOSSES = ["block_type_mse", "elementwise_mse", "gauge_invariant_mse", "block_type_mse_per_structure"]


@pytest.mark.parametrize("name", LOSSES)
def test_complete_labels_give_identical_loss_without_mask(name, monkeypatch):
    torch.manual_seed(2)
    want, want_grad = _loss(name, _batch(None))

    def no_mask(values):
        raise AssertionError("complete labels must not build a NaN mask")

    monkeypatch.setattr(metrics, "_valid_label_rows", no_mask)
    torch.manual_seed(2)
    got, got_grad = _loss(name, _batch(True))
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    for g, w in zip(got_grad, want_grad):
        torch.testing.assert_close(g, w, rtol=0, atol=0)


@pytest.mark.parametrize("name", LOSSES)
def test_flag_is_ignored_for_labels_other_than_the_batch_labels(name):
    batch = _batch(True)
    nodes_ref = batch["point_labels"].clone()
    nodes_ref[1, 0] = float("nan")
    nodes = torch.zeros_like(nodes_ref, requires_grad=True)
    loss, _ = getattr(metrics, name)()(nodes_pred=nodes, nodes_ref=nodes_ref, edges_pred=batch["edge_labels"],
                                      edges_ref=batch["edge_labels"], batch=batch, basis_table=OneLabelPerSite())
    assert torch.isfinite(loss)
    (grad,) = torch.autograd.grad(loss, nodes)
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad[1]) == 0


def test_collate_flags_complete_labels_only_without_nan():
    from graph2mat.core.data.tests.test_coefficient_orientation import grouped_readout
    from graph2mat.tools.lightning.data import TypeLayoutCollate
    from graph2mat.tools.lightning.tests.test_coefficient_target_collate import samples_for

    readout = grouped_readout(["1x0e+1x1o", "1x1o+1x2e"], "point_type", True, 1)
    samples = samples_for(readout)
    loader = DataLoader(samples, batch_size=3)
    loader.collate_fn = TypeLayoutCollate(loader.collate_fn, readout.basis_table)
    assert getattr(next(iter(loader)), LAYOUT_ATTR)["labels_complete"] is True
    samples[2].edge_labels[0] = float("nan")
    assert getattr(next(iter(loader)), LAYOUT_ATTR)["labels_complete"] is False
