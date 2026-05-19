import pytest
import torch

from graph2mat.core.data.metrics import block_type_huber, block_type_mse


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
