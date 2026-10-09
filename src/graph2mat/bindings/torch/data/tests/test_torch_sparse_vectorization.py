"""Values and differentiability are unchanged by CPU coordinate vectorization."""

import pytest
import torch

from graph2mat.bindings.torch.data.formats import nodes_and_edges_to_coo


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("threshold", [None, .5])
def test_torch_coo_rectangular_images_values_and_gradients(dtype, symmetric, threshold):
    orbitals = torch.tensor([2, 0, 3])
    edges = torch.tensor([[0, 2, 1], [2, 0, 0]])
    images = torch.tensor([1, 2, 0])
    nodes = (torch.arange(13, dtype=dtype) / 8).requires_grad_()
    values = (torch.arange(12, dtype=dtype) / 8 - .75).requires_grad_()
    with torch.no_grad():
        nodes[0] = float("nan")
        values[1] = float("nan")
    actual = nodes_and_edges_to_coo(nodes, values, edges, orbitals, n_supercells=3,
        edge_neigh_isc=images, threshold=threshold, symmetrize_edges=symmetric).coalesce()
    rows, cols, val = [], [], []
    node_index = edge_index = 0
    first = [0, 2, 2, 5]
    for atom, count in enumerate([2, 0, 3]):
        for i in range(count):
            for j in range(count):
                rows.append(first[atom] + i)
                cols.append(first[atom] + j)
                val.append(nodes[node_index])
                node_index += 1
    for (source, target), image in zip(edges.T.tolist(), images.tolist()):
        for i in range(orbitals[source]):
            for j in range(orbitals[target]):
                rows.append(first[source] + i)
                cols.append(first[target] + j + 5 * image)
                val.append(values[edge_index])
                if symmetric:
                    rows.append(first[target] + j)
                    cols.append(first[source] + i + 5 * (0 if image == 0 else 3 - image))
                    val.append(values[edge_index])
                edge_index += 1
    expected_values = torch.stack(val)
    mask = expected_values == expected_values if threshold is None else expected_values.abs() > threshold
    expected = torch.sparse_coo_tensor(torch.tensor([rows, cols])[:, mask], expected_values[mask], (5, 15)).coalesce()
    torch.testing.assert_close(actual.indices(), expected.indices(), rtol=0, atol=0)
    torch.testing.assert_close(actual.values(), expected.values(), rtol=0, atol=0)
    weights = 1 + actual.indices()[0] + actual.indices()[1] / 32
    actual_grads = torch.autograd.grad((actual.values() * weights).sum(), (nodes, values), retain_graph=True)
    expected_grads = torch.autograd.grad((expected.values() * weights).sum(), (nodes, values))
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("ghost", [False, True])
def test_torch_empty_coo_keeps_shape_and_connected_values(ghost):
    orbitals = torch.tensor([0, 0] if ghost else [], dtype=torch.int64)
    nodes = torch.empty(0, requires_grad=True)
    edges = torch.empty(0, requires_grad=True)
    coo = nodes_and_edges_to_coo(nodes, edges, torch.empty((2, 0), dtype=torch.int64), orbitals)
    assert coo.shape == (0, 0)
    assert coo._indices().dtype == torch.int64
    node_grad, edge_grad = torch.autograd.grad(coo.to_dense().sum(), (nodes, edges))
    assert node_grad.numel() == edge_grad.numel() == 0
