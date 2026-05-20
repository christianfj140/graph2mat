from types import SimpleNamespace

import pytest
import torch
from e3nn import o3

from graph2mat.bindings.e3nn.modules.hamiltonian_context import (
    HamiltonianLocalContext,
)


class _Batch(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _vectors_and_lengths(positions, edge_index):
    vectors = positions[edge_index[1]] - positions[edge_index[0]]
    lengths = torch.linalg.norm(vectors, dim=-1)
    return vectors, lengths


def _water_like_batch(dtype=torch.float64):
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.25, 1.12, 0.0],
        ],
        dtype=dtype,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 0, 2, 1, 2],
            [1, 0, 2, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    return _Batch(
        point_types=torch.tensor([0, 1, 1], dtype=torch.long),
        edge_index=edge_index,
        positions=positions,
        shifts=torch.zeros(edge_index.shape[1], 3, dtype=dtype),
        node_attrs=torch.eye(2, dtype=dtype)[torch.tensor([0, 1, 1])],
    )


def test_hamiltonian_local_context_instantiation_and_shapes():
    batch = _water_like_batch()
    vectors, lengths = _vectors_and_lengths(batch.positions, batch.edge_index)
    context = HamiltonianLocalContext(num_types=2, r_max=3.0, num_radial=3)

    output = context(batch, edge_vectors=vectors, edge_lengths=lengths)

    assert output.node.shape == (3, context.node_context_dim)
    assert output.edge.shape == (6, context.edge_context_dim)
    assert context.node_context_irreps == o3.Irreps(f"{context.node_context_dim}x0e")
    assert context.edge_context_irreps == o3.Irreps(f"{context.edge_context_dim}x0e")


def test_hamiltonian_local_context_is_deterministic():
    batch = _water_like_batch()
    vectors, lengths = _vectors_and_lengths(batch.positions, batch.edge_index)
    context = HamiltonianLocalContext(num_types=2, r_max=3.0, num_radial=3)

    first = context(batch, edge_vectors=vectors, edge_lengths=lengths)
    second = context(batch, edge_vectors=vectors, edge_lengths=lengths)

    assert torch.allclose(first.node, second.node)
    assert torch.allclose(first.edge, second.edge)


def test_hamiltonian_local_context_has_position_gradients():
    batch = _water_like_batch()
    batch.positions.requires_grad_(True)
    vectors, lengths = _vectors_and_lengths(batch.positions, batch.edge_index)
    context = HamiltonianLocalContext(num_types=2, r_max=3.0, num_radial=3)

    output = context(batch, edge_vectors=vectors, edge_lengths=lengths)
    loss = output.node.sum() + output.edge.sum()
    loss.backward()

    assert batch.positions.grad is not None
    assert torch.any(batch.positions.grad != 0)


def test_hamiltonian_local_context_preserves_edge_permutation_behavior():
    batch = _water_like_batch()
    vectors, lengths = _vectors_and_lengths(batch.positions, batch.edge_index)
    context = HamiltonianLocalContext(num_types=2, r_max=3.0, num_radial=3)
    reference = context(batch, edge_vectors=vectors, edge_lengths=lengths)

    permutation = torch.tensor([2, 0, 5, 1, 4, 3], dtype=torch.long)
    permuted = _Batch(
        point_types=batch.point_types,
        edge_index=batch.edge_index[:, permutation],
    )
    output = context(
        permuted,
        edge_vectors=vectors[permutation],
        edge_lengths=lengths[permutation],
    )

    assert torch.allclose(output.node, reference.node)
    assert torch.allclose(output.edge, reference.edge[permutation])


def test_hamiltonian_local_context_is_translation_invariant():
    batch = _water_like_batch()
    vectors, lengths = _vectors_and_lengths(batch.positions, batch.edge_index)
    context = HamiltonianLocalContext(num_types=2, r_max=3.0, num_radial=3)
    reference = context(batch, edge_vectors=vectors, edge_lengths=lengths)

    shifted = _water_like_batch()
    shifted.positions = shifted.positions + torch.tensor([4.0, -2.0, 7.0])
    shifted_vectors, shifted_lengths = _vectors_and_lengths(
        shifted.positions, shifted.edge_index
    )
    output = context(
        shifted,
        edge_vectors=shifted_vectors,
        edge_lengths=shifted_lengths,
    )

    assert torch.allclose(output.node, reference.node)
    assert torch.allclose(output.edge, reference.edge)


def test_hamiltonian_local_context_distinguishes_h2o_like_oh_edges():
    batch = _water_like_batch()
    vectors, lengths = _vectors_and_lengths(batch.positions, batch.edge_index)
    context = HamiltonianLocalContext(num_types=2, r_max=3.0, num_radial=4)

    output = context(batch, edge_vectors=vectors, edge_lengths=lengths)

    assert not torch.allclose(output.edge[0], output.edge[2])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"num_types": 0, "r_max": 3.0}, "num_types"),
        ({"num_types": 2, "r_max": 0.0}, "r_max"),
        ({"num_types": 2, "r_max": 3.0, "num_radial": 0}, "num_radial"),
    ],
)
def test_hamiltonian_local_context_rejects_invalid_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HamiltonianLocalContext(**kwargs)


class _FakeMACE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.arange(2)
        self.r_max = 3.0
        self.interactions = [
            SimpleNamespace(
                hidden_irreps=o3.Irreps("2x0e"),
                node_attrs_irreps=o3.Irreps("2x0e"),
                edge_attrs_irreps=o3.Irreps("1x0e"),
                edge_feats_irreps=o3.Irreps("1x0e"),
            ),
            SimpleNamespace(
                hidden_irreps=o3.Irreps("2x0e"),
                node_attrs_irreps=o3.Irreps("2x0e"),
                edge_attrs_irreps=o3.Irreps("1x0e"),
                edge_feats_irreps=o3.Irreps("1x0e"),
            ),
        ]

    def forward(self, data, compute_force=False, **kwargs):
        return {"node_feats": data["mace_node_feats"]}

    def spherical_harmonics(self, vectors):
        return torch.ones(vectors.shape[0], 1, dtype=vectors.dtype)

    def radial_embedding(self, lengths, node_attrs, edge_index, atomic_numbers):
        return lengths.reshape(-1, 1)


class _RecordingReadout(torch.nn.Module):
    def __init__(self, irreps, **kwargs):
        super().__init__()
        self.irreps = irreps
        self.kwargs = kwargs
        self.last_node_feats = None
        self.last_edge_feats = None

    def forward(self, data, node_feats, return_coefficients=False):
        self.last_node_feats = node_feats
        self.last_edge_feats = data["edge_feats"]
        node_labels = torch.zeros(
            node_feats.shape[0],
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        edge_labels = torch.zeros(
            data["edge_index"].shape[1],
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        return node_labels, edge_labels


class _IndexedReadout(torch.nn.Module):
    next_index = 0

    @classmethod
    def reset(cls):
        cls.next_index = 0

    def __init__(self, irreps, **kwargs):
        super().__init__()
        self.irreps = irreps
        self.kwargs = kwargs
        self.index = _IndexedReadout.next_index
        _IndexedReadout.next_index += 1
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, data, node_feats, return_coefficients=False):
        value = self.scale * float(self.index + 1)
        node_labels = torch.ones(
            node_feats.shape[0],
            dtype=node_feats.dtype,
            device=node_feats.device,
        ) * value.to(node_feats)
        edge_labels = torch.ones(
            data["edge_index"].shape[1],
            dtype=node_feats.dtype,
            device=node_feats.device,
        ) * value.to(node_feats)

        if not return_coefficients:
            return node_labels, edge_labels

        coefficients = {
            "node": {"node:0": node_labels.unsqueeze(-1)},
            "edge": {"edge:(0, 0, 0)": edge_labels.unsqueeze(-1)},
            "metadata": {
                "node": {"node:0": {"coefficient_dim": 1}},
                "edge": {"edge:(0, 0, 0)": {"coefficient_dim": 1}},
            },
        }
        return node_labels, edge_labels, coefficients


def _matrix_mace_batch():
    batch = _water_like_batch(dtype=torch.get_default_dtype())
    batch["edge_types"] = torch.ones(batch.edge_index.shape[1], dtype=torch.long)
    batch["mace_node_feats"] = torch.randn(batch.point_types.shape[0], 4)
    return batch


def test_matrix_mace_disabled_path_does_not_add_context():
    pytest.importorskip("mace")
    from graph2mat.models.mace import MatrixMACE

    model = MatrixMACE(
        mace=_FakeMACE(),
        graph2mat_cls=_RecordingReadout,
        hamiltonian_local_context=False,
    )

    output = model(_matrix_mace_batch())
    readout = model.matrix_readouts

    assert readout.last_node_feats.shape[-1] == 4
    assert readout.last_edge_feats.shape[-1] == 1
    assert readout.irreps["node_feats_irreps"].dim == 4
    assert readout.irreps["edge_feats_irreps"].dim == 1
    assert "hamiltonian_node_context" not in output
    assert "hamiltonian_edge_context" not in output


def test_matrix_mace_rejects_context_kwargs_when_disabled():
    pytest.importorskip("mace")
    from graph2mat.models.mace import MatrixMACE

    with pytest.raises(ValueError, match="context is disabled"):
        MatrixMACE(
            mace=_FakeMACE(),
            graph2mat_cls=_RecordingReadout,
            hamiltonian_local_context=False,
            hamiltonian_local_context_kwargs={"num_radial": 2},
        )


def test_matrix_mace_enabled_path_adds_configured_context():
    pytest.importorskip("mace")
    from graph2mat.models.mace import MatrixMACE

    model = MatrixMACE(
        mace=_FakeMACE(),
        graph2mat_cls=_RecordingReadout,
        hamiltonian_local_context=True,
        hamiltonian_local_context_kwargs={"num_radial": 2},
    )

    output = model(_matrix_mace_batch())
    readout = model.matrix_readouts
    context = model.hamiltonian_local_context

    assert readout.last_node_feats.shape[-1] == 4 + context.node_context_dim
    assert readout.last_edge_feats.shape[-1] == 1 + context.edge_context_dim
    assert readout.irreps["node_feats_irreps"].dim == 4 + context.node_context_dim
    assert readout.irreps["edge_feats_irreps"].dim == 1 + context.edge_context_dim
    assert output["hamiltonian_node_context"].shape[-1] == context.node_context_dim
    assert output["hamiltonian_edge_context"].shape[-1] == context.edge_context_dim


def test_matrix_mace_per_interaction_sum_aggregation_is_default():
    pytest.importorskip("mace")
    from graph2mat.models.mace import MatrixMACE

    _IndexedReadout.reset()
    model = MatrixMACE(
        mace=_FakeMACE(),
        graph2mat_cls=_IndexedReadout,
        readout_per_interaction=True,
    )

    output = model(_matrix_mace_batch())

    assert output["node_labels"].shape == (3,)
    assert output["edge_labels"].shape == (6,)
    assert torch.allclose(
        output["node_labels"], torch.full_like(output["node_labels"], 3.0)
    )
    assert torch.allclose(
        output["edge_labels"], torch.full_like(output["edge_labels"], 3.0)
    )


def test_matrix_mace_per_interaction_mean_aggregation_is_explicit():
    pytest.importorskip("mace")
    from graph2mat.models.mace import MatrixMACE

    _IndexedReadout.reset()
    model = MatrixMACE(
        mace=_FakeMACE(),
        graph2mat_cls=_IndexedReadout,
        readout_per_interaction=True,
        readout_interaction_aggregation="mean",
    )

    output = model(_matrix_mace_batch())

    assert torch.allclose(
        output["node_labels"], torch.full_like(output["node_labels"], 1.5)
    )
    assert torch.allclose(
        output["edge_labels"], torch.full_like(output["edge_labels"], 1.5)
    )


def test_matrix_mace_rejects_invalid_readout_interaction_aggregation():
    pytest.importorskip("mace")
    from graph2mat.models.mace import MatrixMACE

    with pytest.raises(ValueError, match="readout_interaction_aggregation"):
        MatrixMACE(
            mace=_FakeMACE(),
            graph2mat_cls=_IndexedReadout,
            readout_per_interaction=True,
            readout_interaction_aggregation="median",
        )


def test_matrix_mace_per_interaction_return_coefficients_and_gradients():
    pytest.importorskip("mace")
    from graph2mat.models.mace import MatrixMACE

    _IndexedReadout.reset()
    model = MatrixMACE(
        mace=_FakeMACE(),
        graph2mat_cls=_IndexedReadout,
        readout_per_interaction=True,
        return_coefficients=True,
    )

    output = model(_matrix_mace_batch())

    assert torch.allclose(
        output["node_labels"], torch.full_like(output["node_labels"], 3.0)
    )
    assert torch.allclose(
        output["edge_labels"], torch.full_like(output["edge_labels"], 3.0)
    )
    assert torch.allclose(
        output["node_coefficients"]["node:0"],
        torch.full_like(output["node_coefficients"]["node:0"], 3.0),
    )
    assert torch.allclose(
        output["edge_coefficients"]["edge:(0, 0, 0)"],
        torch.full_like(output["edge_coefficients"]["edge:(0, 0, 0)"], 3.0),
    )
    assert output["coefficient_metadata"]["node"]["node:0"]["coefficient_dim"] == 1

    loss = (
        output["node_labels"].sum()
        + output["edge_labels"].sum()
        + output["node_coefficients"]["node:0"].sum()
        + output["edge_coefficients"]["edge:(0, 0, 0)"].sum()
    )
    loss.backward()

    readout_gradients = [
        readout.scale.grad for readout in model.matrix_readouts
    ]
    assert all(gradient is not None for gradient in readout_gradients)
    assert all(torch.abs(gradient).item() > 0 for gradient in readout_gradients)
