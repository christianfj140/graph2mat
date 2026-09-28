import pytest

from graph2mat import OrbitalMatrixMetric


# Test that all metrics run
@pytest.mark.parametrize(
    "metric",
    [
        metric()
        for metric in OrbitalMatrixMetric.__subclasses__()
        if not getattr(metric, "requires_model_output", False)
        and not getattr(metric, "requires_overlap_labels", False)
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_metric_runs(density_data, density_z_table, metric):
    metric(
        nodes_pred=density_data.point_labels - 0.001,
        nodes_ref=density_data.point_labels,
        edges_pred=density_data.edge_labels,
        edges_ref=density_data.edge_labels,
        batch=density_data,
        basis_table=density_z_table,
    )


def test_normalized_density_error_of_exact_prediction_is_zero(
    density_matrix, density_z_table, symmetric
):
    """With sub_point_matrix=True the labels are DM - DM_atomic; the error of an exact
    prediction must not get the atomic density added back."""
    from graph2mat import MatrixDataProcessor, OrbitalConfiguration
    from graph2mat.bindings.torch import TorchBasisMatrixData
    from graph2mat.core.data.matrices import DensityMatrix
    from graph2mat.core.data.metrics import normalized_density_error
    from graph2mat.core.data.sparse import csr_to_block_dict

    dm_block = csr_to_block_dict(
        density_matrix._csr, density_matrix.atoms, nsc=density_matrix.nsc,
        matrix_cls=DensityMatrix,
    )
    density_config = OrbitalConfiguration.from_geometry(
        geometry=density_matrix.geometry, matrix=dm_block
    )

    processor = MatrixDataProcessor(
        basis_table=density_z_table,
        sub_point_matrix=True,
        symmetric_matrix=symmetric,
        out_matrix="density_matrix",
    )
    data = TorchBasisMatrixData.from_config(density_config, data_processor=processor)
    error, _ = normalized_density_error(
        nodes_pred=data.point_labels,
        nodes_ref=data.point_labels,
        edges_pred=data.edge_labels,
        edges_ref=data.edge_labels,
        batch=data,
        basis_table=density_z_table,
        grid_spacing=0.3,
    )
    assert abs(error) < 1e-8


def test_elementwise_mse_edge_weight_counts_edges_twice():
    import torch
    from graph2mat.core.data.metrics import elementwise_mse

    nodes_pred, nodes_ref = torch.tensor([1.0, 2.0]), torch.zeros(2)
    edges_pred, edges_ref = torch.tensor([3.0]), torch.zeros(1)
    loss, _ = elementwise_mse(nodes_pred=nodes_pred, nodes_ref=nodes_ref,
                              edges_pred=edges_pred, edges_ref=edges_ref)
    assert torch.isclose(loss, torch.tensor((1 + 4 + 9) / 3))
    # The full matrix of a symmetric storage has the edge element twice.
    # (as passed through loss_kwargs in the lightning models)
    loss, _ = elementwise_mse(edge_weight=2)(nodes_pred=nodes_pred, nodes_ref=nodes_ref,
                                             edges_pred=edges_pred, edges_ref=edges_ref)
    assert torch.isclose(loss, torch.tensor((1 + 4 + 2 * 9) / 4))


@pytest.mark.parametrize(
    "name", ["block_type_mape", "block_type_mapemaemix", "block_type_mapemsemix", "block_type_mapestdmix"]
)
def test_relative_losses_have_finite_gradients_with_exact_zeros(name):
    """H has exact zeros (symmetry-zero elements): relative losses must skip them
    without producing NaN gradients (0 * inf in the backward pass)."""
    import torch
    from graph2mat.core.data import metrics

    nodes_ref = torch.tensor([0.0, 2.0, -1.0, 0.0, 3.0])
    edges_ref = torch.tensor([0.0, 0.5, float("nan"), -0.25])
    nodes_pred = (nodes_ref + 0.1).requires_grad_(True)
    edges_pred = torch.tensor([0.1, 0.6, 0.3, -0.2], requires_grad=True)
    loss, _ = getattr(metrics, name)(nodes_pred=nodes_pred, nodes_ref=nodes_ref,
                                     edges_pred=edges_pred, edges_ref=edges_ref)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(nodes_pred.grad).all() and torch.isfinite(edges_pred.grad).all()


def test_gauge_invariant_mse_ignores_a_per_structure_shift():
    """Labels (H, S): predicting H + c_g S with a different c per structure costs ~0."""
    import numpy as np
    import torch
    from graph2mat.core.data.metrics import elementwise_mse, gauge_invariant_mse

    class OneLabelPerSite:
        edge_block_size = np.array([1])

        @staticmethod
        def point_block_pointer(point_types):
            return np.arange(len(point_types) + 1)

    n = 6  # structures 0 (nodes 0-2) and 1 (nodes 3-5), one edge per node inside its structure
    batch = {"point_types": np.zeros(n, dtype=int), "batch": np.array([0, 0, 0, 1, 1, 1]),
             "edge_types": np.zeros(n, dtype=int),
             "edge_index": np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]])}
    torch.manual_seed(0)
    nodes_ref = torch.stack([torch.randn(n), 1 + 0.1 * torch.rand(n)], dim=1)  # (H, S); S_ii ~ 1
    edges_ref = torch.stack([torch.randn(n), 0.3 * torch.rand(n)], dim=1)
    shift = torch.tensor([0.7, -1.3])
    node_c, edge_c = shift[torch.as_tensor(batch["batch"])], shift[torch.as_tensor(batch["batch"])]
    nodes_pred = nodes_ref.clone(); nodes_pred[:, 0] += node_c * nodes_ref[:, 1]
    edges_pred = edges_ref.clone(); edges_pred[:, 0] += edge_c * edges_ref[:, 1]
    nodes_pred.requires_grad_(True)

    kwargs = dict(nodes_pred=nodes_pred, nodes_ref=nodes_ref, edges_pred=edges_pred, edges_ref=edges_ref,
                  batch=batch, basis_table=OneLabelPerSite())
    loss, stats = gauge_invariant_mse(**kwargs)
    assert loss.item() < 1e-10
    assert torch.allclose(stats["gauge_shift_mean_abs"], shift.abs().mean())
    assert elementwise_mse(**kwargs)[0].item() > 0.1
    loss.backward()
    assert torch.isfinite(nodes_pred.grad).all()
