import pytest

from graph2mat import OrbitalMatrixMetric


# Test that all metrics run
@pytest.mark.parametrize(
    "metric",
    [
        metric()
        for metric in OrbitalMatrixMetric.__subclasses__()
        if not getattr(metric, "requires_model_output", False)
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
