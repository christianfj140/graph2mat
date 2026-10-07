import numpy as np
import pytorch_lightning as pl
import sisl

import pytest

from graph2mat import (
    BasisConfiguration,
    PointBasis,
    BasisTableWithEdges,
)

from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
from graph2mat.tools.lightning import (
    MatrixDataModule,
    MatrixWriter,
    SamplewiseMetricsLogger,
    PlotMatrixError,
)


@pytest.mark.parametrize("out_matrix", ["density_matrix", "hamiltonian"])
def test_lightning_tools_python(out_matrix, tmp_path):
    pl.seed_everything(0, workers=True)
    # The basis
    point_1 = PointBasis("A", R=2, basis="0e", basis_convention="spherical")
    point_2 = PointBasis("B", R=5, basis="2x0e + 1o", basis_convention="spherical")

    basis = [point_1, point_2]

    # The basis table.
    table = BasisTableWithEdges(basis)

    # Lightning model
    model = LitMACEMatrixModel(
        basis_table=table,
        hidden_irreps="0e + 1o + 2e",
        symmetric_matrix=True,
    )

    # Symmetric labels with support restricted to the physical graph.
    matrix = np.random.default_rng(0).random((7, 7))
    matrix = (matrix + matrix.T) / 2
    matrix[0, 6] = matrix[6, 0] = 0  # The distant A-A pair has no edge.
    config1 = BasisConfiguration(
        point_types=["A", "B", "A"],
        positions=np.array([[0, 0, 0], [6.0, 0, 0], [12, 0, 0]]),
        basis=basis,
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
        matrix=matrix,
    )
    configs = [config1]

    # Create the datamodule
    datamodule = MatrixDataModule(
        out_matrix,
        basis_table=table,
        symmetric_matrix=True,
        sub_point_matrix=False,
        train_runs=configs,
        val_runs=configs,
        test_runs=configs,
        loader_threads=0,
    )

    # Temporary files for the callbacks
    matrix_file = tmp_path / ("prediction.DM" if out_matrix == "density_matrix" else "prediction.HSX")
    metrics_file = tmp_path / "metrics.csv"

    # List of callbacks to use during training
    callbacks = [
        MatrixWriter(str(matrix_file), splits=["test"]),
        SamplewiseMetricsLogger(splits=["val"], output_file=str(metrics_file)),
        PlotMatrixError(split="test", show=False, store_in_logger=False),
    ]

    # Create the trainer
    trainer = pl.Trainer(
        callbacks=callbacks, max_epochs=1, logger=False, enable_checkpointing=False,
        accelerator="cpu", devices=1,
        enable_progress_bar=False, enable_model_summary=False,
    )

    # Run training (1 epoch)
    trainer.fit(model, datamodule=datamodule)

    # Run validation to get the validation metrics
    val_metrics, *_ = trainer.validate(model, datamodule=datamodule)
    # Make sure the matrix writer hasn't written anything yet
    # (it is set to only write on test)
    assert not matrix_file.exists()
    assert metrics_file.read_text().strip()

    # Run test to get the test metrics
    test_metrics, *_ = trainer.test(model, datamodule=datamodule)

    # The test set and validation sets are the same, therefore
    # the metrics should be the same
    for k in val_metrics:
        if not k.startswith("val_"):
            continue
        test_k = k.replace("val_", "test_")
        assert (
            val_metrics[k] == test_metrics[test_k]
        ), f"Validation and test metrics do not match for {k} and {test_k}"

    # Check that the written matrix is correct (it should be the same
    # as the one returned by the model)
    data = next(iter(datamodule.train_dataloader()))
    out = model(data)
    pred_matrix = datamodule.data_processor.matrix_from_data(data, out)[0]
    if out_matrix == "density_matrix":
        read_matrix = sisl.get_sile(matrix_file).read_density_matrix()
    elif out_matrix == "hamiltonian":
        read_matrix = sisl.get_sile(matrix_file).read_hamiltonian()

    assert np.allclose(read_matrix.tocsr().toarray(), pred_matrix.tocsr().toarray())
    datamodule.teardown(None)
