"""Real two-rank Gloo training, isolated from the pytest process."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


def _fit(output_dir, layout):
    import numpy as np
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.strategies import DDPStrategy

    from graph2mat import BasisConfiguration, BasisTableWithEdges, PointBasis
    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    class GradientCoverage(pl.Callback):
        def on_after_backward(self, trainer, pl_module):
            missing = [name for name, parameter in pl_module.named_parameters()
                       if parameter.requires_grad and parameter.grad is None]
            assert not missing, f"Rank {trainer.global_rank}: trainable parameters without gradients: {missing}"

    pl.seed_everything(0, workers=True)
    table = BasisTableWithEdges([PointBasis(kind, R=2, basis=[1]) for kind in ("A", "B")])
    configurations = []
    for index in range(4):
        kinds = ["A", "A", "B", "B"] if layout != "missing_types" else ["A" if index % 2 else "B"] * 2
        configurations.append(BasisConfiguration(
            point_types=kinds, positions=np.array([[i * 0.7, 0., 0.] for i in range(len(kinds))]),
            basis=table, cell=np.eye(3) * 10, pbc=(False, False, False),
            matrix=np.full((len(kinds), len(kinds)), 0.5),
        ))
    module = MatrixDataModule(
        basis_table=table, symmetric_matrix=True, sub_point_matrix=False,
        train_runs=configurations, val_runs=[], loader_threads=0, batch_size=1,
    )
    model = LitMACEMatrixModel(
        basis_table=table, symmetric_matrix=True, hidden_irreps="2x0e+2x1o",
        edge_hidden_irreps="2x0e+2x1o", num_bessel=2, max_ell=1,
    )
    before = {key: parameter.detach().clone() for key, parameter in model.named_parameters()}
    devices = 1 if layout == "reference" else 2
    strategy = (DDPStrategy(process_group_backend="gloo", start_method="popen", find_unused_parameters=False)
                if devices == 2 else "auto")
    trainer = pl.Trainer(
        accelerator="cpu", devices=devices, strategy=strategy,
        max_epochs=2, limit_train_batches=1, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=output_dir,
        callbacks=[GradientCoverage()] if devices == 2 else [],
    )
    trainer.fit(model, datamodule=module)
    if trainer.is_global_zero:
        changed = any(not torch.equal(parameter.detach(), before[key])
                      for key, parameter in model.named_parameters())
        result = {"world_size": trainer.world_size, "global_step": trainer.global_step,
                  "loss": float(trainer.callback_metrics["train_loss_epoch"]), "weights_changed": changed}
        (Path(output_dir) / "result.json").write_text(json.dumps(result))
        torch.save(model.state_dict(), Path(output_dir) / "weights.pt")
    module.teardown(None)


def _run_training_script(output_dir, layout):
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), str(output_dir), layout],
        cwd=output_dir, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        # The isolated process group belongs to this test, including both ranks.
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("Two-rank training timed out:\n" + stdout + stderr)
    (output_dir / "training.log").write_text(stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    return json.loads((output_dir / "result.json").read_text())


def _fit_pair_epoch_metrics(output_dir, variant):
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.strategies import DDPStrategy
    from torch_geometric.loader import DataLoader

    from graph2mat.core.data.metrics import pair_difference_mse
    from graph2mat.tools.lightning.model import LitBasisMatrixModel
    from graph2mat.tools.lightning.tests.test_epoch_metrics import _PredictionModel
    from graph2mat.tools.lightning.tests.test_pair_epoch_metrics import pair_model, pair_samples

    class TrainModel(LitBasisMatrixModel):
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0)

    class PairData(pl.LightningDataModule):
        def train_dataloader(self):
            samples = pair_samples(2 if variant == "spin" else 1)
            if self.trainer.global_rank == 0:
                samples = samples[:1 if variant == "empty_rank" else 2]
                if variant == "empty_rank":
                    samples[0].point_labels.fill_(float("nan"))
            else:
                samples = samples[2:]
            return DataLoader(samples, batch_size=len(samples))

    class Recorder(pl.Callback):
        def __init__(self):
            self.values = []

        def on_after_backward(self, trainer, pl_module):
            assert pl_module.model.offset.grad is not None
            assert torch.isfinite(pl_module.model.offset.grad)

        def on_train_epoch_end(self, trainer, pl_module):
            self.values.append({name: float(trainer.callback_metrics["train_" + name + "_epoch"])
                                for name in ("loss", "pair_rmse", "n_pairs")})

    model = TrainModel(model_cls=_PredictionModel, basis_table=pair_model().basis_table,
                       loss=pair_difference_mse, loss_kwargs={"pair_weight": .25})
    model.init_model()
    recorder = Recorder()
    trainer = pl.Trainer(accelerator="cpu", devices=2, max_epochs=2, logger=False,
                         strategy=DDPStrategy(process_group_backend="gloo", start_method="popen"),
                         callbacks=[recorder], use_distributed_sampler=False, num_sanity_val_steps=0,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         default_root_dir=output_dir)
    trainer.fit(model, datamodule=PairData())
    result = {"world_size": trainer.world_size, "epochs": recorder.values}
    (Path(output_dir) / f"rank{trainer.global_rank}.json").write_text(json.dumps(result))
    if trainer.is_global_zero:
        (Path(output_dir) / "result.json").write_text(json.dumps(result))


def _evaluate_epoch_metrics(output_dir, spin, huber=False, physical=False, coefficient=None, special=None, gauge=None, relative=None):
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.strategies import DDPStrategy
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from graph2mat import BasisTableWithEdges, PointBasis
    from graph2mat.core.data.metrics import block_type_mse, block_type_huber, block_normalized_huber
    from graph2mat.tools.lightning.model import LitBasisMatrixModel

    class Predictions(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            return {"node_labels": batch.node_prediction + self.offset,
                    "edge_labels": batch.edge_prediction + self.offset}

    samples = [
        Data(num_nodes=3, point_labels=torch.zeros(3), node_prediction=torch.ones(3),
             edge_labels=torch.tensor([0., float("nan")]), edge_prediction=torch.tensor([2., 99.])),
        Data(num_nodes=2, point_labels=torch.zeros(2), node_prediction=torch.full((2,), 3.),
             edge_labels=torch.zeros(3), edge_prediction=torch.full((3,), 4.)),
    ]
    if spin:
        for sample in samples:
            for key in ("point_labels", "node_prediction", "edge_labels", "edge_prediction"):
                values = sample[key]
                sample[key] = torch.stack([values, 2 * values], dim=1)
    table = BasisTableWithEdges([PointBasis("A", R=2, basis="0e", basis_convention="spherical")])
    loss_kwargs = {}
    if physical:
        table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1]), PointBasis("B", R=2, basis=[2])])
        samples = [
            Data(num_nodes=3, point_types=torch.zeros(3, dtype=torch.long),
                 point_labels=torch.zeros(3), node_prediction=torch.ones(3),
                 edge_labels=torch.zeros(2), edge_prediction=torch.full((2,), 2.),
                 edge_types=torch.zeros(4, dtype=torch.long),
                 edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])),
            Data(num_nodes=1, point_types=torch.ones(1, dtype=torch.long),
                 point_labels=torch.zeros(4), node_prediction=torch.full((4,), 3.),
                 edge_labels=torch.empty(0), edge_prediction=torch.empty(0),
                 edge_types=torch.empty(0, dtype=torch.long),
                 edge_index=torch.empty((2, 0), dtype=torch.long)),
        ]
        loss_kwargs = {"beta": 1., "node_weight": .25, "edge_weight": .5}
    model = LitBasisMatrixModel(model_cls=Predictions, basis_table=table,
                               loss=block_normalized_huber if physical else block_type_huber if huber else block_type_mse,
                               loss_kwargs=loss_kwargs)
    model.init_model()
    if coefficient:
        from graph2mat.tools.lightning.tests.test_coefficient_epoch_metrics import coefficient_model, coefficient_samples

        loss_name, missing_b = coefficient.split(":")
        samples = coefficient_samples(missing_b == "nan")
        samples[0].num_nodes = 4
        samples[0].point_types = torch.zeros(4, dtype=torch.long)
        samples[0].point_labels = torch.zeros(4)
        samples[0].node_prediction = torch.tensor([1., 1., 1., 3.])
        samples[0].edge_labels = torch.zeros(3)
        samples[0].edge_prediction = torch.tensor([2., 2., 4.])
        samples[0].edge_types = torch.zeros(6, dtype=torch.long)
        samples[0].edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        samples = [samples[0], samples[2]]
        model = coefficient_model(loss_name)
    if special:
        from graph2mat.core.data import metrics as metrics_module

        kwargs = {"threshold": 2.}
        loss_name = special
        if special.startswith("domain"):
            samples = [Data(num_nodes=n, point_types=torch.zeros(n, dtype=torch.long),
                            point_labels=torch.full((n,), float("nan") if special.endswith("nan") and n == 3 else 0.),
                            node_prediction=torch.full((n,), error), edge_labels=torch.empty(0), edge_prediction=torch.empty(0),
                            edge_types=torch.empty(0, dtype=torch.long), edge_index=torch.empty((2, 0), dtype=torch.long))
                       for n, error in ((2, 3.), (3, 2.))]
            kwargs = {"domain_threshold_atoms": 3, "small_domain_weight": .9, "large_domain_weight": .1}
            loss_name = "block_type_mse_per_domain"
        model = LitBasisMatrixModel(model_cls=Predictions, basis_table=table,
                                   loss=getattr(metrics_module, loss_name), loss_kwargs=kwargs)
        model.init_model()
    if gauge:
        from graph2mat.tools.lightning.tests.test_gauge_epoch_metrics import gauge_model, gauge_samples

        samples = gauge_samples(2 if gauge == "spin" else 1, gauge != "h_only")[:2]
        if gauge == "missing_rank":
            samples[0].point_labels.fill_(float("nan"))
            samples[0].edge_labels.fill_(float("nan"))
        model = gauge_model(overlap_weight=.3, spectral_weight=.25 if gauge in ("h_only", "h_s") else 0.)
    if relative:
        from graph2mat.tools.lightning.tests.test_relative_epoch_metrics import relative_model, relative_samples

        loss_name, components, missing = relative.split(":")
        samples = relative_samples(int(components), missing == "nan")
        model = relative_model(loss_name)
    trainer = pl.Trainer(
        accelerator="cpu", devices=2,
        strategy=DDPStrategy(process_group_backend="gloo", start_method="popen"),
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=output_dir,
    )
    metrics = trainer.validate(model, dataloaders=DataLoader(samples, batch_size=1), verbose=False)[0]
    # Exactly two samples for two ranks: DistributedSampler needs no padding/duplicates.
    result = {"world_size": trainer.world_size, "metrics": metrics}
    (Path(output_dir) / f"rank{trainer.global_rank}.json").write_text(json.dumps(result))
    if trainer.is_global_zero:
        (Path(output_dir) / "result.json").write_text(json.dumps(result))


@pytest.mark.parametrize("layout", ["all_types", "missing_types"])
def test_two_cpu_ranks_train_real_mace_for_multiple_steps(tmp_path, layout):
    import math
    import torch

    training = _run_training_script(tmp_path, layout)
    assert training["world_size"] == 2 and training["global_step"] == 2
    assert math.isfinite(training["loss"]) and training["weights_changed"]
    if layout == "all_types":
        # Identical samples on both ranks must yield the same updates as serial training.
        reference_dir = tmp_path / "reference"
        reference_dir.mkdir()
        reference = _run_training_script(reference_dir, "reference")
        assert reference["world_size"] == 1 and reference["global_step"] == 2
        actual = torch.load(tmp_path / "weights.pt", weights_only=True)
        expected = torch.load(reference_dir / "weights.pt", weights_only=True)
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-7, msg=name)


@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("huber", [False, True])
def test_two_cpu_ranks_reduce_global_epoch_statistics(tmp_path, spin, huber):
    import math

    layout = "epoch_metrics" + ("_huber" if huber else "") + ("_spin" if spin else "")
    _run_training_script(tmp_path, layout)
    for rank in (0, 1):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["world_size"] == 2
        metrics = result["metrics"]
        expected = {
            "val_loss": 43 if spin else 86 / 5,
            "hp_metric": 43 if spin else 86 / 5,
            "val_node_rmse": math.sqrt(10.5 if spin else 21 / 5),
            "val_edge_rmse": math.sqrt(32.5 if spin else 13),
            "val_node_mean": 2.7 if spin else 1.8,
            "val_edge_mean": 5.25 if spin else 3.5,
            "val_node_std": math.sqrt(107 / 30 if spin else 1.2),
            "val_edge_std": math.sqrt(79 / 14) if spin else 1,
            "val_node_max": 6 if spin else 3,
            "val_edge_max": 8 if spin else 4,
        }
        if spin:
            expected.update({"val_node_rmse_spin0": math.sqrt(21 / 5),
                             "val_node_rmse_spin1": math.sqrt(84 / 5),
                             "val_edge_rmse_spin1": math.sqrt(52)})
        if huber:
            expected.update({"val_loss": 6.95 if spin else 4.3,
                             "hp_metric": 6.95 if spin else 4.3,
                             "val_node_smooth_l1": 2.2 if spin else 1.3,
                             "val_edge_smooth_l1": 4.75 if spin else 3.})
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value), f"Rank {rank}, {key}"


def test_two_cpu_ranks_reduce_physical_huber_with_disjoint_types(tmp_path):
    import math

    _run_training_script(tmp_path, "epoch_metrics_physical_huber")
    for rank in (0, 1):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["world_size"] == 2
        expected = {
            "val_loss": 1., "hp_metric": 1.,
            "val_node_block_huber": 1., "val_edge_block_huber": 1.5,
            "val_node_blocks": 4., "val_edge_blocks": 2.,
            "val_node_type0_block_huber": .5, "val_node_type1_block_huber": 2.5,
            "val_edge_type0_block_huber": 1.5, "val_edge_type1_block_huber": 0.,
            "val_edge_type2_block_huber": 0., "val_node_rmse": math.sqrt(39 / 7),
            "val_edge_rmse": 2.,
        }
        for key, value in expected.items():
            assert result["metrics"][key] == pytest.approx(value), f"Rank {rank}, {key}"


@pytest.mark.parametrize("loss_name", ["coefficient_space_mse", "coefficient_space_mae", "hamiltonian_composite_loss"])
@pytest.mark.parametrize("missing_b", [False, True])
def test_two_cpu_ranks_reduce_coefficients_with_disjoint_operations(tmp_path, loss_name, missing_b):
    import math

    _run_training_script(tmp_path, "epoch_coefficient:" + loss_name + (":nan" if missing_b else ":valid"))
    coefficient = (25 / 12 if missing_b else 55 / 18) if loss_name == "coefficient_space_mae" else (5.5 if missing_b else 12.)
    label = 4 / 3 if missing_b else 181 / 120
    want = label + .3 * coefficient if loss_name == "hamiltonian_composite_loss" else coefficient
    prefix = "coefficient_" if loss_name == "hamiltonian_composite_loss" else ""
    suffix = "mae" if loss_name == "coefficient_space_mae" else "rmse"
    for rank in (0, 1):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["world_size"] == 2
        expected = {
            "val_loss": want, "hp_metric": want,
            f"val_{prefix}node_node:0_coeff_{suffix}": 1.5 if suffix == "mae" else math.sqrt(3),
            f"val_{prefix}node_node:1_coeff_{suffix}": 0. if missing_b else 5.,
            f"val_{prefix}edge_edge:(0, 0, 0)_coeff_{suffix}": 8 / 3 if suffix == "mae" else math.sqrt(8),
            f"val_{prefix}edge_edge:(0, 1, 1)_coeff_{suffix}": 0.,
            f"val_{prefix}edge_edge:(1, 1, 2)_coeff_{suffix}": 0.,
            f"val_{prefix}coefficient_blocks": 2 if missing_b else 3,
        }
        if loss_name == "hamiltonian_composite_loss":
            expected.update(val_label_huber=label, val_coefficient_mse=coefficient)
        for key, value in expected.items():
            assert result["metrics"][key] == pytest.approx(value), f"Rank {rank}, {key}"


@pytest.mark.parametrize("special,want", [("domain", 8.5), ("domain_nan", 8.1),
                                          ("block_type_mse_threshold", 66.),
                                          ("block_type_mse_sigmoid_thresh", 67.),
                                          ("block_type_mae_sigmoid_thresh", 19.)])
def test_two_cpu_ranks_reduce_size_domains_and_threshold_selections(tmp_path, special, want):
    _run_training_script(tmp_path, "epoch_special:" + special)
    for rank in (0, 1):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["world_size"] == 2
        expected = {"val_loss": want, "hp_metric": want}
        if special.startswith("domain"):
            expected.update(val_n_structures=2, val_n_large_structures=1)
        else:
            expected.update(val_node_above_threshold_frac=.4, val_node_above_threshold_mean=3.,
                            val_edge_above_threshold_frac=.75, val_edge_above_threshold_mean=4.)
        for key, value in expected.items():
            assert result["metrics"][key] == pytest.approx(value), f"Rank {rank}, {key}"


@pytest.mark.parametrize("gauge", ["h_only", "h_s", "spin", "missing_rank"])
def test_two_cpu_ranks_reduce_gauge_and_spectral_denominators(tmp_path, gauge):
    _run_training_script(tmp_path, "epoch_gauge:" + gauge)
    h = 9. if gauge == "missing_rank" else 95 / 7 if gauge == "spin" else 38 / 7
    s = 0. if gauge == "h_only" else 1. if gauge == "missing_rank" else 19 / 28
    loss = h + .3 * s + (.25 * 19 / 10 if gauge in ("h_only", "h_s") else 0.)
    for rank in (0, 1):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["world_size"] == 2
        expected = {"val_loss": loss, "hp_metric": loss, "val_gauge_h_rmse": h ** .5,
                    "val_overlap_rmse": s ** .5,
                    "val_gauge_shift_mean_abs": 10. if gauge == "missing_rank" else 22.5 if gauge == "spin" else 15.}
        if gauge in ("h_only", "h_s"):
            expected.update(val_spectral_rmse=(19 / 10) ** .5, val_spectral_elements=20)
        for name, value in expected.items():
            assert result["metrics"][name] == pytest.approx(value), f"Rank {rank}, {name}"


@pytest.mark.parametrize("variant", ["scalar", "spin", "empty_rank"])
def test_two_cpu_ranks_reduce_pair_means_counts_and_training_reset(tmp_path, variant):
    _run_training_script(tmp_path, "epoch_pair:" + variant)
    factor = 2.5 if variant == "spin" else 1.
    expected = {"loss": 12.5 if variant == "empty_rank" else 59 / 6 * factor,
                "pair_rmse": (26. if variant == "empty_rank" else 56 / 3 * factor) ** .5,
                "n_pairs": 2 if variant == "empty_rank" else 3}
    for rank in (0, 1):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["world_size"] == 2 and len(result["epochs"]) == 2
        for epoch in result["epochs"]:
            assert epoch == pytest.approx(expected), f"Rank {rank}"


@pytest.mark.parametrize("loss_name,components,missing", [
    ("block_type_mape", 1, "valid"), ("block_type_mapemaemix", 1, "valid"),
    ("block_type_mapemsemix", 1, "valid"), ("block_type_mapestdmix", 1, "valid"),
    ("block_type_mapestdmix", 2, "valid"), ("block_type_mapestdmix", 1, "nan"),
])
def test_two_cpu_ranks_reduce_relative_selections_and_sample_variance(tmp_path, loss_name, components, missing):
    from graph2mat.tools.lightning.tests.test_relative_epoch_metrics import relative_expected

    _run_training_script(tmp_path, f"epoch_relative:{loss_name}:{components}:{missing}")
    for rank in (0, 1):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["world_size"] == 2
        expected = relative_expected(loss_name, components, missing == "nan")
        assert result["metrics"]["val_loss"] == pytest.approx(expected), f"Rank {rank}"
        assert result["metrics"]["hp_metric"] == pytest.approx(expected), f"Rank {rank}"


if __name__ == "__main__":
    if sys.argv[2].startswith("epoch_relative:"):
        _evaluate_epoch_metrics(sys.argv[1], False, relative=sys.argv[2].split(":", 1)[1])
    elif sys.argv[2].startswith("epoch_pair:"):
        _fit_pair_epoch_metrics(sys.argv[1], sys.argv[2].split(":", 1)[1])
    elif sys.argv[2].startswith("epoch_gauge:"):
        _evaluate_epoch_metrics(sys.argv[1], False, gauge=sys.argv[2].split(":", 1)[1])
    elif sys.argv[2].startswith("epoch_special:"):
        _evaluate_epoch_metrics(sys.argv[1], False, special=sys.argv[2].split(":", 1)[1])
    elif sys.argv[2].startswith("epoch_coefficient:"):
        _evaluate_epoch_metrics(sys.argv[1], False, coefficient=sys.argv[2].split(":", 1)[1])
    elif sys.argv[2].startswith("epoch_metrics"):
        _evaluate_epoch_metrics(sys.argv[1], sys.argv[2].endswith("spin"), "huber" in sys.argv[2], "physical" in sys.argv[2])
    else:
        _fit(sys.argv[1], sys.argv[2])
