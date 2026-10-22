import pytest
import torch

from graph2mat import BasisTableWithEdges, PointBasis
from graph2mat.core.data.metrics import block_type_huber, block_type_mse
from graph2mat.tools.lightning.model import LitBasisMatrixModel


class _DummyMatrixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mace = torch.nn.Linear(1, 1, bias=False)
        self.matrix_readouts = torch.nn.Linear(1, 1, bias=False)
        self.hamiltonian_local_context = torch.nn.Linear(1, 1, bias=False)

    def forward(self, batch):
        value = batch["x"]
        return {
            "node_labels": self.matrix_readouts(value).reshape(-1),
            "edge_labels": self.mace(value).reshape(-1),
        }


def _basis_table():
    return BasisTableWithEdges([PointBasis("H", R=2.0, basis="1x0e")])


def _lit_model(training_stages=None):
    model = LitBasisMatrixModel(
        model_cls=_DummyMatrixModel,
        basis_table=_basis_table(),
        loss=block_type_mse,
        training_stages=training_stages,
    )
    model.init_model()
    return model


def _requires_grad(module):
    return [parameter.requires_grad for parameter in module.parameters()]


def test_default_training_stage_behavior_is_unchanged():
    model = _lit_model()

    model._apply_training_stage_for_epoch(0)

    assert model._active_training_stage_name is None
    assert isinstance(model.loss_fn, block_type_mse)
    assert all(_requires_grad(model.model.mace))
    assert all(_requires_grad(model.model.matrix_readouts))


def test_training_stage_freezes_and_unfreezes_parameter_groups():
    model = _lit_model(
        training_stages=[
            {
                "name": "coefficient_warmup",
                "freeze": ["mace"],
                "train": ["readout", "hamiltonian_context"],
                "loss": "block_type_huber",
                "loss_kwargs": {"beta": 0.01},
                "max_epochs": 2,
            },
            {
                "name": "full_finetune",
                "freeze": [],
                "max_epochs": 3,
            },
        ]
    )

    model._apply_training_stage_for_epoch(0)

    assert model._active_training_stage_name == "coefficient_warmup"
    assert not any(_requires_grad(model.model.mace))
    assert all(_requires_grad(model.model.matrix_readouts))
    assert all(_requires_grad(model.model.hamiltonian_local_context))
    assert isinstance(model.loss_fn, block_type_huber)
    assert model.loss_fn.beta == pytest.approx(0.01)

    model._apply_training_stage_for_epoch(2)

    assert model._active_training_stage_name == "full_finetune"
    assert all(_requires_grad(model.model.mace))
    assert all(_requires_grad(model.model.matrix_readouts))
    assert isinstance(model.loss_fn, block_type_mse)


@pytest.mark.parametrize(
    "stage, match",
    [
        ({"name": "bad", "freeze": ["unknown"], "max_epochs": 1}, "Unknown"),
        ({"name": "bad", "max_epochs": 0}, "max_epochs"),
        ({"name": "bad", "surprise": True, "max_epochs": 1}, "Unknown"),
        ({"name": "bad", "loss": "not_a_metric", "max_epochs": 1}, "metric"),
    ],
)
def test_training_stage_config_validation(stage, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _lit_model(training_stages=[stage])


def test_freezing_stages_need_find_unused_parameters_under_ddp():
    """Audit: a stage that freezes MACE crashed the first DDP step; it now fails early."""
    from types import SimpleNamespace

    from graph2mat.tools.lightning.model import LitBasisMatrixModel

    def check(stages, ddp_kwargs, world_size=2):
        fake = SimpleNamespace(_validated_training_stages=stages, _trainer=SimpleNamespace(
            world_size=world_size, strategy=SimpleNamespace(_ddp_kwargs=ddp_kwargs)))
        LitBasisMatrixModel._check_stages_under_ddp(fake)

    freezing = [{"name": "a", "freeze": ["mace"], "train": None}]
    with pytest.raises(ValueError, match="find_unused_parameters"):
        check(freezing, {})
    check(freezing, {"find_unused_parameters": True})
    check(freezing, {}, world_size=1)
    check([{"name": "a", "freeze": [], "train": None}], {})


def test_checkpoint_restores_stage_loss_without_freezing_before_fit():
    stages = [{"name": "warmup", "freeze": ["mace"], "max_epochs": 2,
               "loss": "block_type_huber", "loss_kwargs": {"beta": .2}}]
    original = _lit_model(stages)
    original._apply_training_stage_for_epoch(0)
    checkpoint = {"hyper_parameters": {}}
    original.on_save_checkpoint(checkpoint)

    restored = _lit_model(stages)
    restored.on_load_checkpoint(checkpoint)
    assert isinstance(restored.loss_fn, block_type_huber)
    assert restored.loss_fn.beta == .2
    # DDP must register all parameters before a stage changes requires_grad.
    assert all(_requires_grad(restored.model.mace))
    # Restoring the loss must not make the fit hook skip the parameter policy.
    restored._apply_training_stage_for_epoch(0)
    assert not any(_requires_grad(restored.model.mace))


def test_loading_into_a_used_model_reapplies_the_resumed_stage_policy():
    stages = [{"name": "warmup", "max_epochs": 1, "freeze": ["mace"], "loss": "block_type_huber"},
              {"name": "finetune", "max_epochs": 2}]
    original = _lit_model(stages)
    original._apply_training_stage_for_epoch(0)
    checkpoint = {"hyper_parameters": {}}
    original.on_save_checkpoint(checkpoint)
    restored = _lit_model(stages)
    restored._apply_training_stage_for_epoch(1)
    restored.on_load_checkpoint(checkpoint)
    assert isinstance(restored.loss_fn, block_type_huber)
    restored._apply_training_stage_for_epoch(1)
    assert isinstance(restored.loss_fn, block_type_mse)
    assert all(_requires_grad(restored.model.mace))


def test_checkpoint_rejects_active_stage_without_matching_recipe():
    checkpoint = {"hyper_parameters": {}, "active_training_stage": "missing"}
    with pytest.raises(ValueError, match="active training stage 'missing'.*missing or ambiguous"):
        _lit_model().on_load_checkpoint(checkpoint)


@pytest.mark.parametrize("checkpoint_stage", ["early", "last"])
def test_checkpoint_stage_loss_matches_live_validate_and_test(tmp_path, checkpoint_stage):
    import numpy as np
    import pytorch_lightning as pl
    from graph2mat import BasisConfiguration
    from graph2mat.bindings.torch.load import load_from_lit_ckpt
    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    table = _basis_table()
    config = BasisConfiguration(
        point_types=["H", "H"], positions=np.array([[0., 0., 0.], [.7, 0., 0.]]),
        basis=table, cell=np.eye(3) * 10, pbc=(False, False, False), matrix=np.array([[4., 3.], [3., 5.]]),
    )
    data = MatrixDataModule(basis_table=table, symmetric_matrix=True, sub_point_matrix=False,
        train_runs=[config], val_runs=[config], test_runs=[config], loader_threads=0)
    model = LitMACEMatrixModel(basis_table=table, symmetric_matrix=True,
        hidden_irreps="2x0e+2x1o", edge_hidden_irreps="2x0e+2x1o", num_bessel=2, max_ell=1,
        optim_lr=0., training_stages=[{"name": "mae", "max_epochs": 1, "loss": "elementwise_mae"},
                                   {"name": "mse", "max_epochs": 1, "loss": "elementwise_mse"}])
    steps = 1 if checkpoint_stage == "early" else 2

    def trainer():
        return pl.Trainer(accelerator="cpu", devices=1, max_steps=steps, max_epochs=steps, logger=False,
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0, default_root_dir=tmp_path)

    fit = trainer()
    fit.fit(model, datamodule=data)
    path = tmp_path / "stage.ckpt"
    fit.save_checkpoint(path)
    restored, _ = load_from_lit_ckpt(path)
    assert type(restored.loss_fn) is type(model.loss_fn)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    model.eval()
    batch = next(iter(data.val_dataloader()))
    with torch.no_grad():
        prediction = model(batch)
        error = torch.cat((prediction["node_labels"] - batch.point_labels,
                           prediction["edge_labels"] - batch.edge_labels))
        expected = float(error.abs().mean() if checkpoint_stage == "early" else error.square().mean())
    for method, key in (("validate", "val_loss"), ("test", "test_loss")):
        live = getattr(trainer(), method)(model, datamodule=data, verbose=False)[0][key]
        reloaded = getattr(trainer(), method)(restored, datamodule=data, verbose=False)[0][key]
        assert live == pytest.approx(expected)
        assert reloaded == pytest.approx(expected)
    data.teardown(None)


@pytest.mark.parametrize("method,key", [("validate", "val_loss"), ("test", "test_loss")])
def test_evaluation_before_fit_preserves_the_default_loss(method, key):
    import pytorch_lightning as pl
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    model = _lit_model([{"name": "future", "max_epochs": 1, "loss": "block_type_huber"}])
    with torch.no_grad():
        model.model.matrix_readouts.weight.fill_(2.)
        model.model.mace.weight.fill_(3.)
    data = Data(num_nodes=1, x=torch.ones(1, 1), point_types=torch.tensor([0]),
        point_labels=torch.zeros(1), edge_labels=torch.zeros(1), edge_types=torch.tensor([0]),
        edge_index=torch.zeros(2, 1, dtype=torch.long))
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False)
    result = getattr(trainer, method)(model, dataloaders=DataLoader([data]), verbose=False)[0]
    assert result[key] == pytest.approx(13.)  # default block MSE: 2**2 + 3**2
    assert isinstance(model.loss_fn, block_type_mse)


def test_output_scaleshift_without_datamodule_has_a_clear_error():
    from types import SimpleNamespace

    readout = SimpleNamespace(output_scaleshift=True, output_scaleshift_fitted=False)
    fake = SimpleNamespace(model=SimpleNamespace(matrix_readouts=readout), trainer=SimpleNamespace(datamodule=None))
    with pytest.raises(ValueError, match="datamodule"):
        LitBasisMatrixModel._fit_output_scaleshift(fake)
