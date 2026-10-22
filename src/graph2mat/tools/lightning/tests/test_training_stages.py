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


@pytest.mark.parametrize("hook, fn, applied", [("on_test_start", "test", True), ("on_validation_start", "validate", True),
                                               ("on_validation_start", "fit", False)])
def test_standalone_evaluation_uses_the_last_stage_loss(hook, fn, applied):
    """Audit: validate/test from a checkpoint used the default loss while the same evaluation
    right after fit used the last stage's, so the same weights reported different losses."""
    from types import SimpleNamespace

    from graph2mat.tools.lightning.model import LitBasisMatrixModel

    calls = []
    fake = SimpleNamespace(_validated_training_stages=[{"name": "a"}, {"name": "b", "loss": "x"}],
                           trainer=SimpleNamespace(state=SimpleNamespace(fn=fn)),
                           _apply_stage_loss=calls.append)
    getattr(LitBasisMatrixModel, hook)(fake)
    assert calls == ([{"name": "b", "loss": "x"}] if applied else [])


def test_output_scaleshift_without_datamodule_has_a_clear_error():
    from types import SimpleNamespace

    from graph2mat.tools.lightning.model import LitBasisMatrixModel

    readout = SimpleNamespace(output_scaleshift=True, output_scaleshift_fitted=False)
    fake = SimpleNamespace(model=SimpleNamespace(matrix_readouts=readout), trainer=SimpleNamespace(datamodule=None))
    with pytest.raises(ValueError, match="datamodule"):
        LitBasisMatrixModel._fit_output_scaleshift(fake)
