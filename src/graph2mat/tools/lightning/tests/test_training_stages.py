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
