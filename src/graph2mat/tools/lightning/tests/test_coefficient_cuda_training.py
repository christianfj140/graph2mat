"""Short CUDA certification; run through bench_pipeline_gpu.py validate."""
import contextlib
import json
import os

import pytest
import pytorch_lightning as pl
import torch

from graph2mat.core.data import metrics
from graph2mat.tools.lightning.tests.test_coefficient_matrix_training import (
    data_module_for, training_inputs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
PRECISIONS = ("32-true", "16-mixed", "bf16-mixed")


@pytest.fixture(autouse=True)
def guarded_cuda():
    if os.environ.get("G2M_GPU_GUARDED") != "1":
        pytest.skip("run through auditorias_mejoras/bench_pipeline_gpu.py validate for VRAM guard")
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def autocast(precision):
    if precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bf16 unavailable")
    return (contextlib.nullcontext() if precision == "32-true" else
            torch.autocast("cuda", dtype=torch.float16 if precision == "16-mixed" else torch.bfloat16))


@pytest.mark.parametrize("precision", PRECISIONS)
def test_cuda_loss_and_all_parameter_gradients_match_matrix(precision, monkeypatch):
    autocast(precision)  # Check hardware before constructing the model.
    model, cpu, _ = training_inputs(components=2, dtype=torch.float32, readout_fp32=True)
    model.cuda()
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    batch = model.transfer_batch_to_device(cpu, "cuda", 0)
    parameters = list(model.parameters())
    with autocast(precision):
        out = model(batch)
        expected, _ = model.loss_fn(nodes_pred=out["node_labels"], edges_pred=out["edge_labels"],
            nodes_ref=batch.point_labels, edges_ref=batch.edge_labels, batch=batch,
            basis_table=model.basis_table, out=out, model=model.model)
        reference = torch.autograd.grad(expected, parameters, allow_unused=True)
        assert model._can_train_coefficients(batch)
        actual = model.training_step(batch, 0)["loss"]
        gradients = torch.autograd.grad(actual, parameters, allow_unused=True)
    tolerance = (5e-6, 5e-6) if precision == "32-true" else (1e-2, 1e-3)
    torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
    errors, relative = [], []
    for value, target in zip(gradients, reference):
        assert (value is None) == (target is None)
        if value is not None:
            assert torch.isfinite(value).all() and torch.isfinite(target).all()
            torch.testing.assert_close(value, target, rtol=tolerance[0], atol=tolerance[1])
            errors.append(float((value - target).abs().max()))
            relative.append(float((value-target).abs().max() / target.abs().max().clamp_min(1e-12)))
    assert errors and torch.isfinite(actual) and torch.isfinite(expected)
    print("CUDA_NUMERIC", json.dumps(dict(precision=precision, loss=float(actual.detach()),
        loss_error=float((actual-expected).abs().detach()), max_gradient_error=max(errors),
        max_tensor_relative_error=max(relative))))


def trainer(precision, epochs):
    return pl.Trainer(accelerator="gpu", devices=1, precision=precision, max_epochs=epochs,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_cuda_trainer_fit_validation_and_checkpoint_resume(precision, tmp_path):
    autocast(precision)
    def inputs():
        model, _, samples = training_inputs(dtype=torch.float32, readout_fp32=True)
        module = data_module_for(model, samples)
        module.val_dataset = samples
        return model, module

    full, module = inputs()
    routes = []
    handle = full.model.register_forward_pre_hook(
        lambda owner, args, kwargs: routes.append(kwargs.get("coefficients_only", False)), with_kwargs=True)
    continuous = trainer(precision, 2)
    continuous.fit(full, datamodule=module)
    handle.remove()
    assert True in routes and False in routes  # Certified train, reconstructed validation.
    assert continuous.global_step == 2 and continuous.current_epoch == 2
    assert torch.isfinite(continuous.callback_metrics["val_loss"])
    full_path = tmp_path / "full.ckpt"
    continuous.save_checkpoint(full_path)
    module.teardown(None)

    first, module = inputs()
    partial = trainer(precision, 1)
    partial.fit(first, datamodule=module)
    path = tmp_path / "partial.ckpt"
    partial.save_checkpoint(path)
    module.teardown(None)
    resumed, module = inputs()
    continuation = trainer(precision, 2)
    continuation.fit(resumed, datamodule=module, ckpt_path=path)
    final_path = tmp_path / "resumed.ckpt"
    continuation.save_checkpoint(final_path)
    assert continuation.global_step == continuous.global_step
    assert continuation.current_epoch == continuous.current_epoch
    tolerance = (5e-6, 5e-6) if precision == "32-true" else (1e-2, 1e-3)

    def compare(actual, expected):
        if torch.is_tensor(expected):
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                compare(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected)
            for value, target in zip(actual, expected):
                compare(value, target)
        else:
            assert actual == expected

    compare(resumed.state_dict(), full.state_dict())
    compare(continuation.optimizers[0].state_dict(), continuous.optimizers[0].state_dict())
    compare(continuation.callback_metrics["val_loss"], continuous.callback_metrics["val_loss"])
    checkpoints = [torch.load(p, map_location="cpu", weights_only=False) for p in (full_path, final_path)]
    precision_key = type(continuous.precision_plugin).__qualname__
    compare(continuation.precision_plugin.state_dict(), continuous.precision_plugin.state_dict())
    for key in ("global_step", "epoch", precision_key, "native_amp_scaling_state"):
        assert (key in checkpoints[0]) == (key in checkpoints[1])
        if key in checkpoints[0]:
            compare(checkpoints[1][key], checkpoints[0][key])
    if precision == "16-mixed":
        scaler_state = continuous.precision_plugin.scaler.state_dict()
        assert scaler_state and "scale" in scaler_state
        assert precision_key in checkpoints[0]
        compare(checkpoints[0][precision_key], scaler_state)
    assert module._coefficient_mse_projector is None
    module.teardown(None)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_cuda_gauge_and_uncertified_amp_keep_matrix_fallback(precision, monkeypatch):
    autocast(precision)
    model, cpu, _ = training_inputs(components=2, dtype=torch.float32,
        loss=metrics.gauge_invariant_mse, readout_fp32=True)
    model.cuda()
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    batch = model.transfer_batch_to_device(cpu, "cuda", 0)
    with autocast(precision), pytest.warns(UserWarning, match="matrix fallback"):
        result = model.training_step(batch, 0)
    assert set(result) == {"loss", "node_labels", "edge_labels"}
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if precision != "32-true":
        uncertified, cpu, _ = training_inputs(dtype=torch.float32, readout_fp32=False)
        uncertified.cuda()
        batch = uncertified.transfer_batch_to_device(cpu, "cuda", 0)
        with autocast(precision), pytest.warns(UserWarning, match="readout_fp32"):
            assert not uncertified._can_train_coefficients(batch)
