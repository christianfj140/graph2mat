"""Opt-in fusion preserves the full model, optimizer and checkpoint contracts."""

import copy
import json
import os

import pytest
import torch

from graph2mat.bindings.e3nn import E3nnLinearEdgeBlock
from graph2mat.tools.lightning.tests.test_coefficient_matrix_training import training_inputs


@pytest.fixture(autouse=True)
def double_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def paired_models(**kwargs):
    reference, batch, samples = training_inputs(edge_block_readout=E3nnLinearEdgeBlock, **kwargs)
    candidate, other, _ = training_inputs(edge_block_readout=E3nnLinearEdgeBlock,
                                        edge_block_readout_kwargs={"fuse_homonuclear": True}, **kwargs)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    return reference, candidate, batch, other, samples


def loss(model, batch):
    out = model(batch.clone())
    return ((out["node_labels"] - batch.point_labels).square().sum()
            + 2 * (out["edge_labels"] - batch.edge_labels).square().sum()
            ) / (batch.point_labels.numel() + 2 * batch.edge_labels.numel())


@pytest.mark.parametrize("grouping,mode,components", [("point_type", "single", 1), ("point_type", "sum", 2),
                                                      ("basis_shape", "mean", 1), ("basis_shape", "single", 2),
                                                      ("max", "sum", 1), ("max", "mean", 2)])
@pytest.mark.parametrize("checkpoint,layout", [(False, True), (True, False)])
def test_full_model_values_losses_and_gradients(grouping, mode, components, checkpoint, layout):
    reference, candidate, batch, other, _ = paired_models(
        grouping=grouping, mode=mode, components=components,
        coefficient_training=False, gradient_checkpointing=checkpoint)
    if not layout:
        from graph2mat.core.modules.graph2mat import LAYOUT_ATTR
        object.__delattr__(batch, LAYOUT_ATTR)
        object.__delattr__(other, LAYOUT_ATTR)
    expected, actual = reference(batch.clone()), candidate(other.clone())
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-10, atol=1e-12)
    expected_loss, actual_loss = loss(reference, batch), loss(candidate, other)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-10, atol=1e-12)
    ref_parameters, actual_parameters = dict(reference.named_parameters()), dict(candidate.named_parameters())
    assert list(ref_parameters) == list(actual_parameters)
    expected_loss.backward()
    actual_loss.backward()
    ref_grad = [p.grad for p in ref_parameters.values()]
    actual_grad = [p.grad for p in actual_parameters.values()]
    for name, left, right in zip(ref_parameters, ref_grad, actual_grad):
        assert (left is None) == (right is None), name
        if left is not None:
            torch.testing.assert_close(right, left, rtol=1e-10, atol=1e-12, msg=name)
    assert any(getattr(module, "_fused_homonuclear", False) for module in candidate.modules())
    assert list(candidate.state_dict()) == list(reference.state_dict())
    reference.load_state_dict(candidate.state_dict(), strict=True)


@pytest.mark.parametrize("components", [1, 2])
def test_coefficient_training_does_not_reconstruct_matrices(components, monkeypatch):
    reference, candidate, batch, other, _ = paired_models(components=components)
    expected = reference.training_step(batch, 0)["loss"]
    gradients = torch.autograd.grad(expected, tuple(reference.parameters()), allow_unused=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("dense reconstruction in coefficient-only training")

    for module in candidate.modules():
        if hasattr(module, "coefficients_to_block"):
            monkeypatch.setattr(module, "coefficients_to_block", forbidden)
    actual = candidate.training_step(other, 0)["loss"]
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    for left, right in zip(gradients, torch.autograd.grad(actual, tuple(candidate.parameters()), allow_unused=True)):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(right, left, rtol=1e-10, atol=1e-12)


def adam_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss(model, batch).backward()
    optimizer.step()


def test_adam_history_resume_and_off_on(tmp_path):
    reference, candidate, batch, _, _ = paired_models(coefficient_training=False)
    optimizer = torch.optim.Adam(reference.parameters(), lr=0.001, weight_decay=0.017)
    adam_step(reference, optimizer, batch)
    path = tmp_path / "history.pt"
    torch.save({"model": reference.state_dict(), "optimizer": optimizer.state_dict()}, path)
    restored = torch.load(path, weights_only=True)
    candidate.load_state_dict(restored["model"], strict=True)
    resumed = copy.deepcopy(candidate)
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.017) for model in (candidate, resumed)]
    for opt in optimizers:
        opt.load_state_dict(copy.deepcopy(restored["optimizer"]))
    adam_step(reference, optimizer, batch)
    for model, opt in zip((candidate, resumed), optimizers):
        adam_step(model, opt, batch)
        for left, right in zip(reference.parameters(), model.parameters()):
            torch.testing.assert_close(right, left, rtol=1e-10, atol=1e-12)
        expected, actual = optimizer.state_dict(), opt.state_dict()
        assert actual["param_groups"] == expected["param_groups"]
        assert actual["state"].keys() == expected["state"].keys()
        for key, state in expected["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(actual["state"][key][name], value, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("parity", [1, -1])
def test_fused_block_o3_and_endpoint_exchange(parity):
    from e3nn import o3
    from graph2mat.bindings.e3nn.modules.tests.test_homonuclear_fusion import make_block
    block = make_block(basis="1x0e+1x1o+1x2e", components=2, fuse_homonuclear=True)
    source = block.operation.tensor_products[0].irreps_in1
    a, b = (torch.randn(3, source.dim) for _ in range(2))
    rotation = parity * o3.rand_matrix()
    input_d = source.D_from_matrix(rotation)
    output_d = block.i_irreps.D_from_matrix(rotation)
    matrix = block(node_feats=(a, b))
    transformed = block(node_feats=(a @ input_d.T, b @ input_d.T))
    expected = torch.einsum("ij,njkc,lk->nilc", output_d, matrix, output_d)
    torch.testing.assert_close(transformed, expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(block(node_feats=(b, a)), matrix.transpose(1, 2), rtol=1e-10, atol=1e-12)


def test_scripted_tp_and_linear_accept_projected_weights():
    from graph2mat.bindings.e3nn.modules.tests.test_homonuclear_fusion import make_block
    block = make_block(fuse_homonuclear=True)
    for layer in [*block.operation.tensor_products, *block.operation.linears]:
        scripted = torch.jit.script(layer)
        a = torch.randn(3, layer.irreps_in1.dim if hasattr(layer, "irreps_in1") else layer.irreps_in.dim)
        args = (a, a * 2) if hasattr(layer, "irreps_in1") else (a,)
        weight = block.operation._projected_weight(layer)
        torch.testing.assert_close(scripted(*args, weight=weight), layer(*args, weight=weight))


def test_lightning_checkpoint_on_to_off_and_legacy(tmp_path):
    from graph2mat.tools.lightning.tests.test_coefficient_matrix_training import data_module_for, cpu_trainer
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    reference, candidate, batch, _, samples = paired_models(coefficient_training=False)
    trainer = cpu_trainer()
    module = data_module_for(candidate, samples)
    trainer.fit(candidate, datamodule=module)
    path = tmp_path / "fusion.ckpt"
    trainer.save_checkpoint(path)
    restored = LitMACEMatrixModel.load_from_checkpoint(path, weights_only=False,
                                                      edge_block_readout_kwargs={})
    torch.testing.assert_close(restored(batch.clone()), candidate(batch.clone()), rtol=1e-10, atol=1e-12)
    restored.load_state_dict(reference.state_dict(), strict=True)
    assert not any(getattr(m, "_fused_homonuclear", False) for m in restored.modules())
    module.teardown(None)


@pytest.fixture
def guarded_gpu(double_default):
    if os.environ.get("G2M_GPU_GUARDED") != "1" or not torch.cuda.is_available():
        pytest.skip("requires the opt-in GPU supervisor")
    torch.set_default_dtype(torch.float32)
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
def test_cuda_fused_outputs_loss_and_gradients(guarded_gpu, precision, monkeypatch):
    from graph2mat.tools.lightning.tests.test_coefficient_cuda_training import autocast
    reference, candidate, cpu, other, _ = paired_models(components=2, dtype=torch.float32, readout_fp32=True)
    for model in (reference, candidate):
        model.cuda()
        monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    batch = reference.transfer_batch_to_device(cpu, "cuda", 0)
    other = candidate.transfer_batch_to_device(other, "cuda", 0)
    rows = []
    eligible = [m for m in candidate.modules() if getattr(m, "_fused_homonuclear", False)]
    handles = [m.operation.tensor_products[0].register_forward_pre_hook(
        lambda module, args: rows.append(args[0].shape[0])) for m in eligible]
    with autocast(precision):
        expected_out, actual_out = reference(batch.clone()), candidate(other.clone())
        expected, actual = reference.training_step(batch, 0)["loss"], candidate.training_step(other, 0)["loss"]
    for handle in handles:
        handle.remove()
    tolerance = (5e-5, 5e-6) if precision == "32-true" else (1e-2, 1e-3)
    errors = {}
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(actual_out[key], expected_out[key], rtol=tolerance[0], atol=tolerance[1])
        errors[key] = float((actual_out[key] - expected_out[key]).detach().abs().max())
    torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
    left = torch.autograd.grad(expected, tuple(reference.parameters()), allow_unused=True)
    right = torch.autograd.grad(actual, tuple(candidate.parameters()), allow_unused=True)
    for (name, _), a, b in zip(reference.named_parameters(), left, right):
        assert (a is None) == (b is None), name
        if a is not None:
            torch.testing.assert_close(b, a, rtol=tolerance[0], atol=tolerance[1], msg=name)
            errors[name] = float((a - b).detach().abs().max())
    assert eligible and rows
    assert not any(m._homonuclear_rejection for m in eligible)
    print("FUSION_NUMERIC", json.dumps({"precision": precision, "max_abs_errors": errors}))


@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
def test_cuda_fused_trainer_resume(guarded_gpu, precision, tmp_path, monkeypatch):
    from graph2mat.tools.lightning.tests import test_coefficient_cuda_training as existing
    original = existing.training_inputs

    def fused_inputs(*args, **kwargs):
        return original(*args, edge_block_readout_kwargs={"fuse_homonuclear": True}, **kwargs)

    monkeypatch.setattr(existing, "training_inputs", fused_inputs)
    existing.test_cuda_trainer_fit_validation_and_checkpoint_resume(precision, tmp_path)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_pure_low_precision_fallback(guarded_gpu, dtype):
    from graph2mat.bindings.e3nn.modules.tests.test_homonuclear_fusion import make_block
    block = make_block(dtype=torch.float32, fuse_homonuclear=True).cuda().to(dtype)
    size = block.operation.tensor_products[0].irreps_in1.dim
    pair = (torch.randn(3, size, device="cuda", dtype=dtype), torch.randn(3, size, device="cuda", dtype=dtype))
    rows = []
    hook = block.operation.tensor_products[0].register_forward_pre_hook(lambda module, args: rows.append(args[0].shape[0]))
    with pytest.warns(UserWarning, match="uncertified dtype"):
        block(node_feats=pair)
    hook.remove()
    assert rows == [6]


def test_optional_compile_no_cudagraphs_matches(guarded_gpu):
    import time
    started = time.perf_counter()
    reference, candidate, cpu, other, _ = paired_models(dtype=torch.float32, coefficient_training=False,
                                                      torch_compile=True,
                                                      torch_compile_mode="max-autotune-no-cudagraphs")
    reference.cuda()
    candidate.cuda()
    for model in (reference, candidate):
        model.model.compile_submodules(mode=model.hparams.torch_compile_mode)
        assert any(getattr(m, "_compiled_call_impl", None) is not None for m in model.modules())
    batch = reference.transfer_batch_to_device(cpu, "cuda", 0)
    other = candidate.transfer_batch_to_device(other, "cuda", 0)
    expected, actual = loss(reference, batch), loss(candidate, other)
    expected.backward()
    actual.backward()
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print("FUSION_COMPILE_SETUP_SECONDS", time.perf_counter() - started)
    expected, actual = loss(reference, batch), loss(candidate, other)
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
    expected.backward()
    actual.backward()
    for a, b in zip(reference.parameters(), candidate.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-5, atol=5e-6)
