"""A shared MACE backbone exposes semantic targets without reading references."""

from copy import deepcopy

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import graph2mat.models.mace as mace_module
from graph2mat.bindings.torch.data.tests.test_multitarget_dataset import bundle, processor
from graph2mat.bindings.torch import TorchBasisMatrixDataset
from graph2mat.core.data.tests.test_multitarget import TARGETS
from graph2mat.tools.lightning.data import MatrixDataModule, TypeLayoutCollate
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def model_for(p, **options):
    kwargs = dict(basis_table=p.basis_table, symmetric_matrix=p.symmetric_matrix,
        hidden_irreps="2x0e+2x1o", edge_hidden_irreps="2x0e+2x1o", num_bessel=2,
        max_ell=1, num_interactions=2, correlation=1,
        matrix_targets=list(p.matrix_target_layout.targets), matrix_target_spin=p.matrix_target_spin,
        matrix_target_weights={t: 1. for t in p.matrix_target_layout.targets},
        matrix_target_losses={t: {"name": "elementwise_mse"} for t in p.matrix_target_layout.targets})
    kwargs.update(options)
    return LitMACEMatrixModel(**kwargs).double()


def batch_for(p):
    data = TorchBasisMatrixDataset([bundle(2, p.matrix_target_spin), bundle(3, p.matrix_target_spin)], p)
    return TypeLayoutCollate(Batch.from_data_list, p.basis_table,
                              symmetric=p.symmetric_matrix)([data[0], data[1]])


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("options", [{"lean_mace": False}, {"lean_mace": True},
    {"gradient_checkpointing": True}, {"readout_per_interaction": True},
    {"readout_per_interaction": True, "readout_interaction_aggregation": "mean"}])
def test_named_views_share_one_backbone(monkeypatch, spin, options):
    p = processor(spin)
    model = model_for(p, **options)
    calls = []
    if model.model.lean_mace:
        original = mace_module.mace_node_feats
        def observe(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        monkeypatch.setattr(mace_module, "mace_node_feats", observe)
    else:
        model.model.mace.register_forward_hook(lambda *args: calls.append(1))
    out = model(batch_for(p))
    assert len(calls) == 1
    assert set(out["matrix_predictions"]) == set(TARGETS)
    for target, columns in p.matrix_target_layout.slices.items():
        named = out["matrix_predictions"][target]
        torch.testing.assert_close(named["node_labels"], out["node_labels"][:, columns], rtol=0, atol=0)
        torch.testing.assert_close(named["edge_labels"], out["edge_labels"][:, columns], rtol=0, atol=0)
        term = named["node_labels"].square().sum() + named["edge_labels"].square().sum()
        model.zero_grad(set_to_none=True)
        # Reentrant checkpointing deliberately supports backward(), not autograd.grad().
        term.backward(retain_graph=True)
        gradients = [parameter.grad for parameter in model.model.mace.parameters()]
        assert all(torch.isfinite(g).all() for g in gradients if g is not None)
        assert any(g is not None and g.abs().sum() > 0 for g in gradients)


def test_reference_changes_do_not_change_eval_output():
    p = processor()
    model = model_for(p).eval()
    original = batch_for(p)
    changed = deepcopy(original)
    changed.point_labels.fill_(1000.)
    changed.edge_labels.fill_(-500.)
    with torch.no_grad():
        a, b = model(original), model(changed)
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


@pytest.mark.parametrize("option", [{"long_range": True}, {"hamiltonian_local_context": True},
    {"coefficient_training": True}, {"training_stages": [{"max_epochs": 1, "loss": "elementwise_mse"}]}])
def test_single_target_only_options_rejected(option):
    with pytest.raises(ValueError, match="multitarget|matrix_targets"):
        model_for(processor(), **option)


def test_one_cpu_step_checkpoint_and_geometry_predict(tmp_path):
    from graph2mat.bindings.torch.load import load_from_lit_ckpt

    class NamedCallback(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, *args):
            assert set(outputs["matrix_predictions"]) == set(TARGETS)

        def on_validation_batch_end(self, trainer, module, outputs, *args):
            assert set(outputs["matrix_predictions"]) == set(TARGETS)

    p = processor(2)
    model = model_for(p)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    dataset = TorchBasisMatrixDataset([bundle(2, 2), bundle(3, 2)], p)
    dm = MatrixDataModule(basis_table=p.basis_table, out_matrix="hamiltonian",
        sub_point_matrix=False, matrix_targets=list(TARGETS), matrix_target_spin=2,
        train_runs=[dataset[0], dataset[1]], val_runs=[dataset[0]], loader_threads=0)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=1, max_epochs=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, default_root_dir=tmp_path,
        callbacks=[NamedCallback()])
    trainer.fit(model, datamodule=dm)
    assert torch.isfinite(trainer.callback_metrics["val_loss"])
    assert "val/target/density_matrix/loss" in trainer.callback_metrics
    assert any(not torch.equal(value, original[key]) for key, value in model.state_dict().items())
    checkpoint_path = tmp_path / "joint.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    saved = torch.load(checkpoint_path, weights_only=False)
    assert saved["matrix_target_descriptor"] == p.matrix_target_layout.to_dict()
    geometry_data = TorchBasisMatrixDataset([bundle(2, 2)["overlap"].geometry], p, load_labels=False)[0]
    geometry_batch = TypeLayoutCollate(Batch.from_data_list, p.basis_table)([geometry_data])
    model.eval()
    with torch.no_grad():
        expected = model(geometry_batch)
    for as_torch in (False, True):
        loaded, restored_processor = load_from_lit_ckpt(checkpoint_path, as_torch=as_torch)
        assert restored_processor.matrix_target_layout == p.matrix_target_layout
        loaded.eval()
        with torch.no_grad():
            actual = loaded(geometry_batch)
        for key in ("node_labels", "edge_labels"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    dm.teardown(None)


def test_epoch_metrics_independent_of_batch_partition(tmp_path):
    p = processor()
    dataset = TorchBasisMatrixDataset([bundle(n) for n in (2, 3, 2)], p)
    model = model_for(p).eval()
    values = []
    for batch_size in (3, 2):
        loader = DataLoader(dataset, batch_size=batch_size,
            collate_fn=TypeLayoutCollate(Batch.from_data_list, p.basis_table))
        trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmp_path)
        values.append(trainer.validate(model, dataloaders=loader, verbose=False)[0])
    assert values[0]["val_loss"] == pytest.approx(values[1]["val_loss"], rel=1e-10, abs=1e-10)
    expected = sum(values[0][f"val/target/{target}/loss"] for target in TARGETS)
    assert values[0]["val_loss"] == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_checkpoint_same_width_wrong_semantics_rejected():
    p = processor(targets=("hamiltonian", "overlap", "density_matrix"))
    model = model_for(p)
    checkpoint = {"hyper_parameters": dict(model.hparams), "basis_table": p.basis_table}
    model.on_save_checkpoint(checkpoint)
    changed = deepcopy(checkpoint)
    descriptor = changed["matrix_target_descriptor"]
    descriptor["version"] = 2
    with pytest.raises(ValueError, match="descriptor|version|layout"):
        model.on_load_checkpoint(changed)
    from graph2mat.core.data.multitarget import MatrixTargetLayout

    changed = deepcopy(checkpoint)
    changed["matrix_target_descriptor"] = MatrixTargetLayout(
        ("hamiltonian", "overlap", "energy_density_matrix"), 1).to_dict()
    with pytest.raises(ValueError, match="descriptor|layout|semantic"):
        model.on_load_checkpoint(changed)


def test_all_training_samples_preflight_before_first_update(tmp_path):
    p = processor()
    model = model_for(p)
    dataset = TorchBasisMatrixDataset([bundle(2), bundle(3)], p)
    good, bad = dataset[0], dataset[1]
    bad.point_labels[:, 2] = bad.edge_labels[:, 2] = float("nan")
    bad.matrix_known_nodes[:, 2] = bad.matrix_known_edges[:, 2] = False
    loader = DataLoader([good, bad], batch_size=1,
        collate_fn=TypeLayoutCollate(Batch.from_data_list, p.basis_table))
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, limit_train_batches=1, limit_val_batches=0,
        default_root_dir=tmp_path)
    with pytest.raises(ValueError, match="density_matrix.*observ|sample 1"):
        trainer.fit(model, train_dataloaders=loader)
    assert trainer.global_step == 0


def test_legacy_numerics_keys_and_checkpoint_without_new_fields(tmp_path):
    import hashlib
    from graph2mat.bindings.torch import TorchBasisMatrixData
    from graph2mat.bindings.torch.load import load_from_lit_ckpt
    from graph2mat.core.data.tests.physical_fixtures import prepared_sample

    torch.manual_seed(766)
    data, table = prepared_sample(target="hamiltonian", data_cls=TorchBasisMatrixData)
    kwargs = dict(basis_table=table, symmetric_matrix=True, hidden_irreps="2x0e+2x1o",
                  edge_hidden_irreps="2x0e+2x1o", num_bessel=2, max_ell=1,
                  num_interactions=2, correlation=1)
    model = LitMACEMatrixModel(**kwargs).double().eval()
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        out = model(batch)
    assert "matrix_predictions" not in out
    # Literal outputs/key digest captured before the multitarget MatrixMACE changes.
    torch.testing.assert_close(out["node_labels"], torch.tensor([.039056375364447476] * 2), rtol=0, atol=1e-14)
    torch.testing.assert_close(out["edge_labels"], torch.tensor([-.0073608053886730015]), rtol=0, atol=1e-14)
    digest = hashlib.sha256("\n".join(sorted(model.state_dict())).encode()).hexdigest()
    assert digest == "3829a7bf849a64ee053604cfbaaba2c5274fb27986d2da3a5777364fa90a8c98"
    checkpoint = dict(state_dict=model.state_dict(), hyper_parameters=kwargs, basis_table=table,
        datamodule_hyper_parameters=dict(out_matrix="hamiltonian", sub_point_matrix=False,
                                        symmetric_matrix=True),
        **{"pytorch-lightning_version": pl.__version__})
    path = tmp_path / "legacy.ckpt"
    torch.save(checkpoint, path)
    loaded, p = load_from_lit_ckpt(path)
    assert p.matrix_target_layout is None
    loaded.eval()
    with torch.no_grad():
        actual = loaded(batch)
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(actual[key], out[key], rtol=0, atol=0)


def test_epoch_physics_and_component_masks_keep_separate_denominators(tmp_path):
    import numpy as np
    from scipy.sparse import csr_array
    from graph2mat import MatrixDataProcessor, AtomicTableWithEdges
    from graph2mat.bindings.torch import TorchBasisMatrixData
    from graph2mat.core.data.multitarget import prepare_multitarget_data
    from graph2mat.core.data.tests.physical_fixtures import analytic_matrices, periodic_matrices

    matrices = [analytic_matrices(), periodic_matrices(), analytic_matrices(fractional=True)]
    p = MatrixDataProcessor(basis_table=AtomicTableWithEdges(matrices[0]["overlap"].atoms.atom),
        out_matrix="hamiltonian", symmetric_matrix=True, sub_point_matrix=False,
        matrix_targets=list(TARGETS), matrix_target_spin=1)
    samples = []
    for index, source in enumerate(matrices):
        masks = {"density_matrix": [csr_array(np.eye(2, dtype=bool))]} if index == 2 else None
        samples.append(prepare_multitarget_data(p, **source, observation_masks=masks,
            physical_labels={"hamiltonian": ["energy"], "energy_density_matrix": ["energy"]},
            data_cls=TorchBasisMatrixData))
    losses = {t: {"name": "elementwise_mse", "kwargs": {}} for t in TARGETS}
    losses["hamiltonian"]["kwargs"]["energy_weight"] = 2.
    losses["energy_density_matrix"]["kwargs"]["energy_weight"] = 3.
    model = model_for(p, matrix_target_losses=losses).eval()
    results = []
    for batch_size in (3, 2):
        loader = DataLoader(samples, batch_size=batch_size,
            collate_fn=TypeLayoutCollate(Batch.from_data_list, p.basis_table))
        trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmp_path)
        results.append(trainer.validate(model, dataloaders=loader, verbose=False)[0])
    for key in ("val_loss", "val/target/density_matrix/matrix_count",
                "val/target/hamiltonian/physical_energy_mse",
                "val/target/energy_density_matrix/physical_energy_mse"):
        assert results[0][key] == pytest.approx(results[1][key], rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("options", [{"basis_grouping": "max"}, {"pair_envelope": True},
                                     {"simplify_node_irreps": True}])
def test_existing_readout_options_keep_named_outputs(options):
    p = processor()
    out = model_for(p, **options)(batch_for(p))
    assert set(out["matrix_predictions"]) == set(TARGETS)
    assert torch.isfinite(out["node_labels"]).all()


def test_lowrank_multitarget_and_existing_sk_component_limit():
    from graph2mat.bindings.e3nn import E3nnLowRankEdgeBlock, E3nnSlaterKosterEdgeBlock

    p = processor(2)
    out = model_for(p, edge_block_readout=E3nnLowRankEdgeBlock,
                    edge_block_readout_kwargs={"rank": 2})(batch_for(p))
    assert out["matrix_predictions"]["density_matrix"]["edge_labels"].shape[1] == 2
    with pytest.raises(ValueError, match="SK supports"):
        model_for(p, edge_block_readout=E3nnSlaterKosterEdgeBlock,
                  edge_block_readout_kwargs={"rank": 2})


def test_output_scaleshift_per_component_and_fp32_readout():
    p = processor()
    model = model_for(p, output_scaleshift=True)
    batch = batch_for(p)
    model.model.matrix_readouts.fit_output_scaleshift([batch], p.basis_table)
    assert model.model.matrix_readouts.output_scaleshift_fitted
    assert torch.isfinite(model(batch)["node_labels"]).all()
    model = model_for(p, readout_fp32=True).float()
    batch = batch_for(p).apply(lambda value: value.float() if value.is_floating_point() else value)
    out = model(batch)
    assert out["node_labels"].dtype == torch.float32
    assert set(out["matrix_predictions"]) == set(TARGETS)


@pytest.mark.parametrize("finite_placeholder", [False, True])
def test_output_scaleshift_rejects_partial_labels_without_mutation(finite_placeholder):
    import numpy as np
    from scipy.sparse import csr_array
    from graph2mat.bindings.torch import TorchBasisMatrixData
    from graph2mat.core.data.multitarget import prepare_multitarget_data

    p = processor(2)
    sample = prepare_multitarget_data(p, **bundle(2, 2), data_cls=TorchBasisMatrixData,
        observation_masks={"density_matrix": [csr_array(np.eye(2, dtype=bool)),
                                             csr_array(np.ones((2, 2), dtype=bool))]})
    batch = TypeLayoutCollate(Batch.from_data_list, p.basis_table)([sample])
    if finite_placeholder:
        batch.edge_labels = batch.edge_labels.nan_to_num()
    model = model_for(p, output_scaleshift=True)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    # A later incomplete batch must not leave an earlier block's buffers fitted.
    with pytest.raises(ValueError, match="output_scaleshift.*fully observed"):
        model.model.matrix_readouts.fit_output_scaleshift([batch_for(p), batch], p.basis_table)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    assert not model.model.matrix_readouts.output_scaleshift_fitted
    out = model(batch)
    assert torch.isfinite(out["node_labels"]).all() and torch.isfinite(out["edge_labels"]).all()
    loss, _ = model.loss_fn(nodes_pred=out["node_labels"], nodes_ref=batch.point_labels,
        edges_pred=out["edge_labels"], edges_ref=batch.edge_labels, batch=batch, basis_table=p.basis_table)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.model.mace.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_real_cli_multitarget_checkpoint_predict_without_labels(tmp_path):
    from pathlib import Path
    from graph2mat.tools.lightning.cli import OrbitalMatrixCLI

    p = processor(2)
    model = model_for(p).eval()
    checkpoint = {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
        "basis_table": p.basis_table, "pytorch-lightning_version": pl.__version__,
        "datamodule_hyper_parameters": dict(basis_table=p.basis_table, out_matrix="hamiltonian",
            sub_point_matrix=False, symmetric_matrix=True, matrix_targets=list(TARGETS),
            matrix_target_spin=2, n_matrix_components=7, loader_threads=0)}
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "joint.ckpt"
    torch.save(checkpoint, path)
    geometry_path = tmp_path / "geometry.xyz"
    bundle(2, 2)["overlap"].geometry.write(geometry_path)
    # The documented example lives in the repository, not in an installed package.
    example = next((root / "docs/examples/multitarget_predict.yaml" for root in Path(__file__).resolve().parents
                    if (root / "docs/examples/multitarget_predict.yaml").is_file()), None)
    if example is None:
        pytest.skip("docs/examples is not available outside a source checkout")
    cli = OrbitalMatrixCLI(model_class=LitMACEMatrixModel, datamodule_class=MatrixDataModule,
        save_config_callback=None, args=["predict", "--config", str(example),
            "--ckpt_path", str(path),
            "--data.root_dir", str(tmp_path), "--data.predict_structs", geometry_path.name,
            "--trainer.accelerator=cpu",
            "--trainer.devices=1", "--trainer.logger=false", "--trainer.enable_checkpointing=false",
            "--trainer.enable_progress_bar=false", "--trainer.enable_model_summary=false",
            "--trainer.callbacks=[]"])
    assert cli.datamodule.data_processor.matrix_target_layout == p.matrix_target_layout
    assert getattr(cli.datamodule.predict_dataset[0], "point_labels", None) is None
    assert cli.trainer.predict_loop.batch_progress.total.completed == 1
    batch = next(iter(cli.datamodule.predict_dataloader()))
    with torch.no_grad():
        expected, actual = model(batch), cli.model.eval()(batch)
    torch.testing.assert_close(actual["node_labels"], expected["node_labels"], rtol=0, atol=0)
    cli.datamodule.teardown(None)


def test_compile_eager_preserves_named_outputs_gradients_and_keys(monkeypatch):
    p = processor()
    model = model_for(p, torch_compile=True)
    reference = deepcopy(model)
    original_compile = torch.nn.Module.compile
    def compile_cpu(module, **kwargs):
        return original_compile(module, **kwargs, backend="eager")
    monkeypatch.setattr(torch.nn.Module, "compile", compile_cpu)
    keys = set(model.state_dict())
    model.model.compile_submodules()
    assert set(model.state_dict()) == keys
    a, b = reference(batch_for(p)), model(batch_for(p))
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(a[key], b[key], rtol=1e-10, atol=1e-12)
    assert set(b["matrix_predictions"]) == set(TARGETS)
    for current, output in ((reference, a), (model, b)):
        (output["node_labels"].square().sum() + output["edge_labels"].square().sum()).backward()
    for parameter, expected in zip(model.parameters(), reference.parameters()):
        if expected.grad is not None:
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-10, atol=1e-12)


def test_empty_edges_and_optimizer_only_stage_are_valid():
    p = processor()
    model = model_for(p, training_stages=[{"max_epochs": 1, "optim_lr": .002}])
    data = TorchBasisMatrixDataset([bundle(1)], p)[0]
    batch = TypeLayoutCollate(Batch.from_data_list, p.basis_table)([data])
    out = model(batch)
    assert out["edge_labels"].shape == (0, 4)
    value, _ = model.loss_fn(nodes_pred=out["node_labels"], nodes_ref=batch.point_labels,
        edges_pred=out["edge_labels"], edges_ref=batch.edge_labels, batch=batch, basis_table=p.basis_table)
    assert torch.isfinite(value)
    value.backward()
