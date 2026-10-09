"""Optional 7.1 selection must survive full MACE inference/training/checkpoints."""

from copy import deepcopy

import numpy as np
import pytest
import pytorch_lightning as pl
import sisl
import torch
from e3nn import o3
from scipy.linalg import block_diag

from graph2mat import (
    AtomicTableWithEdges,
    BasisConfiguration,
    MatrixDataProcessor,
    OrbitalConfiguration,
)
from graph2mat.bindings.e3nn import (
    E3nnGraph2Mat,
    E3nnLowRankEdgeBlock,
    E3nnLinearEdgeBlock,
    HamiltonianLocalContext,
)
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.core.modules.graph2mat import LAYOUT_ATTR
from graph2mat.models.mace import MatrixMACE
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
from graph2mat.tools.lightning.tests.test_coefficient_matrix_training import (
    data_module_for,
    training_inputs,
)


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def low_rank_inputs(**kwargs):
    return training_inputs(
        edge_block_readout=E3nnLowRankEdgeBlock,
        edge_block_readout_kwargs={"rank": 2},
        **kwargs,
    )


@pytest.mark.parametrize("mode", ["single", "sum"])
@pytest.mark.parametrize("reuse_nodes", [False, True])
def test_low_rank_rejects_unenveloped_hamiltonian_context(mode, reuse_nodes):
    # Context species/node channels survive the cutoff and would reopen radial gates.
    with pytest.raises(
        ValueError, match="E3nnLowRankEdgeBlock.*hamiltonian_local_context"
    ):
        low_rank_inputs(
            mode=mode,
            coefficient_training=False,
            preprocessing_edges_reuse_nodes=reuse_nodes,
            hamiltonian_local_context=True,
            pair_envelope=True,
        )


def test_direct_matrix_mace_rejects_context_with_custom_low_rank_readout():
    # The guard must inspect actual operations, not just the Lightning selector.
    class LowRankReadout(E3nnGraph2Mat):
        def __init__(self, **kwargs):
            super().__init__(edge_operation=E3nnLowRankEdgeBlock, **kwargs)

    model, batch, _ = training_inputs(
        coefficient_training=False, hamiltonian_local_context=True
    )
    assert torch.isfinite(model(batch.clone())["edge_labels"]).all()
    with pytest.raises(
        ValueError, match="E3nnLowRankEdgeBlock.*hamiltonian_local_context"
    ):
        MatrixMACE(
            mace=deepcopy(model.model.mace),
            graph2mat_cls=LowRankReadout,
            unique_basis=model.basis_table,
            edge_hidden_irreps=o3.Irreps("2x0e+2x1o+2x2e"),
            hamiltonian_local_context=HamiltonianLocalContext(num_types=2, r_max=4),
        )


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize(
    "grouping,mode", [("point_type", "single"), ("max", "sum"), ("basis_shape", "mean")]
)
def test_low_rank_full_mace_coefficients_loss_and_layout(components, grouping, mode):
    # Catches wrong selection/forwarded rank, radial plumbing, C2 reshaping and source routing.
    model, batch, _ = low_rank_inputs(
        components=components,
        grouping=grouping,
        mode=mode,
        preprocessing_edges_reuse_nodes=True,
    )
    readouts = model.model.matrix_readouts
    for readout in (
        readouts if isinstance(readouts, torch.nn.ModuleList) else [readouts]
    ):
        for block in readout.interactions.values():
            assert isinstance(block.operation, E3nnLowRankEdgeBlock)
            assert all(
                mul == 2 for mul, _ in block.operation.tensor_products[0].irreps_in1
            )
    out = model(batch.clone())
    assert out["edge_labels"].shape == batch.edge_labels.shape
    expected = (
        (out["node_labels"] - batch.point_labels).square().sum()
        + 2 * (out["edge_labels"] - batch.edge_labels).square().sum()
    ) / (batch.point_labels.numel() + 2 * batch.edge_labels.numel())
    loss = model.training_step(batch.clone(), 0)["loss"]
    torch.testing.assert_close(loss, expected, rtol=1e-10, atol=1e-12)
    loss.backward()
    # No B-B edge: basis_shape sorts B before A, so its unused operation is 0-0.
    unused = {"point_type": ".(1, 1, 2).", "basis_shape": ".(0, 0, 0).", "max": None}[
        grouping
    ]
    edge_params = [
        param
        for name, param in model.named_parameters()
        if "radial_gates" in name and (unused is None or unused not in name)
    ]
    assert edge_params and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in edge_params
    )
    assert any(p.grad.abs().sum() > 0 for p in edge_params)
    dynamic = batch.clone()
    if hasattr(dynamic, LAYOUT_ATTR):
        object.__delattr__(dynamic, LAYOUT_ATTR)
    actual = model(dynamic)
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(actual[key], out[key], rtol=0, atol=0)


def test_pair_operations_do_not_share_weights_by_default():
    model, batch, _ = low_rank_inputs(coefficient_training=False)
    readout = model.model.matrix_readouts
    before = model(batch.clone())["edge_labels"].detach()
    aa, ab = readout.interactions["(0, 0, 0)"], readout.interactions["(0, 1, 1)"]
    other = deepcopy(ab.operation.state_dict())
    with torch.no_grad():
        for param in aa.operation.parameters():
            param.zero_()
    after = model(batch.clone())["edge_labels"].detach()
    assert not torch.equal(before, after)
    for key, value in ab.operation.state_dict().items():
        torch.testing.assert_close(value, other[key], rtol=0, atol=0)


@pytest.mark.parametrize("candidate", [False, True])
def test_real_fit_checkpoint_and_legacy_default(tmp_path, candidate):
    model, batch, samples = (low_rank_inputs if candidate else training_inputs)(
        coefficient_training=False, optim_lr=1e-3
    )
    initial = deepcopy(model.state_dict())
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    module = data_module_for(model, samples)
    trainer.fit(model, datamodule=module)
    assert trainer.global_step == 2
    if candidate:
        changed = [
            not torch.equal(initial[name], value)
            for name, value in model.state_dict().items()
            if "radial_gates" in name
        ]
        assert changed and any(changed)
    path = tmp_path / "readout.ckpt"
    trainer.save_checkpoint(path)
    if not candidate:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["hyper_parameters"].pop("edge_block_readout_kwargs", None)
        torch.save(checkpoint, path)
    restored = LitMACEMatrixModel.load_from_checkpoint(
        path, map_location="cpu", weights_only=False
    )
    model.eval()
    restored.eval()
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(
            restored(batch.clone())[key], model(batch.clone())[key], rtol=0, atol=0
        )
    for block in restored.model.matrix_readouts.interactions.values():
        assert isinstance(
            block.operation, E3nnLowRankEdgeBlock if candidate else E3nnLinearEdgeBlock
        )
    if not candidate:
        assert not any(
            "radial_gates" in key or "left_projections" in key
            for key in model.state_dict()
        )
    module.teardown(None)


def test_full_periodic_spdf_rotation_inversion_exchange_and_permutation():
    # Physical orbital matrix oracle independent of the low-rank operation.
    def shell(l):
        return [sisl.AtomicOrbital(n=l + 1, l=l, m=m, R=1.7) for m in range(-l, l + 1)]

    a = sisl.Atom("H", orbitals=shell(0) + shell(1))
    b = sisl.Atom("O", orbitals=shell(0) + shell(1) + shell(2) + shell(3))
    table = AtomicTableWithEdges([a, b])
    processor = MatrixDataProcessor(
        basis_table=table, symmetric_matrix=True, sub_point_matrix=False
    )
    torch.manual_seed(2)
    model = LitMACEMatrixModel(
        basis_table=table,
        symmetric_matrix=True,
        hidden_irreps="2x0e+2x1o+2x2e+2x3o",
        edge_hidden_irreps="2x0e+2x1o+2x2e+2x3o",
        num_bessel=3,
        max_ell=3,
        num_interactions=2,
        correlation=1,
        edge_block_readout=E3nnLowRankEdgeBlock,
        edge_block_readout_kwargs={"rank": 2},
    ).eval()
    geom = sisl.Geometry(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [2.0, 0.1, 0.4]],
        atoms=[a, b, a],
        lattice=[4, 20, 20],
    )
    geom.set_nsc([3, 1, 1])

    def predict(geometry):
        data = TorchBasisMatrixData.from_config(
            OrbitalConfiguration.from_geometry(geometry), data_processor=processor
        )
        with torch.no_grad():
            out = model(data)
        return processor.matrix_from_data(data, out, out_format="sisl_DM")

    original = predict(geom)
    matrix = original.tocsr().toarray()
    assert np.abs(matrix).max() > 1e-5
    # Explicit periodic transpose: M(R)=M(-R)^T for each image, including heteropairs.
    for index, offset in enumerate(geom.sc_off):
        opposite = geom.sc_index(-offset)
        np.testing.assert_allclose(
            matrix[:, index * geom.no : (index + 1) * geom.no],
            matrix[:, opposite * geom.no : (opposite + 1) * geom.no].T,
            rtol=1e-10,
            atol=1e-12,
        )
    assert (
        np.max(np.abs(matrix[13:20])) > 1e-6
    )  # O f-orbital rows must not be all zero.
    for angles, sign in [((0.31, 0.72, -0.48), 1), ((0.91, -0.27, 0.63), -1)]:
        rotation = (
            sign
            * o3.angles_to_matrix(*[torch.tensor(value) for value in angles]).numpy()
        )
        rotated = geom.copy()
        rotated.xyz[:] = geom.xyz @ rotation.T
        rotated.cell[:] = geom.cell @ rotation.T
        cob = table.change_of_basis
        orbital = block_diag(
            *[
                table.basis[species]
                .e3nn_irreps.D_from_matrix(torch.as_tensor(cob @ rotation @ cob.T))
                .numpy()
                for species in geom.atoms.species
            ]
        )
        expected = np.concatenate(
            [orbital @ cell @ orbital.T for cell in np.split(matrix, geom.n_s, axis=1)],
            axis=1,
        )
        np.testing.assert_allclose(
            predict(rotated).tocsr().toarray(), expected, rtol=2e-8, atol=2e-10
        )
    permuted = original.sub([2, 1, 0]).tocsr().toarray()
    np.testing.assert_allclose(
        predict(geom.sub([2, 1, 0])).tocsr().toarray(), permuted, rtol=1e-10, atol=1e-12
    )


def test_no_edges_full_model_is_finite():
    model, _, _ = low_rank_inputs(coefficient_training=False)
    # An isolated atom must not enter a reshape with an inferred zero-sized axis.
    processor = MatrixDataProcessor(
        basis_table=model.basis_table, symmetric_matrix=True, sub_point_matrix=False
    )
    config = BasisConfiguration(
        point_types=["A"],
        positions=np.zeros((1, 3)),
        basis=model.basis_table,
        cell=np.eye(3) * 20,
        pbc=(False, False, False),
    )
    sample = TorchBasisMatrixData.from_config(config, data_processor=processor)
    out = model(sample)
    assert out["edge_labels"].numel() == 0
    assert out["node_labels"].numel() > 0 and torch.isfinite(out["node_labels"]).all()
