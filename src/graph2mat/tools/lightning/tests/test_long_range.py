"""Roadmap 8.4: learned charges, Ewald potential and (V_i + V_j) / 2 * S_ij."""

import numpy as np
import pytest
import torch
from e3nn import o3

from graph2mat import BasisConfiguration, BasisTableWithEdges, MatrixDataProcessor, PointBasis, conversions
from graph2mat.models.long_range import COULOMB, ewald_potentials


def test_ewald_madelung_rocksalt():
    """Rocksalt Madelung constant, independent of a rotation and of the batch layout."""
    a = 5.64
    frac = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]])
    positions = torch.tensor(frac * a)
    q = torch.tensor([1.0 if (f * 2).sum() % 2 == 0 else -1.0 for f in frac], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * a
    expected = -1.747565 * COULOMB / (a / 2)
    v = ewald_potentials(q, positions, cell, torch.zeros(8, dtype=torch.long), smearing=0.3)
    np.testing.assert_allclose(v.numpy(), expected * q.numpy(), rtol=1e-6)
    rot = torch.tensor(np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0])
    two = ewald_potentials(torch.cat([q, q]), torch.cat([positions, positions @ rot.T]),
                           torch.cat([cell, cell @ rot.T]), torch.tensor([0] * 8 + [1] * 8), smearing=0.3)
    np.testing.assert_allclose(two.numpy(), np.tile(expected * q.numpy(), 2), rtol=1e-6)


def _model_and_batch(long_range, n_components: int = 2, n_outputs: int = 1):
    mace_modules = pytest.importorskip("mace.modules")
    from torch_geometric.data import Batch
    from graph2mat.models.mace import MatrixMACE

    basis = [PointBasis("A", R=2.5, basis="0e + 1o", basis_convention="cartesian"),
             PointBasis("B", R=2.0, basis="2x0e + 1o", basis_convention="cartesian")]
    table = BasisTableWithEdges(basis)
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    conv = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")
    rng = np.random.default_rng(3)
    data = []
    for n in (4, 3):
        config = BasisConfiguration(point_types=(["A", "B"] * 2)[:n], positions=rng.uniform(0, 4.0, (n, 3)),
                                    basis=basis, cell=np.diag([4.5, 5.0, 5.5]), pbc=(True, True, True))
        d = conv(config, data_processor=processor)
        n_nodes = int(table.point_block_pointer(d.point_types.numpy())[-1])
        n_edges = int(table.edge_block_size[d.edge_types.numpy()[::2]].sum())
        d.point_labels = torch.tensor(rng.normal(size=(n_nodes, n_components)), dtype=torch.get_default_dtype())
        d.edge_labels = torch.tensor(rng.normal(size=(n_edges, n_components)), dtype=torch.get_default_dtype())
        data.append(d)
    batch = Batch.from_data_list(data)
    batch.node_attrs = torch.eye(2)[batch.point_types]
    torch.manual_seed(0)
    mace = mace_modules.MACE(
        r_max=5.0, num_bessel=4, num_polynomial_cutoff=6, max_ell=2,
        interaction_cls=mace_modules.RealAgnosticResidualInteractionBlock,
        interaction_cls_first=mace_modules.RealAgnosticResidualInteractionBlock,
        num_interactions=2, num_elements=2, hidden_irreps=o3.Irreps("4x0e+4x1o"),
        MLP_irreps=o3.Irreps("1x0e"), atomic_energies=torch.zeros(2), avg_num_neighbors=3.0,
        atomic_numbers=torch.arange(2), correlation=2, gate=None,
    )
    torch.manual_seed(1)
    model = MatrixMACE(mace=mace, unique_basis=table, symmetric=True, edge_hidden_irreps=o3.Irreps("2x0e+2x1o"),
                       n_matrix_components=n_outputs, long_range=long_range, long_range_smearing=1.0)
    return model, batch, processor, table


def test_long_range_adds_potential_times_overlap():
    """Zero charges at initialization (same output as without it); with charges, the output
    changes by exactly (V_i + V_j) / 2 * S_ij on every orbital pair, S being the labels' last
    component, and the charges of each structure add up to zero."""
    plain, batch, processor, table = _model_and_batch(False)
    model, _, _, _ = _model_and_batch(True)
    base = plain(batch)
    out = model(batch)
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(out[key], base[key])
    torch.nn.init.normal_(model.charge_readout.weight)
    out = model(batch)
    q, v = out["node_charges"].detach(), out["node_potential"].detach()
    assert q.abs().max() > 1e-3
    np.testing.assert_allclose(torch.zeros(2).index_add(0, batch.batch, q).numpy(), 0, atol=1e-5)
    for s in range(2):
        dense = lambda nodes, edges: processor.matrix_from_data(
            batch[s], predictions={"node_labels": nodes, "edge_labels": edges}, out_format="numpy")
        pick = lambda t: t.detach().reshape(t.shape[0], -1)[:, 0]
        nodes_mask = torch.repeat_interleave(batch.batch, torch.as_tensor(table.point_block_size[batch.point_types.numpy()]))
        edge_struct = torch.repeat_interleave(
            batch.batch[batch.edge_index[0, ::2]], torch.as_tensor(table.edge_block_size[np.abs(batch.edge_types.numpy()[::2])]))
        sel_n, sel_e = nodes_mask == s, edge_struct == s
        diff = dense(pick(out["node_labels"])[sel_n] - pick(base["node_labels"])[sel_n],
                     pick(out["edge_labels"])[sel_e] - pick(base["edge_labels"])[sel_e])
        overlap = dense(batch.point_labels[sel_n, -1], batch.edge_labels[sel_e, -1])
        atoms = np.repeat(np.arange(int(batch.ptr[s + 1] - batch.ptr[s])),
                          table.basis_size[batch[s].point_types.numpy()])
        v_s = v[batch.batch == s].numpy()
        v_orb = v_s[atoms]
        n = len(atoms)
        expected = 0.5 * (v_orb[:, None] + np.tile(v_orb, diff.shape[1] // n)[None, :]) * overlap
        np.testing.assert_allclose(diff, expected, atol=1e-5)


def test_long_range_needs_the_overlap():
    model, batch, _, _ = _model_and_batch(True, n_components=1)
    with pytest.raises(ValueError, match="overlap"):
        model(batch)


def test_equilibrate_charges_solves_qeq():
    """The charges add up to Q and make chi + J q + A q the same at every atom (the
    stationarity condition with its Lagrange multiplier), structure by structure."""
    from graph2mat.models.long_range import equilibrate_charges, ewald_matrix

    rng = np.random.default_rng(5)
    cells = [np.diag([5.0, 6.0, 7.0]), np.array([[4.0, 0, 0], [1.0, 5.0, 0], [0, 0.5, 6.0]])]
    sizes, totals = [5, 4], torch.tensor([0.0, 1.0], dtype=torch.float64)
    positions = torch.tensor(np.concatenate([rng.uniform(0, 4, (n, 3)) for n in sizes]))
    cell = torch.tensor(np.concatenate(cells))
    batch = torch.tensor([0] * sizes[0] + [1] * sizes[1])
    chi = torch.tensor(rng.normal(size=sum(sizes)))
    hardness = torch.tensor(rng.uniform(5, 15, sum(sizes)))
    q = equilibrate_charges(chi, hardness, positions, cell, batch, totals, smearing=1.0)
    for s in range(2):
        m = batch == s
        assert abs(float(q[m].sum() - totals[s])) < 1e-10
        a = ewald_matrix(positions[m], cell[3 * s:3 * s + 3], smearing=1.0)
        mu = chi[m] + hardness[m] * q[m] + a @ q[m]
        np.testing.assert_allclose(mu.numpy(), float(mu.mean()), atol=1e-8)
    assert torch.all(equilibrate_charges(torch.zeros_like(chi), hardness, positions, cell, batch,
                                         torch.zeros(2, dtype=torch.float64), smearing=1.0) == 0)


def test_qeq_mode_starts_neutral_and_takes_the_total_charge():
    """long_range='qeq': same output as without it for neutral structures at the start; a
    total charge (SIESTA NetCharge, > 0 when electrons are removed) gives electronic charges
    that add up to minus it (audit M18), and changes H."""
    plain, batch, _, _ = _model_and_batch(False)
    model, _, _, _ = _model_and_batch("qeq")
    base = plain(batch)
    out = model(batch)
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(out[key], base[key])
    batch.total_charge = torch.tensor([1.0, -2.0])
    out = model(batch)
    sums = torch.zeros(2).index_add(0, batch.batch, out["node_charges"].detach())
    np.testing.assert_allclose(sums.numpy(), [-1.0, 2.0], atol=1e-5)
    assert (out["node_labels"] - base["node_labels"]).abs().max() > 1e-4


def test_long_range_shifts_every_hamiltonian_column_and_not_the_overlap():
    """Spin-polarized h_and_overlap labels [H_up, H_down, S]: both spin channels get the same
    (V_i + V_j) / 2 * S_ij shift and a predicted S column none."""
    plain, batch, _, _ = _model_and_batch(False, n_components=3, n_outputs=3)
    model, _, _, _ = _model_and_batch(True, n_components=3, n_outputs=3)
    torch.nn.init.normal_(model.charge_readout.weight)
    base, out = plain(batch), model(batch)
    for key in ("node_labels", "edge_labels"):
        diff = (out[key] - base[key]).detach()
        assert diff[:, 0].abs().max() > 1e-4
        torch.testing.assert_close(diff[:, 1], diff[:, 0])
        torch.testing.assert_close(diff[:, 2], torch.zeros_like(diff[:, 2]))


def test_lightning_long_range_requires_h_and_overlap_data():
    from types import SimpleNamespace
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    for policy in ("spin_h_only", "h_only"):
        fake = SimpleNamespace(hparams={"long_range": True, "spin_orbit": False, "magnetic_moments": None,
                                        "edge_cutoff_padding": 0.0},
                               trainer=SimpleNamespace(datamodule=SimpleNamespace(
                                   matrix_component_policy=policy, edge_cutoff_padding=0.0, magnetic_moments=None)))
        with pytest.raises(ValueError, match="long_range"):
            LitMACEMatrixModel._check_data_agreement(fake)
