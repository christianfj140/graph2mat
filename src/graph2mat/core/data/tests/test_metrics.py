import pytest

from graph2mat import OrbitalMatrixMetric


# Arguments that some metrics require (they raise without them).
REQUIRED_KWARGS = {"block_type_mse_per_domain": {"domain_threshold_atoms": 2}}


# Test that all metrics run
@pytest.mark.parametrize(
    "metric",
    [
        metric()
        for metric in OrbitalMatrixMetric.__subclasses__()
        if not getattr(metric, "requires_model_output", False)
        and not getattr(metric, "requires_overlap_labels", False)
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_metric_runs(density_data, density_z_table, metric):
    metric(
        nodes_pred=density_data.point_labels - 0.001,
        nodes_ref=density_data.point_labels,
        edges_pred=density_data.edge_labels,
        edges_ref=density_data.edge_labels,
        batch=density_data,
        basis_table=density_z_table,
        **REQUIRED_KWARGS.get(metric.__class__.__name__, {}),
    )


def test_normalized_density_error_of_exact_prediction_is_zero(
    density_matrix, density_z_table, symmetric
):
    """With sub_point_matrix=True the labels are DM - DM_atomic; the error of an exact
    prediction must not get the atomic density added back."""
    from graph2mat import MatrixDataProcessor, OrbitalConfiguration
    from graph2mat.bindings.torch import TorchBasisMatrixData
    from graph2mat.core.data.matrices import DensityMatrix
    from graph2mat.core.data.metrics import normalized_density_error
    from graph2mat.core.data.sparse import csr_to_block_dict

    dm_block = csr_to_block_dict(
        density_matrix._csr, density_matrix.atoms, nsc=density_matrix.nsc,
        matrix_cls=DensityMatrix,
    )
    density_config = OrbitalConfiguration.from_geometry(
        geometry=density_matrix.geometry, matrix=dm_block
    )

    processor = MatrixDataProcessor(
        basis_table=density_z_table,
        sub_point_matrix=True,
        symmetric_matrix=symmetric,
        out_matrix="density_matrix",
    )
    data = TorchBasisMatrixData.from_config(density_config, data_processor=processor)
    error, _ = normalized_density_error(
        nodes_pred=data.point_labels,
        nodes_ref=data.point_labels,
        edges_pred=data.edge_labels,
        edges_ref=data.edge_labels,
        batch=data,
        basis_table=density_z_table,
        grid_spacing=0.3,
    )
    assert abs(error) < 1e-8


def test_elementwise_mse_edge_weight_counts_edges_twice():
    import torch
    from graph2mat.core.data.metrics import elementwise_mse

    nodes_pred, nodes_ref = torch.tensor([1.0, 2.0]), torch.zeros(2)
    edges_pred, edges_ref = torch.tensor([3.0]), torch.zeros(1)
    loss, _ = elementwise_mse(nodes_pred=nodes_pred, nodes_ref=nodes_ref,
                              edges_pred=edges_pred, edges_ref=edges_ref)
    assert torch.isclose(loss, torch.tensor((1 + 4 + 9) / 3))
    # The full matrix of a symmetric storage has the edge element twice.
    # (as passed through loss_kwargs in the lightning models)
    loss, _ = elementwise_mse(edge_weight=2)(nodes_pred=nodes_pred, nodes_ref=nodes_ref,
                                             edges_pred=edges_pred, edges_ref=edges_ref)
    assert torch.isclose(loss, torch.tensor((1 + 4 + 2 * 9) / 4))


@pytest.mark.parametrize(
    "name", ["block_type_mape", "block_type_mapemaemix", "block_type_mapemsemix", "block_type_mapestdmix"]
)
def test_relative_losses_have_finite_gradients_with_exact_zeros(name):
    """H has exact zeros (symmetry-zero elements): relative losses must skip them
    without producing NaN gradients (0 * inf in the backward pass)."""
    import torch
    from graph2mat.core.data import metrics

    nodes_ref = torch.tensor([0.0, 2.0, -1.0, 0.0, 3.0])
    edges_ref = torch.tensor([0.0, 0.5, float("nan"), -0.25])
    nodes_pred = (nodes_ref + 0.1).requires_grad_(True)
    edges_pred = torch.tensor([0.1, 0.6, 0.3, -0.2], requires_grad=True)
    loss, _ = getattr(metrics, name)(nodes_pred=nodes_pred, nodes_ref=nodes_ref,
                                     edges_pred=edges_pred, edges_ref=edges_ref)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(nodes_pred.grad).all() and torch.isfinite(edges_pred.grad).all()


def test_gauge_invariant_mse_ignores_a_per_structure_shift():
    """Labels (H, S): predicting H + c_g S with a different c per structure costs ~0."""
    import numpy as np
    import torch
    from graph2mat.core.data.metrics import elementwise_mse, gauge_invariant_mse

    class OneLabelPerSite:
        edge_block_size = np.array([1])

        @staticmethod
        def point_block_pointer(point_types):
            return np.arange(len(point_types) + 1)

    n = 6  # structures 0 (nodes 0-2) and 1 (nodes 3-5), one edge per node inside its structure
    batch = {"point_types": np.zeros(n, dtype=int), "batch": np.array([0, 0, 0, 1, 1, 1]),
             "edge_types": np.zeros(n, dtype=int),
             "edge_index": np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]])}
    torch.manual_seed(0)
    nodes_ref = torch.stack([torch.randn(n), 1 + 0.1 * torch.rand(n)], dim=1)  # (H, S); S_ii ~ 1
    edges_ref = torch.stack([torch.randn(n), 0.3 * torch.rand(n)], dim=1)
    shift = torch.tensor([0.7, -1.3])
    node_c, edge_c = shift[torch.as_tensor(batch["batch"])], shift[torch.as_tensor(batch["batch"])]
    nodes_pred = nodes_ref.clone(); nodes_pred[:, 0] += node_c * nodes_ref[:, 1]
    edges_pred = edges_ref.clone(); edges_pred[:, 0] += edge_c * edges_ref[:, 1]
    nodes_pred.requires_grad_(True)

    kwargs = dict(nodes_pred=nodes_pred, nodes_ref=nodes_ref, edges_pred=edges_pred, edges_ref=edges_ref,
                  batch=batch, basis_table=OneLabelPerSite())
    loss, stats = gauge_invariant_mse(**kwargs)
    assert loss.item() < 1e-10
    assert torch.allclose(stats["gauge_shift_mean_abs"], shift.abs().mean())
    assert elementwise_mse(**kwargs)[0].item() > 0.1
    loss.backward()
    assert torch.isfinite(nodes_pred.grad).all()

    # S as a label only: predicting H alone (1-D, one component fewer) gives the same H loss.
    h_only = dict(kwargs, nodes_pred=nodes_pred[:, 0].detach(), edges_pred=edges_pred[:, 0])
    loss_h, stats_h = gauge_invariant_mse(**h_only)
    assert loss_h.item() < 1e-10
    assert stats_h["overlap_rmse"].item() == 0
    wrong = dict(h_only, nodes_pred=h_only["nodes_pred"] + 0.1 * torch.randn(n))
    assert torch.isclose(gauge_invariant_mse(**wrong)[0],
                         gauge_invariant_mse(**dict(kwargs, nodes_pred=torch.stack(
                             [wrong["nodes_pred"], nodes_ref[:, 1]], dim=1)))[0])


def test_spectral_window_loss_matches_gamma_point_matrices():
    """Roadmap 4.4: the label elements land on the right Gamma-point orbitals (homo- and
    heteronuclear blocks, periodic images, a batch of two structures), the window loss
    ignores H -> H + c S, and its diagonal is the first-order change of the eigenvalues."""
    import numpy as np
    import scipy.linalg
    import torch
    from torch_geometric.data import Batch

    import graph2mat.bindings.torch  # noqa: F401 (registers the torch converters)
    from graph2mat import BasisConfiguration, BasisTableWithEdges, MatrixDataProcessor, PointBasis, conversions
    from graph2mat.core.data.metrics import _per_element_structure_ids, gauge_invariant_mse, spectral_window_layout

    basis = [PointBasis("A", R=2.5, basis="2x0e + 1o", basis_convention="cartesian"),
             PointBasis("B", R=2.0, basis="0e + 1o", basis_convention="cartesian")]
    table = BasisTableWithEdges(basis)
    proc = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    conv = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")
    rng = np.random.default_rng(0)
    structs = [conv(BasisConfiguration(point_types=types, positions=rng.uniform(0, 3.0, (len(types), 3)), basis=basis,
                                       cell=np.eye(3) * 3.0, pbc=(True, True, False)), data_processor=proc)
               for types in (["A", "B", "A", "B"], ["B", "A", "A"])]

    def random_labels(d, scale):
        """Random labels of a symmetric matrix (on-site blocks symmetric too)."""
        sizes = table.basis_size[d.point_types.numpy()]
        nodes = np.concatenate([(lambda b: (b + b.T) / 2)(rng.normal(size=(n, n))).ravel() for n in sizes])
        n_edges = int(table.edge_block_size[np.abs(d.edge_types.numpy()[::2])].sum())
        return torch.tensor(scale * nodes), torch.tensor(scale * rng.normal(size=n_edges))

    def gamma(d, nodes, edges):
        m = proc.matrix_from_data(d, predictions={"node_labels": nodes, "edge_labels": edges}, out_format="numpy")
        return m.reshape(m.shape[0], -1, m.shape[0]).sum(1)

    h, s, x = [], [], []
    for d in structs:
        h.append(random_labels(d, 1.0))
        s_nodes, s_edges = random_labels(d, 0.02)
        lay = spectral_window_layout(s_edges, d, table)
        s_nodes[lay["node_row"] == lay["node_col"]] += 1.0  # S(Gamma) ~ identity: positive definite
        s.append((s_nodes, s_edges))
        x.append(random_labels(d, 1.0))
    n_window = 6
    for i, d in enumerate(structs):
        h_g, s_g = gamma(d, *h[i]), gamma(d, *s[i])
        _, vecs = scipy.linalg.eigh(h_g, s_g)
        k = 4 if i == 0 else n_window  # structure 0 padded with zero columns
        c = np.zeros((h_g.shape[0], n_window))
        c[:, :k] = vecs[:, 5:5 + k]
        d.spectral_C = torch.tensor(c)
        d.point_labels = torch.stack([h[i][0], s[i][0]], dim=1)
        d.edge_labels = torch.stack([h[i][1], s[i][1]], dim=1)
    batch = Batch.from_data_list(structs)

    # The label elements reproduce the Gamma-point matrix of each structure.
    lay = spectral_window_layout(batch.edge_labels, batch, table)
    node_struct, edge_struct, _ = _per_element_structure_ids(batch.point_labels, batch.edge_labels, batch, table)
    dense = torch.zeros((2, lay["no_max"], lay["no_max"]), dtype=torch.float64).index_put(
        (torch.cat([node_struct, edge_struct, edge_struct]), torch.cat([lay["node_row"], lay["edge_row"], lay["edge_col"]]),
         torch.cat([lay["node_col"], lay["edge_col"], lay["edge_row"]])),
        torch.cat([batch.point_labels[:, 0], batch.edge_labels[:, 0], batch.edge_labels[:, 0]]), accumulate=True)
    for i, d in enumerate(structs):
        g = gamma(d, *h[i])
        assert np.abs(dense[i, :len(g), :len(g)].numpy() - g).max() < 1e-12
        assert not dense[i, len(g):].any()

    def loss(nodes_h, edges_h):
        return gauge_invariant_mse(nodes_pred=nodes_h, nodes_ref=batch.point_labels, edges_pred=edges_h,
                                   edges_ref=batch.edge_labels, batch=batch, basis_table=table, spectral_weight=1.0)

    # H + c S (a different c per structure) costs nothing.
    shift = torch.tensor([0.7, -1.3])
    nodes_shift = batch.point_labels[:, 0] + shift[node_struct] * batch.point_labels[:, 1]
    edges_shift = batch.edge_labels[:, 0] + shift[edge_struct] * batch.edge_labels[:, 1]
    total, stats = loss(nodes_shift, edges_shift)
    assert total.item() < 1e-20 and stats["spectral_rmse"].item() < 1e-10

    # A perturbation eps X: the window matrix is C^T X(Gamma) C, and its diagonal is the
    # first-order change of the window eigenvalues (both minus their mean).
    eps = 1e-5
    x_nodes, x_edges = torch.cat([n for n, _ in x]), torch.cat([e for _, e in x])
    nodes_pred = (batch.point_labels[:, 0] + eps * x_nodes).requires_grad_(True)
    _, stats = loss(nodes_pred, batch.edge_labels[:, 0] + eps * x_edges)
    expected, count = 0.0, 0
    for i, d in enumerate(structs):
        c = d.spectral_C.numpy()[:, :4 if i == 0 else n_window]
        h_g, s_g, x_g = gamma(d, *h[i]), gamma(d, *s[i]), gamma(d, *x[i])
        m = c.T @ x_g @ c
        first_order = np.diag(m) - np.diag(m).mean()
        exact = (scipy.linalg.eigvalsh(h_g + eps * x_g, s_g) - scipy.linalg.eigvalsh(h_g, s_g))[5:5 + c.shape[1]] / eps
        assert np.abs(first_order - (exact - exact.mean())).max() < 1e-3 * np.abs(first_order).max()
        expected += ((m - np.diag(m).mean() * np.eye(len(m))) ** 2).sum()
        count += len(m) ** 2
    assert np.isclose(stats["spectral_rmse"].item(), eps * (expected / count) ** 0.5, rtol=1e-6)
    total, _ = loss(nodes_pred, batch.edge_labels[:, 0] + eps * x_edges)
    total.backward()
    assert torch.isfinite(nodes_pred.grad).all() and nodes_pred.grad.abs().max() > 0


def _pair_batch(displacement=0.02):
    """Two +/- pairs (one across a periodic boundary) and the shared machinery."""
    import numpy as np
    import torch
    from torch_geometric.data import Batch

    import graph2mat.bindings.torch  # noqa: F401 (registers the torch converters)
    from graph2mat import BasisConfiguration, BasisTableWithEdges, MatrixDataProcessor, PointBasis, conversions

    basis = [PointBasis("A", R=2.5, basis="0e + 1o", basis_convention="cartesian")]
    table = BasisTableWithEdges(basis)
    proc = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    conv = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")
    rng = np.random.default_rng(1)
    cell = np.eye(3) * 4.0
    base_a = rng.uniform(0.5, 3.5, (4, 3))
    base_b = rng.uniform(0.5, 3.5, (4, 3))
    base_b[0, 0] = 3.999  # this atom crosses x = 4 Ang in the + partner
    u = rng.normal(size=(4, 3)); u /= np.linalg.norm(u, axis=1).max()
    u[0] = (1.0, 0.0, 0.0)
    plus_b = base_b + displacement * u
    plus_b[0, 0] -= 4.0  # wrapped into the cell
    positions = [base_a + displacement * u, base_a - displacement * u, plus_b, base_b - displacement * u]
    structs = [conv(BasisConfiguration(point_types=["A"] * 4, positions=p, basis=basis, cell=cell,
                                       pbc=(True, True, False)), data_processor=proc) for p in positions]
    for d in structs:
        n_nodes = int(table.basis_size[d.point_types.numpy()].sum() * 4)  # 4 x 4 on-site blocks
        n_edges = int(table.edge_block_size[np.abs(d.edge_types.numpy()[::2])].sum())
        d.point_labels = torch.tensor(rng.normal(size=n_nodes))
        d.edge_labels = torch.tensor(rng.normal(size=n_edges))
    return Batch.from_data_list(structs), table


def test_find_displacement_pairs_uses_minimum_image():
    from graph2mat.core.data.metrics import find_displacement_pairs

    batch, _ = _pair_batch()
    assert find_displacement_pairs(batch, 0.05) == [(0, 1), (2, 3)]
    assert find_displacement_pairs(batch, 0.03) == []  # partners differ by 2 x 0.02
    assert find_displacement_pairs(batch, 0.05, 0.035) == [(0, 1), (2, 3)]  # window around 2 delta
    assert find_displacement_pairs(batch, 0.05, 0.045) == []


def test_pair_difference_mse_penalises_only_the_antisymmetric_error():
    import pytest
    import torch
    from graph2mat.core.data.metrics import _gamma_matrices, elementwise_mse, pair_difference_mse

    batch, table = _pair_batch()
    nodes_ref, edges_ref = batch.point_labels, batch.edge_labels
    struct = batch.batch.repeat_interleave(16)  # node elements per structure (4x4 blocks)
    kwargs = dict(nodes_ref=nodes_ref, edges_ref=edges_ref, batch=batch, basis_table=table, expected_pairs=2)

    # Exact prediction: zero.
    loss, stats = pair_difference_mse(nodes_pred=nodes_ref.clone(), edges_pred=edges_ref.clone(), **kwargs)
    assert loss.item() == 0 and stats["n_pairs"] == 2

    # The same error on both partners of a pair: only the plain MSE sees it.
    common = torch.randn(int((struct == 0).sum()), dtype=nodes_ref.dtype)
    nodes = nodes_ref.clone()
    for a, b in ((0, 1), (2, 3)):
        nodes[struct == a] += common
        nodes[struct == b] += common
    loss, stats = pair_difference_mse(nodes_pred=nodes, edges_pred=edges_ref.clone(), **kwargs)
    plain, _ = elementwise_mse(nodes_pred=nodes, nodes_ref=nodes_ref, edges_pred=edges_ref, edges_ref=edges_ref)
    assert stats["pair_rmse"].item() < 1e-12 and torch.isclose(loss, plain)

    # An antisymmetric error: the term is the Gamma-point difference, per label element.
    nodes = nodes_ref.clone()
    nodes[struct == 0] += common
    nodes[struct == 1] -= common
    nodes.requires_grad_(True)
    loss, stats = pair_difference_mse(nodes_pred=nodes, edges_pred=edges_ref.clone(), pair_weight=1.0, **kwargs)
    gamma, node_struct, edge_struct = _gamma_matrices(nodes - nodes_ref, edges_ref * 0, edges_ref, batch, table)
    n_el = (node_struct == 0).sum() + (edge_struct == 0).sum()
    expected = ((gamma[0] - gamma[1]) ** 2).sum() / n_el / 2  # mean over the two pairs; pair 2 is exact
    assert torch.isclose(stats["pair_rmse"] ** 2, expected)
    loss.backward()
    assert torch.isfinite(nodes.grad).all() and nodes.grad.abs().sum() > 0

    # A batch with another number of pairs is an error; one without pairs has no term.
    with pytest.raises(ValueError):
        pair_difference_mse(nodes_pred=nodes_ref, edges_pred=edges_ref, **dict(kwargs, expected_pairs=3))
    with pytest.raises(ValueError):  # no pairs found where some are expected
        pair_difference_mse(nodes_pred=nodes_ref, edges_pred=edges_ref, **dict(kwargs, pair_max_displacement=0.01))
    _, stats = pair_difference_mse(nodes_pred=nodes_ref, edges_pred=edges_ref,
                                   **dict(kwargs, pair_max_displacement=0.01, expected_pairs=None))
    assert stats["n_pairs"] == 0

    # Validation (model in eval mode): plain MSE, no pairing at all.
    model = torch.nn.Linear(1, 1).eval()
    loss, stats = pair_difference_mse(nodes_pred=nodes.detach(), edges_pred=edges_ref, model=model, **kwargs)
    plain, _ = elementwise_mse(nodes_pred=nodes.detach(), nodes_ref=nodes_ref, edges_pred=edges_ref, edges_ref=edges_ref)
    assert torch.isclose(loss, plain) and "n_pairs" not in stats
    loss_train, _ = pair_difference_mse(nodes_pred=nodes.detach(), edges_pred=edges_ref, model=model.train(), **kwargs)
    assert loss_train > loss
