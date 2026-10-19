"""Physical objects use selected channels and predicted, not reference, overlap."""

import numpy as np
import pytest
import sisl
import torch
from scipy.linalg import eigh
from torch_geometric.data import Batch

from graph2mat import BasisMatrixData
from graph2mat.bindings.torch import TorchBasisMatrixData, TorchBasisMatrixDataset
from graph2mat.core.data.multitarget import prepare_multitarget_data
from graph2mat.core.data.tests.physical_fixtures import analytic_matrices
from graph2mat.core.data.tests.test_multitarget import TARGETS, processor_for


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def predicted_labels(spin):
    nodes = [[.75, 1.2, .9, .2]] * 2 if spin == 1 else [[.75, 1.5, 1.2, .9, .7, .2, .3]] * 2
    edges = [[-.25, .35, .4, .1]] if spin == 1 else [[-.25, -.5, .35, .4, .2, .1, .05]]
    return {"node_labels": np.array(nodes), "edge_labels": np.array(edges)}


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("target", TARGETS)
def test_target_reconstruction_uses_predicted_overlap(spin, backend, target):
    matrices = analytic_matrices(spin=spin)
    p = processor_for(matrices, spin)
    data = prepare_multitarget_data(p, **matrices)
    original_nodes = data.point_labels.copy()
    original_edges = data.edge_labels.copy()
    prediction = predicted_labels(spin)
    if backend == "torch":
        data = data.copy(cls=TorchBasisMatrixData)
        prediction = {k: torch.tensor(v, dtype=torch.float64) for k, v in prediction.items()}
    result = p.matrix_from_data(data, predictions=prediction, matrix_target=target)
    expected_s = np.array([[1.2, .35], [.35, 1.2]])
    expected = {
        "hamiltonian": [[[.75, -.25], [-.25, .75]], [[1.5, -.5], [-.5, 1.5]]],
        "density_matrix": [[[.9, .4], [.4, .9]], [[.7, .2], [.2, .7]]],
        "energy_density_matrix": [[[.2, .1], [.1, .2]], [[.3, .05], [.05, .3]]],
    }
    if target == "overlap":
        assert isinstance(result, sisl.Overlap)
        np.testing.assert_allclose(result.tocsr(0).toarray(), expected_s, rtol=0, atol=1e-14)
    else:
        assert isinstance(result, {"hamiltonian": sisl.Hamiltonian, "density_matrix": sisl.DensityMatrix,
                                   "energy_density_matrix": sisl.EnergyDensityMatrix}[target])
        assert result.spin.is_polarized == (spin == 2)
        np.testing.assert_allclose(result.tocsr(result.S_idx).toarray(), expected_s, rtol=0, atol=1e-14)
        for channel in range(spin):
            np.testing.assert_allclose(result.tocsr(channel).toarray(), expected[target][channel], rtol=0, atol=1e-14)
            if target == "hamiltonian":
                np.testing.assert_allclose(result.eigenvalue(spin=channel).eig,
                    eigh(expected[target][channel], expected_s, eigvals_only=True), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(data.ensure_numpy(data.point_labels), original_nodes,
                               rtol=0, atol=0)
    np.testing.assert_allclose(data.ensure_numpy(data.edge_labels), original_edges,
                               rtol=0, atol=0)


def test_target_required_before_physical_conversion():
    matrices = analytic_matrices()
    p = processor_for(matrices)
    data = prepare_multitarget_data(p, **matrices)
    with pytest.raises(ValueError, match="target|select"):
        p.matrix_from_data(data)
    with pytest.raises(ValueError, match="target|select"):
        data.convert_to("sisl_H")
    with pytest.raises(ValueError, match="target|select"):
        data.convert_to("scipy_csr")
    with pytest.raises(ValueError, match="target|select"):
        p.matrix_from_data(data, matrix_target="unknown")


def test_batch_geometry_predict_and_overlap_singularity():
    from graph2mat.bindings.torch.data.tests.test_multitarget_dataset import bundle, processor
    from graph2mat.tools.lightning.data import TypeLayoutCollate

    p = processor()
    dataset = TorchBasisMatrixDataset([bundle(2), bundle(3)], p)
    batch = TypeLayoutCollate(Batch.from_data_list, p.basis_table)([dataset[0], dataset[1]])
    prediction = {"node_labels": batch.point_labels.clone(), "edge_labels": batch.edge_labels.clone()}
    prediction["node_labels"][:, 1] = 1.2
    prediction["edge_labels"][:, 1] = .25
    result = p.matrix_from_data(batch, prediction, matrix_target="hamiltonian")
    assert [matrix.no for matrix in result] == [2, 3]
    for matrix in result:
        expected_s = np.full((matrix.no, matrix.no), .25)
        np.fill_diagonal(expected_s, 1.2)
        np.testing.assert_allclose(matrix.tocsr(matrix.S_idx).toarray(), expected_s, rtol=1e-10, atol=1e-10)
    single = dataset[0]
    values = {"node_labels": single.point_labels.clone(), "edge_labels": single.edge_labels.clone()}
    values["node_labels"][:, 1] = values["edge_labels"][:, 1] = 1.
    singular = p.matrix_from_data(single, values, matrix_target="hamiltonian")
    np.testing.assert_array_equal(singular.tocsr(singular.S_idx).toarray(), np.ones((2, 2)))
    with pytest.raises(np.linalg.LinAlgError):
        eigh(singular.tocsr(0).toarray(), singular.tocsr(singular.S_idx).toarray())


def test_torch_predict_requires_selection_and_never_reads_reference_labels():
    matrices = analytic_matrices()
    p = processor_for(matrices)
    def predict(data):
        assert getattr(data, "point_labels", None) is None
        assert data.matrix_target_signature.tolist() == [[1, 1, 1, 1, 1, 1]]
        return {name: torch.tensor(value, dtype=torch.float64) for name, value in predicted_labels(1).items()}
    with pytest.raises(ValueError, match="target|select"):
        p.torch_predict(predict, matrices["overlap"].geometry)
    matrix = p.torch_predict(predict, matrices["overlap"].geometry, matrix_target="hamiltonian")
    np.testing.assert_allclose(matrix.tocsr(matrix.S_idx).toarray(), [[1.2, .35], [.35, 1.2]], rtol=0, atol=1e-14)


def test_reconstruction_rejects_same_width_wrong_semantics():
    matrices = analytic_matrices()
    p = processor_for(matrices, targets=TARGETS[:3])
    q = processor_for(matrices, targets=("hamiltonian", "overlap", "energy_density_matrix"))
    data = prepare_multitarget_data(p, **matrices)
    with pytest.raises(ValueError, match="layout|signature|semantic"):
        q.matrix_from_data(data, matrix_target="energy_density_matrix")


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
def test_periodic_multi_orbital_species_and_component_order(spin, symmetric):
    from scipy.sparse import csr_array

    def orbitals(shells):
        return [sisl.AtomicOrbital(n=l+1, l=l, m=m, R=2.8)
                for l in shells for m in range(-l, l+1)]
    atoms = [sisl.Atom("H", orbitals=orbitals((0, 2, 3))),
             sisl.Atom("O", orbitals=orbitals((0, 1)))]
    geometry = sisl.Geometry([[0., 0., 0.], [.5, 0., 0.]], atoms=atoms, lattice=[5., 20., 20.])
    geometry.set_nsc([3, 1, 1])
    rng = np.random.default_rng(76)
    fields = []
    for component in range(3*spin+1):
        center = rng.normal(size=(geometry.no, geometry.no))
        center = (center + center.T) / 2
        image = rng.normal(size=center.shape)
        if component == spin:
            center = np.eye(geometry.no) + .001 * center
            image *= .001
        blocks = {(0, 0, 0): center, (1, 0, 0): image, (-1, 0, 0): image.T}
        fields.append(np.concatenate([blocks[tuple(offset)] for offset in geometry.sc_off], axis=1))
    csrs = [csr_array(field) for field in fields]
    s = csrs[spin]
    matrices = dict(overlap=sisl.Overlap.fromsp(geometry, s))
    for key, cls, start in (("hamiltonian", sisl.Hamiltonian, 0),
                            ("density", sisl.DensityMatrix, spin+1),
                            ("energy_density", sisl.EnergyDensityMatrix, 2*spin+1)):
        matrices[key] = cls.fromsp(geometry, csrs[start] if spin == 1 else csrs[start:start+spin], S=s)
    p = processor_for(matrices, spin, symmetric)
    data = prepare_multitarget_data(p, **matrices)
    for target, columns in p.matrix_target_layout.slices.items():
        reconstructed = p.matrix_from_data(data, matrix_target=target, threshold=None)
        np.testing.assert_array_equal(reconstructed.geometry.atoms.Z, [1, 8])
        np.testing.assert_array_equal(reconstructed.geometry.orbitals, [13, 4])
        np.testing.assert_array_equal(reconstructed.geometry.sc_off, geometry.sc_off)
        for channel, expected in enumerate(fields[columns]):
            np.testing.assert_allclose(reconstructed.tocsr(channel).toarray(), expected, rtol=0, atol=1e-14)


def test_matrix_threshold_does_not_silently_threshold_predicted_overlap():
    matrices = analytic_matrices()
    p = processor_for(matrices)
    data = prepare_multitarget_data(p, **matrices)
    prediction = predicted_labels(1)
    prediction["edge_labels"][0, 1] = 1e-9
    matrix = p.matrix_from_data(data, prediction, threshold=1e-6, matrix_target="hamiltonian")
    assert matrix.tocsr(matrix.S_idx)[0, 1] == pytest.approx(1e-9, rel=0, abs=1e-20)
