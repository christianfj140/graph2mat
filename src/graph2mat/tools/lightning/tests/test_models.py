"""Full MACE matrix predictions: rotations, periodicity and tiled supercells."""

import numpy as np
import pytest
import sisl
import torch
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor, OrbitalConfiguration
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel


@pytest.fixture(scope="module")
def float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@pytest.fixture(scope="module", params=["same_basis", "DZP"])
def model_and_processor(request, float64):
    def shell(l):
        return [sisl.AtomicOrbital(n=l + 1, l=l, m=m, R=2.1) for m in range(-l, l + 1)]

    orbitals = shell(0) + shell(1)
    if request.param == "same_basis":
        atoms = [sisl.Atom("H", orbitals=orbitals), sisl.Atom("O", orbitals=orbitals)]
    else:
        double = shell(0) + shell(0) + shell(1) + shell(1)
        atoms = [sisl.Atom("H", orbitals=double), sisl.Atom("O", orbitals=double + shell(2))]
    table = AtomicTableWithEdges(atoms)
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    torch.manual_seed(0)
    model = LitMACEMatrixModel(
        basis_table=table, symmetric_matrix=True, hidden_irreps="2x0e+2x1o+2x2e",
        edge_hidden_irreps="2x0e+2x1o+2x2e", num_bessel=3, max_ell=2,
        num_interactions=2, correlation=2,
    ).eval()
    return model, processor, atoms


@pytest.fixture(scope="module", params=["bimolecule", "square", "chain"])
def geometry(request, model_and_processor):
    _, _, (h, o) = model_and_processor
    if request.param == "bimolecule":
        return sisl.Geometry([[0., 0., 0.], [1., 0.3, 0.2]], atoms=[h, o], lattice=[20, 20, 20])
    if request.param == "square":
        return sisl.Geometry([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
                             atoms=[h, o, h, o], lattice=[20, 20, 20])
    geom = sisl.Geometry([[0., 0., 0.], [1.3, 0.2, 0.]], atoms=[h, o], lattice=[4, 20, 20])
    geom.set_nsc([3, 1, 1])
    return geom


def _matrix(model, processor, geometry, out_format="numpy"):
    config = OrbitalConfiguration.from_geometry(geometry)
    data = TorchBasisMatrixData.from_config(config, data_processor=processor)
    with torch.no_grad():
        predictions = model(data)
    return processor.matrix_from_data(data, predictions, out_format=out_format)


def test_mace_radial_embedding_matches_interaction_contract(model_and_processor, geometry):
    from mace.modules.utils import get_edge_vectors_and_lengths

    model, processor, _ = model_and_processor
    data = TorchBasisMatrixData.from_config(
        OrbitalConfiguration.from_geometry(geometry), data_processor=processor,
    )
    mace = model.model.mace
    _, lengths = get_edge_vectors_and_lengths(
        positions=data.positions, edge_index=data.edge_index, shifts=data.shifts,
    )
    features, cutoff = mace.radial_embedding(
        lengths, data.node_attrs, data.edge_index, mace.atomic_numbers,
    )
    assert features.shape == (data.edge_index.shape[1], mace.interactions[0].edge_feats_irreps.dim)
    # The default embedding has already multiplied by the cutoff.
    assert cutoff is None
    assert torch.isfinite(features).all()


def test_full_mace_matrix_equivariance_under_seeded_random_rotations(model_and_processor, geometry):
    model, processor, _ = model_and_processor
    original = _matrix(model, processor, geometry)
    assert np.isfinite(original).all() and np.max(np.abs(original)) > 1e-6
    cob = processor.basis_table.change_of_basis
    for rotation in Rotation.random(64, random_state=np.random.default_rng(0)).as_matrix():
        rotated = geometry.copy()
        rotated.xyz[:] = geometry.xyz @ rotation.T
        rotated.cell[:] = geometry.cell @ rotation.T
        actual = _matrix(model, processor, rotated)
        basis_rotation = torch.as_tensor(cob @ rotation @ cob.T)
        orbital_rotation = block_diag(*[
            basis.e3nn_irreps.D_from_matrix(basis_rotation).numpy()
            for basis in (processor.basis_table.basis[int(species)] for species in geometry.atoms.species)
        ])
        expected = np.concatenate([
            orbital_rotation @ cell @ orbital_rotation.T
            for cell in np.split(original, original.shape[1] // geometry.no, axis=1)
        ], axis=1)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_full_mace_prediction_commutes_with_supercell_tiling(model_and_processor, geometry):
    model, processor, _ = model_and_processor
    predicted = _matrix(model, processor, geometry, out_format="sisl_DM")
    expected = predicted.tile(2, 0).tile(2, 1).tile(2, 2)
    actual = _matrix(model, processor, geometry.tile(2, 0).tile(2, 1).tile(2, 2), out_format="sisl_DM")
    assert np.max(np.abs(expected.tocsr().toarray())) > 1e-6
    np.testing.assert_allclose(actual.tocsr().toarray(), expected.tocsr().toarray(), rtol=1e-9, atol=1e-11)
