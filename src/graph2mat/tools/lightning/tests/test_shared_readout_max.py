"""Roadmap 7.5: basis_grouping="max" is the readout shared between species (one module,
padded to the largest basis and masked per species). With species of different basis
the full MACE model must stay equivariant."""

import numpy as np
import sisl
import torch
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor, OrbitalConfiguration
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel


def _shell(l):
    return [sisl.AtomicOrbital(n=l + 1, l=l, m=m, R=2.1) for m in range(-l, l + 1)]


def test_max_grouping_full_model_is_equivariant_across_species():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        atoms = [sisl.Atom("H", orbitals=_shell(0) + _shell(1)),
                 sisl.Atom("O", orbitals=_shell(0) + _shell(0) + _shell(1) + _shell(2))]
        table = AtomicTableWithEdges(atoms)
        processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
        torch.manual_seed(0)
        model = LitMACEMatrixModel(
            basis_table=table, symmetric_matrix=True, hidden_irreps="2x0e+2x1o+2x2e",
            edge_hidden_irreps="2x0e+2x1o+2x2e", num_bessel=3, max_ell=2, num_interactions=2,
            correlation=2, basis_grouping="max",
        ).eval()
        geometry = sisl.Geometry([[0, 0, 0], [1.0, 0.3, 0.2], [0.2, 1.1, -0.3]],
                                 atoms=[atoms[0], atoms[1], atoms[0]], lattice=[20, 20, 20])

        def matrix(geom):
            data = TorchBasisMatrixData.from_config(OrbitalConfiguration.from_geometry(geom), data_processor=processor)
            with torch.no_grad():
                return processor.matrix_from_data(data, model(data), out_format="numpy")

        original = matrix(geometry)
        assert np.abs(original).max() > 1e-3
        cob = table.change_of_basis
        for rotation in Rotation.random(4, random_state=np.random.default_rng(0)).as_matrix():
            rotated = geometry.copy()
            rotated.xyz[:] = geometry.xyz @ rotation.T
            rotated.cell[:] = geometry.cell @ rotation.T
            D = block_diag(*[table.basis[int(s)].e3nn_irreps.D_from_matrix(torch.as_tensor(cob @ rotation @ cob.T)).numpy()
                             for s in geometry.atoms.species])
            expected = np.concatenate([D @ c @ D.T for c in np.split(original, original.shape[1] // geometry.no, axis=1)], axis=1)
            np.testing.assert_allclose(matrix(rotated), expected, atol=1e-10)
    finally:
        torch.set_default_dtype(previous)
