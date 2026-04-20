import numpy as np
import sisl

from graph2mat import (
    MatrixDataProcessor,
    PointBasis,
    BasisTableWithEdges,
    OrbitalConfiguration,
    conversions,
    Formats,
)


def _single_orbital_geometry():
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s")])
    return sisl.Geometry([[0.0, 0.0, 0.0]], atoms=[atom], lattice=[10, 10, 10])


def test_hamiltonian_collinear_read_preserves_two_components():
    from scipy.sparse import csr_array

    geometry = _single_orbital_geometry()
    h = sisl.Hamiltonian(geometry, spin=sisl.Spin("polarized"))
    h._csr = h._csr.fromsp([csr_array([[1.0]]), csr_array([[2.0]])])

    config = OrbitalConfiguration.from_matrix(h, labels=True)
    block = config.matrix.block_dict[(0, 0, 0)]

    assert block.shape == (1, 1, 2)
    np.testing.assert_allclose(block[0, 0, 0], 1.0)
    np.testing.assert_allclose(block[0, 0, 1], 2.0)


def test_flatten_nodes_and_edges_multicomponent():
    from graph2mat.core.data.matrices import OrbitalMatrix

    matrix = OrbitalMatrix(
        block_dict={(0, 0, 0): np.array([[[1.0, 2.0]]])},
        nsc=np.array([1, 1, 1]),
        orbital_count=np.array([1]),
    )

    node_labels, edge_labels = matrix.to_flat_nodes_and_edges(
        edge_index=np.empty((2, 0), dtype=np.int64),
        edge_sc_shifts=np.empty((0,), dtype=np.int64),
    )

    assert node_labels.shape == (1, 2)
    assert edge_labels.shape == (0, 2)
    np.testing.assert_allclose(node_labels[0], np.array([1.0, 2.0]))


def test_yield_from_batch_multicomponent_labels():
    basis_table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    processor = MatrixDataProcessor(
        basis_table=basis_table,
        symmetric_matrix=True,
        sub_point_matrix=False,
        n_matrix_components=2,
    )

    class FakeBatch:
        num_graphs = 2

        def __init__(self):
            self._arrays = type(
                "A",
                (),
                dict(
                    ptr=np.array([0, 1, 2]),
                    n_edges=np.array([3, 3]),
                    point_types=np.array([0, 0]),
                    edge_types=np.array([0, 0, 0, 0, 0, 0]),
                ),
            )()
            self._examples = [type("E", (), {})(), type("E", (), {})()]

        def numpy_arrays(self):
            return self._arrays

        def get_example(self, index):
            return self._examples[index]

    batch = FakeBatch()
    predictions = {
        "node_labels": np.array([[10.0, 11.0], [20.0, 21.0]]),
        "edge_labels": np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1]]),
    }

    outputs = list(processor.yield_from_batch(batch, predictions=predictions, as_matrix=False))
    assert outputs[0].point_labels.shape == (1, 2)
    assert outputs[1].edge_labels.shape == (2, 2)


def test_nodes_edges_to_polarized_hamiltonian():
    geometry = _single_orbital_geometry()
    converter = conversions.get_converter(Formats.NODESEDGES, Formats.SISL_H)

    h = converter(
        node_vals=np.array([[1.0, 2.0]]),
        edge_vals=np.empty((0, 2)),
        edge_index=np.empty((2, 0), dtype=np.int64),
        geometry=geometry,
    )

    assert isinstance(h, sisl.Hamiltonian)
    assert h._csr.data.shape[1] == 2
    np.testing.assert_allclose(h.tocsr(0).toarray()[0, 0], 1.0)
    np.testing.assert_allclose(h.tocsr(1).toarray()[0, 0], 2.0)
