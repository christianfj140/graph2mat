"""Independent real-space fields and Bloch sums, not self-roundtrip expectations."""

import numpy as np
import pytest
import sisl
from scipy.sparse import csr_array
from scipy.linalg import eigh

from graph2mat import AtomicTableWithEdges, BasisMatrixData, Formats, MatrixDataProcessor


@pytest.mark.parametrize("shells", [(0, 1), (0, 2, 3)], ids=["s-p", "s-d-f"])
@pytest.mark.parametrize("spin,channels", [("unpolarized", 1), ("polarized", 2), ("non-colinear", 4)])
@pytest.mark.parametrize("symmetric", [False, True])
def test_periodic_components_and_bloch_hermiticity_against_independent_fields(shells, spin, channels, symmetric):
    def orbitals(ls):
        return [sisl.AtomicOrbital(n=l + 1, l=l, m=m, R=2.8) for l in ls for m in range(-l, l + 1)]

    atoms = [sisl.Atom("H", orbitals=orbitals(shells)), sisl.Atom("O", orbitals=orbitals((0, 1)))]
    geometry = sisl.Geometry([[0., 0., 0.], [.5, 0., 0.]], atoms=atoms, lattice=[5., 20., 20.])
    geometry.set_nsc([3, 1, 1])
    rng = np.random.default_rng(0)
    fields = []
    by_image = []
    for channel in range(channels + 1):
        center = rng.normal(size=(geometry.no, geometry.no))
        center = (center + center.T) / 2
        image = rng.normal(size=center.shape)
        if channel == channels:
            center = np.eye(geometry.no) + .001 * center
            image *= .001
        blocks = {(0, 0, 0): center, (1, 0, 0): image, (-1, 0, 0): image.T}
        by_image.append(blocks)
        fields.append(np.concatenate([blocks[tuple(offset)] for offset in geometry.sc_off], axis=1))
    csr = [csr_array(field) for field in fields]
    original = sisl.Hamiltonian.fromsp(geometry, csr[0] if channels == 1 else csr[:-1], S=csr[-1], spin=spin)
    policy = "raw_components" if channels == 4 else "h_and_overlap"
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(atoms), out_matrix="hamiltonian",
        symmetric_matrix=symmetric, sub_point_matrix=False,
        n_matrix_components=channels + 1, matrix_component_policy=policy,
    )
    data = BasisMatrixData.new(original, data_processor=processor, labels=True)
    nodes, edges = data.point_labels.copy(), data.edge_labels.copy()
    assert nodes.shape[1] == edges.shape[1] == channels + 1
    np.testing.assert_array_equal(data.nsc.reshape(-1), [3, 1, 1])
    recovered = []
    for channel, expected in enumerate(fields):
        # Raw non-collinear fields have no implicit physical writer. Each numeric
        # component can still use the existing scalar NumPy reconstruction.
        actual = processor.matrix_from_data(data, {"node_labels": nodes[:, channel], "edge_labels": edges[:, channel]},
                                            out_format=Formats.NUMPY, threshold=None)
        np.testing.assert_array_equal(actual, expected)
        recovered.append(csr_array(actual))
        for offset, block in by_image[channel].items():
            opposite = tuple(-x for x in offset)
            np.testing.assert_array_equal(block, by_image[channel][opposite].T)
    np.testing.assert_array_equal(data.point_labels, nodes)
    np.testing.assert_array_equal(data.edge_labels, edges)
    if channels == 4:
        with pytest.raises(ValueError, match="raw_components"):
            processor.matrix_from_data(data, out_format=Formats.SISL_H)
        # Assemble explicitly through sisl only; Graph2Mat does not infer raw spin.
        reconstructed = sisl.Hamiltonian.fromsp(geometry, recovered[:-1], S=recovered[-1], spin=spin)
    else:
        reconstructed = processor.matrix_from_data(data, out_format=Formats.SISL_H, threshold=None)
    assert not reconstructed.orthogonal
    assert reconstructed.S_idx == channels
    np.testing.assert_array_equal(reconstructed.geometry.atoms.Z, geometry.atoms.Z)
    np.testing.assert_array_equal(reconstructed.geometry.orbitals, geometry.orbitals)
    np.testing.assert_array_equal(reconstructed.geometry.xyz, geometry.xyz)
    np.testing.assert_array_equal(reconstructed.geometry.cell, geometry.cell)
    np.testing.assert_array_equal(reconstructed.geometry.sc_off, geometry.sc_off)
    for channel, field in enumerate(fields):
        np.testing.assert_array_equal(reconstructed.tocsr(channel).toarray(), field)
    for k in (np.array([0., 0., 0.]), np.array([.17, 0., 0.]), np.array([-.31, 0., 0.])):
        bloch = [sum(block * np.exp(2j * np.pi * np.dot(k, offset)) for offset, block in blocks.items())
                 for blocks in by_image]
        assert np.linalg.eigvalsh(bloch[-1]).min() > .9
        expected_s = np.kron(bloch[-1], np.eye(2)) if channels == 4 else bloch[-1]
        np.testing.assert_allclose(reconstructed.Sk(k, format="array"), expected_s, rtol=0, atol=2e-14)
        if channels == 4:
            expected = np.zeros((2 * geometry.no, 2 * geometry.no), dtype=complex)
            expected[0::2, 0::2], expected[1::2, 1::2] = bloch[:2]
            expected[0::2, 1::2], expected[1::2, 0::2] = bloch[2] + 1j * bloch[3], bloch[2] - 1j * bloch[3]
            actual = reconstructed.Hk(k, format="array")
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
            np.testing.assert_allclose(actual, actual.conj().T, rtol=0, atol=2e-14)
        else:
            for channel in range(channels):
                actual = reconstructed.Hk(k, format="array", **({"spin": channel} if channels == 2 else {}))
                np.testing.assert_allclose(actual, bloch[channel], rtol=0, atol=2e-14)
                np.testing.assert_allclose(actual, actual.conj().T, rtol=0, atol=2e-14)
                expected_levels = eigh(bloch[channel], bloch[-1], eigvals_only=True)
                levels = reconstructed.eigenvalue(k=k, spin=channel).eig
                assert levels.shape == (geometry.no,)
                np.testing.assert_allclose(levels, expected_levels, rtol=1e-10, atol=1e-10)
