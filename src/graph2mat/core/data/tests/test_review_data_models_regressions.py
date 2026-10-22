"""Remaining data-contract regressions found while reviewing the audit fixes."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import sisl
import torch

from graph2mat import (
    AtomicTableWithEdges,
    BasisConfiguration,
    BasisMatrixData,
    BasisTableWithEdges,
    MatrixDataProcessor,
    OrbitalConfiguration,
    PointBasis,
)
from graph2mat.bindings.torch import TorchBasisMatrixData, TorchBasisMatrixDataset
from graph2mat.core.data import _sparse


def test_same_z_equal_dimension_different_angular_shells_are_rejected():
    three_s = sisl.Atom(6, orbitals=[sisl.AtomicOrbital(n=n, l=0, m=0, R=0.8) for n in (2, 3, 4)])
    one_p = sisl.Atom(6, orbitals=[sisl.AtomicOrbital(n=2, l=1, m=m, R=0.8) for m in (-1, 0, 1)])
    geometry = sisl.Geometry([[0, 0, 0], [1, 0, 0]], atoms=[three_s, one_p], lattice=[10] * 3)
    matrix = sisl.Hamiltonian(geometry)
    for orbital in range(6):
        matrix[orbital, orbital] = orbital + 1
    processor = MatrixDataProcessor(BasisTableWithEdges([PointBasis.from_sisl_atom(three_s)]), sub_point_matrix=False)
    with pytest.raises(ValueError, match="same Z.*different basis"):
        BasisMatrixData.from_config(OrbitalConfiguration.from_matrix(matrix), processor)


def test_configuration_shell_order_must_match_the_table():
    actual = PointBasis(6, R=1.0, basis="1x1o+1x0e")
    expected = PointBasis(6, R=1.0, basis="1x0e+1x1o")
    config = BasisConfiguration(point_types=[6], positions=np.zeros((1, 3)), basis=[actual],
                                cell=np.eye(3) * 10, pbc=(False,) * 3, matrix=np.eye(4))
    processor = MatrixDataProcessor(BasisTableWithEdges([expected]), sub_point_matrix=False)
    with pytest.raises(ValueError, match="same Z.*different basis"):
        BasisMatrixData.from_config(config, processor)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_scalar_csr_blocks_preserve_complex_values(compiled, dtype):
    module = _sparse
    if not compiled:
        # The compiled extension may live elsewhere (scikit-build editable installs): find the
        # pure-Python source next to its sibling module instead.
        from graph2mat.core.data import sparse as sparse_wrapper

        source = Path(sparse_wrapper.__file__).with_name("_sparse.py")
        if not source.is_file():
            pytest.skip("pure-Python _sparse.py is not installed")
        pytest.importorskip("cython")  # pure mode imports the cython shim
        spec = importlib.util.spec_from_file_location("review_sparse_python", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    blocks = module._csr_to_block_dict(
        data=np.array([0.25 + 0.5j, -0.75 - 1j], dtype=dtype),
        ptr=np.array([0, 1, 2], dtype=np.int32), cols=np.array([0, 1], dtype=np.int32),
        atom_first_orb=np.array([0, 2], dtype=np.int32), orbitals=np.array([2], dtype=np.int32),
        n_atoms=1, fill_value=0,
    )
    np.testing.assert_array_equal(blocks[0, 0, 0], [[0.25 + 0.5j, 0], [0, -0.75 - 1j]])
    assert blocks[0, 0, 0].dtype.kind == "c"


def test_complex_scalar_hamiltonian_conversion_roundtrip():
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s", R=0.8)])
    geometry = sisl.Geometry([[0, 0, 0], [1, 0, 0]], atoms=atom, lattice=[10] * 3)
    matrix = sisl.Hamiltonian(geometry, dtype=np.complex128)
    matrix[0, 0], matrix[1, 1] = 1, 2
    matrix[0, 1], matrix[1, 0] = 0.25 + 0.5j, 0.25 - 0.5j
    processor = MatrixDataProcessor(AtomicTableWithEdges([atom]), sub_point_matrix=False, out_matrix="hamiltonian")
    data = BasisMatrixData.new(matrix, processor)
    result = data.convert_to("sisl_H").tocsr(0).toarray()
    np.testing.assert_array_equal(result, [[1, 0.25 + 0.5j], [0.25 - 0.5j, 2]])


def _labeled_configuration():
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s", R=0.8)])
    geometry = sisl.Geometry([[0, 0, 0], [1, 0, 0]], atoms=atom, lattice=[10] * 3)
    matrix = sisl.Hamiltonian(geometry, orthogonal=False)
    matrix[0, 0, :], matrix[1, 1, :] = [42, 1], [43, 1]
    matrix[0, 1, :], matrix[1, 0, :] = [0.2, 0.1], [0.2, 0.1]
    processor = MatrixDataProcessor(AtomicTableWithEdges([atom]), sub_point_matrix=False, symmetric_matrix=True,
                                    out_matrix="hamiltonian", matrix_component_policy="h_and_overlap",
                                    n_matrix_components=2)
    return OrbitalConfiguration.from_matrix(matrix, matrix_component_policy="h_and_overlap"), processor


@pytest.mark.parametrize("prepared", [False, True])
def test_dataset_labels_false_omits_targets_without_mutating_source(prepared):
    config, processor = _labeled_configuration()
    source = TorchBasisMatrixData.new(config, processor) if prepared else config
    source_matrix = config.matrix
    if prepared:
        source.total_charge = torch.tensor([2.0])
        source.extra_marker = torch.tensor([9.0])
        source.metadata["case"] = "kept"
        before_nodes, before_edges = source.point_labels.clone(), source.edge_labels.clone()
    result = TorchBasisMatrixDataset([source], processor, load_labels=False)[0]
    assert result is not source
    assert getattr(result, "point_labels", None) is None and getattr(result, "edge_labels", None) is None
    assert result._data.get("point_labels") is None and result._data.get("edge_labels") is None
    assert config.matrix is source_matrix
    if prepared:
        torch.testing.assert_close(source.point_labels, before_nodes)
        torch.testing.assert_close(source.edge_labels, before_edges)
        torch.testing.assert_close(result.total_charge, torch.tensor([2.0]))
        torch.testing.assert_close(result.extra_marker, torch.tensor([9.0]))
        assert result.metadata["case"] == "kept"
        torch.testing.assert_close(result["node_attrs"], source["node_attrs"])
    labeled = TorchBasisMatrixDataset([source], processor, load_labels=True)[0]
    torch.testing.assert_close(labeled.point_labels, torch.tensor([[42.0, 1.0], [43.0, 1.0]]))
    torch.testing.assert_close(labeled.edge_labels, torch.tensor([[0.2, 0.1]]))


def test_numpy_prepared_labels_false_omits_targets_without_mutation():
    config, processor = _labeled_configuration()
    source = BasisMatrixData.new(config, processor)
    nodes = source.point_labels.copy()
    source.extra_marker = np.array([9.0])
    result = BasisMatrixData.new(source, processor, labels=False)
    assert result is not source
    assert result.point_labels is None and result.edge_labels is None
    assert result._data.get("point_labels") is None and result._data.get("edge_labels") is None
    np.testing.assert_array_equal(source.point_labels, nodes)
    np.testing.assert_array_equal(result.extra_marker, [9.0])
