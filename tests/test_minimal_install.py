"""Core import/NumPy regressions, outside the Torch-dependent package conftest."""

import subprocess
import sys

import pytest


SMOKE = r'''
import importlib.abc
import sys

mode = sys.argv[1]
blocked = {"torch_geometric"}
if mode != "torch_only":
    blocked |= {"torch", "e3nn", "mace", "pytorch_lightning"}

class MissingML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root in blocked:
            name = "sympy" if mode == "broken_torch" and root == "torch" else root
            raise ModuleNotFoundError("simulated missing dependency: " + name, name=name)

sys.meta_path.insert(0, MissingML())

import numpy as np
import graph2mat as g
from graph2mat.core.data.metrics import elementwise_mse, block_type_mape
from graph2mat.core.modules._labels_resort import get_labels_resorting_array

indices = get_labels_resorting_array(
    np.array([1, 0, -1], dtype=np.int64),
    np.array([[1, 2], [1, 3]], dtype=np.int64),
    transpose_neg=True,
)
np.testing.assert_array_equal(indices, [1, 2, 3, 4, 5, 6, 0, 7, 10, 8, 11, 9, 12])

basis = g.BasisTableWithEdges([
    g.PointBasis("A", R=2, basis=[1], basis_convention="cartesian")
])
processor = g.MatrixDataProcessor(
    basis_table=basis, symmetric_matrix=True,
    sub_point_matrix=False, node_attr_getters=[],
)
config = g.BasisConfiguration(
    point_types=["A", "A"], positions=np.array([[0., 0., 0.], [1., 0., 0.]]),
    basis=basis, cell=np.eye(3)*10, pbc=(False, False, False),
    matrix=np.array([[1., .25], [.25, 1.]]),
)
data = g.BasisMatrixData.new(config, processor)
assert len(data.point_labels) == 2 and len(data.edge_labels) == 1
loss, _ = elementwise_mse(
    nodes_pred=data.point_labels+1, nodes_ref=data.point_labels,
    edges_pred=data.edge_labels+1, edges_ref=data.edge_labels,
)
assert np.isclose(loss, 1)
loss, _ = block_type_mape(
    nodes_pred=np.array([0., 1.]), nodes_ref=np.array([0., 1.]),
    edges_pred=np.ones(1), edges_ref=np.ones(1),
)
assert loss == 0
loss, _ = elementwise_mse(
    nodes_pred=np.ones((2, 2)), nodes_ref=np.zeros((2, 2)),
    edges_pred=np.array([[1., 1.], [99., 99.]]),
    edges_ref=np.array([[0., 0.], [np.nan, np.nan]]),
)
assert np.isclose(loss, 1)
'''


@pytest.mark.parametrize("mode", ["base", "torch_only", "broken_torch"])
def test_core_optional_torch(mode):
    result = subprocess.run(
        [sys.executable, "-c", SMOKE, mode], capture_output=True, text=True,
    )
    if mode == "broken_torch":
        assert result.returncode != 0
        assert "simulated missing dependency: sympy" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
