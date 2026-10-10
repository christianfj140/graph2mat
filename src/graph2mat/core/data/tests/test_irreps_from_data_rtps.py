"""Roadmap 3.7: irreps_from_data builds its RTPs with graph2mat's fast decomposition
(identical change of basis to e3nn's) once per (irreps, symmetry, dtype), not per call."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
o3 = pytest.importorskip("e3nn.o3")

from graph2mat import BasisTableWithEdges, MatrixDataProcessor, PointBasis


def _case():
    table = BasisTableWithEdges([
        PointBasis("A", R=2, basis="2x0e + 1x1o + 1x2e", basis_convention="spherical"),
        PointBasis("B", R=2, basis="1x0e + 1x1o", basis_convention="spherical"),
    ])
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    point_types = np.array([0, 1])
    edge_index = np.array([[0, 1], [1, 0]])
    edge_types = table.point_type_to_edge_type(point_types[edge_index])
    torch.manual_seed(0)
    n_points = table.point_block_pointer(point_types)[-1]
    n_edges = int(table.edge_block_size[abs(edge_types[0])])

    class Data:
        point_labels = torch.randn(n_points, dtype=torch.float64)
        edge_labels = torch.randn(n_edges, dtype=torch.float64)

        @staticmethod
        def numpy_arrays():
            return {"point_types": point_types, "edge_index": edge_index, "edge_types": edge_types}

    return table, processor, point_types, edge_types, Data()


def test_irreps_from_data_matches_e3nn_reduced_tensor_products():
    table, processor, point_types, edge_types, data = _case()
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        got = processor.irreps_from_data(data)
        ref_nodes, offset = [], 0
        for t in point_types:
            irreps = table.basis[t].e3nn_irreps
            rtp = o3.ReducedTensorProducts("ij=ji", i=irreps, j=irreps)
            size = irreps.dim
            block = data.point_labels[offset:offset + size * size].reshape(size, size)
            ref_nodes.append(torch.einsum("zij,ij->z", rtp.change_of_basis, block))
            offset += size * size
        i, j = (table.basis[t].e3nn_irreps for t in table.edge_type_to_point_types[abs(edge_types[0])])
        rtp = o3.ReducedTensorProducts("ij", i=i, j=j)
        ref_edge = torch.einsum("zij,ij->z", rtp.change_of_basis, data.edge_labels.reshape(i.dim, j.dim))
    finally:
        torch.set_default_dtype(default)
    torch.testing.assert_close(got["node_labels"], torch.cat(ref_nodes), rtol=0, atol=1e-12)
    torch.testing.assert_close(got["edge_labels"], ref_edge, rtol=0, atol=1e-12)


def test_irreps_from_data_reuses_rtps_across_calls(monkeypatch):
    _, processor, _, _, data = _case()
    processor.irreps_from_data(data)

    def rebuilt(*args, **kwargs):
        raise AssertionError("RTPs must be reused between calls")

    monkeypatch.setattr(o3, "ReducedTensorProducts", rebuilt)
    import graph2mat.bindings.e3nn.irreps_tools as tools
    monkeypatch.setattr(tools.ReducedTensorProducts, "__init__", rebuilt)
    processor.irreps_from_data(data)
