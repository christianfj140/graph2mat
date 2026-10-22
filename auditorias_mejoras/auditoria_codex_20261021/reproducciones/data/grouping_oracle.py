"""Graph2Mat resorting oracle: block ops return the TRUE block of the edge they are asked for;
forward output must then equal the data labels for every basis_grouping / symmetry / species order."""
import itertools
import numpy as np
import sisl
import common
from common import geometry, random_matrix, dense
from graph2mat import BasisMatrixData, MatrixDataProcessor, OrbitalConfiguration
from graph2mat.core.data.table import AtomicTableWithEdges, BasisTableWithEdges
from graph2mat.core.modules.graph2mat import Graph2Mat

STATE = {}


def embed(block, ti, tj):
    filt = STATE["filters"]
    if filt is None:
        return block
    nmax = filt.shape[1]
    out = np.full((nmax, nmax) + block.shape[2:], 777.0)
    out[np.ix_(filt[ti], filt[tj])] = block
    return out


def true_block(i, j, isc):
    D = STATE["dense"]  # (no, no*n_s, C)
    fo = STATE["firsto"]
    no = fo[-1]
    blk = D[fo[i]:fo[i + 1], isc * no + fo[j]: isc * no + fo[j + 1]]
    if STATE["C"] == 1:
        blk = blk[..., 0]
    return blk


class NodeOracle:
    def __init__(self, i_basis, j_basis, symmetry=None, **kw):
        self.i_basis = i_basis

    def __call__(self, node_feats, **kw):
        idx = node_feats[:, 0].astype(int)
        pt = STATE["point_types"]
        return np.stack([embed(true_block(n, n, 0), pt[n], pt[n]) for n in idx]) if len(idx) else np.zeros((0,))


class EdgeOracle:
    def __init__(self, i_basis, j_basis, symmetry=None, **kw):
        pass

    def __call__(self, node_feats, eid, **kw):
        e = eid[0].astype(int)
        ei = STATE["edge_index"]
        isc = STATE["neigh_isc"]
        pt = STATE["point_types"]
        return np.stack([embed(true_block(ei[0, k], ei[1, k], isc[k]), pt[ei[0, k]], pt[ei[1, k]]) for k in e])


def run(atom_list, species, symmetric, grouping, nsc=(5, 5, 1), cell=(3.3, 3.1, 30.0), spin=None, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, 1, (len(species), 3)) * np.minimum(np.array(cell), 4.0)
    xyz[:, 2] = 10 + rng.uniform(0, 1.0, len(species))
    g = sisl.Geometry(xyz, atoms=[atom_list[s] for s in species], lattice=sisl.Lattice(np.diag(cell), nsc=nsc))
    M = random_matrix(g, symmetric=symmetric, seed=seed + 1, spin=spin)
    table = AtomicTableWithEdges(atom_list)
    proc = MatrixDataProcessor(basis_table=table, symmetric_matrix=symmetric, sub_point_matrix=False,
                               out_matrix="hamiltonian", n_matrix_components=1 if spin is None else 2,
                               matrix_component_policy="h_only")
    data = BasisMatrixData.from_config(OrbitalConfiguration.from_matrix(M, labels=True), proc)
    # canonical dense matrix (in graph isc ordering == default sisl ordering here)
    STATE["dense"] = dense(M)
    STATE["C"] = STATE["dense"].shape[-1]
    STATE["firsto"] = g.firsto
    STATE["edge_index"] = np.asarray(data.edge_index)
    STATE["neigh_isc"] = np.asarray(data.neigh_isc)
    STATE["point_types"] = np.asarray(data.point_types)
    g2m = Graph2Mat(unique_basis=BasisTableWithEdges(table.basis), basis_grouping=grouping, symmetric=symmetric,
                    node_operation=NodeOracle, edge_operation=EdgeOracle,
                    n_matrix_components=1 if spin is None else 2)
    STATE["filters"] = g2m.basis_filters
    n = len(data.point_types)
    nodes, edges = g2m.forward(data, node_feats=np.arange(n)[:, None].astype(float),
                               edge_kwargs={"eid": np.arange(data.edge_index.shape[1]).astype(float)})
    okn = nodes.shape == data.point_labels.shape and np.allclose(nodes, data.point_labels)
    oke = edges.shape == data.edge_labels.shape and np.allclose(edges, data.edge_labels)
    return okn, oke, nodes.shape, data.point_labels.shape, edges.shape, data.edge_labels.shape


def s_atom(Z, R=1.6):
    return sisl.Atom(Z, orbitals=[sisl.AtomicOrbital(n=1, l=0, m=0, R=R)])


def sp_atom(Z, R=2.0, nzeta_s=1):
    return sisl.Atom(Z, orbitals=[sisl.AtomicOrbital(n=2, l=0, m=0, zeta=z + 1, R=R) for z in range(nzeta_s)]
                     + [sisl.AtomicOrbital(n=2, l=1, m=m, R=R) for m in (-1, 0, 1)])


ATOMS = {
    "H": s_atom(1), "C": sp_atom(6), "N": sp_atom(7, R=1.9), "O": sp_atom(8, R=1.8, nzeta_s=2),
}
orders = [("H", "C", "O"), ("C", "H", "N"), ("O", "C", "H"), ("C", "H", "N", "O"), ("N", "O", "H", "C")]
fails = 0
for order in orders:
    atom_list = [ATOMS[k] for k in order]
    nsp = len(order)
    species = tuple(list(range(nsp)) + [0, nsp - 1, 1])
    for symmetric, grouping in itertools.product((False, True), ("point_type", "basis_shape", "max")):
        for spin in (None, sisl.Spin("polarized")):
            try:
                r = run(atom_list, species, symmetric, grouping, spin=spin)
                tag = "OK" if r[0] and r[1] else f"FAIL nodes={r[0]} edges={r[1]} shapes={r[2:]}"
            except Exception as e:
                import traceback
                tag = f"EXC {type(e).__name__}: {str(e)[:150]}"
            if not tag.startswith("OK"):
                fails += 1
            print(order, "sym" if symmetric else "nonsym", grouping, "pol" if spin else "", tag)
print("fails", fails)
