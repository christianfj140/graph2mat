import numpy as np
import sisl

from graph2mat import BasisMatrixData, MatrixDataProcessor, OrbitalConfiguration
from graph2mat.core.data.table import AtomicTableWithEdges


def atoms():
    H = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s", R=1.6)])
    C = sisl.Atom(6, orbitals=[sisl.AtomicOrbital(n=2, l=0, m=0, R=2.0)]
                  + [sisl.AtomicOrbital(n=2, l=1, m=m, R=2.0) for m in (-1, 0, 1)])
    O = sisl.Atom(8, orbitals=[sisl.AtomicOrbital(n=2, l=0, m=0, R=1.8), sisl.AtomicOrbital(n=2, l=0, m=0, zeta=2, R=1.8)]
                  + [sisl.AtomicOrbital(n=2, l=1, m=m, R=1.8) for m in (-1, 0, 1)])
    return [H, C, O]


def geometry(species=(0, 1, 2, 0, 1), cell=(3.3, 3.1, 30.0), nsc=(5, 3, 1), seed=0, positions=None):
    rng = np.random.default_rng(seed)
    at = atoms()
    if positions is None:
        positions = rng.uniform(0, 1, (len(species), 3)) * np.minimum(np.array(cell), 4.0)
        positions[:, 2] = 10 + rng.uniform(0, 1.0, len(species))
    g = sisl.Geometry(positions, atoms=[at[s] for s in species], lattice=sisl.Lattice(np.diag(cell), nsc=nsc))
    return g


def pairs(g):
    """(ia, ja, isc) for all pairs with dist < R_i + R_j (excluding ia==ja at isc 0)."""
    out = []
    for ia in range(g.na):
        for isc in range(g.n_s):
            off = g.lattice.sc_off[isc] @ g.cell
            for ja in range(g.na):
                if ia == ja and isc == 0:
                    continue
                d = np.linalg.norm(g.xyz[ja] + off - g.xyz[ia])
                if d < g.atoms[ia].maxR() + g.atoms[ja].maxR() - 1e-3:
                    out.append((ia, ja, isc))
    return out


def random_matrix(g, cls=sisl.Hamiltonian, dim=1, seed=1, symmetric=False, orthogonal=True, spin=None):
    rng = np.random.default_rng(seed)
    kw = {}
    if spin is not None:
        kw["spin"] = spin
    else:
        kw["dim"] = dim
    if not orthogonal:
        kw["orthogonal"] = False
    M = cls(g, **kw)
    no = g.no
    nd = M.dim
    for ia in range(g.na):
        oi = g.a2o(ia, all=True)
        # on-site
        for i in oi:
            for j in oi:
                M[i, j] = rng.normal(size=nd) if nd > 1 else rng.normal()
    for ia, ja, isc in pairs(g):
        oi = g.a2o(ia, all=True)
        oj = g.a2o(ja, all=True) + isc * no
        for i in oi:
            for j in oj:
                M[i, j] = rng.normal(size=nd) if nd > 1 else rng.normal()
    if symmetric:
        M = (M + M.transpose()) / 2
    return M


def processor(**kw):
    table = AtomicTableWithEdges(atoms())
    kw.setdefault("sub_point_matrix", False)
    return MatrixDataProcessor(basis_table=table, **kw)


def roundtrip(M, proc, **cfg_kw):
    config = OrbitalConfiguration.from_matrix(M, labels=True, **cfg_kw)
    data = BasisMatrixData.from_config(config, proc)
    return data, data.convert_to(proc.default_out_format if proc.out_matrix else "sisl")


def dense(M, comp=None):
    return np.stack([M.tocsr(i).toarray() for i in range(M.shape[2])], -1)
