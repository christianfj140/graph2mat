"""Long-range electrostatics for the matrix readout (roadmap 8.4).

A local model cannot see the Madelung potential of distant charges: a LiF nanocube or a polar
ZnO slab has bulk-like local environments in a different electrostatic potential (materials
bench). Here every atom carries a learned charge q_i (neutral per structure), the potential
V_i of the other charges is summed with Ewald, and the matrix gets, to first order in a slowly
varying potential,

    dH_ij = (V_i + V_j) / 2 * S_ij,

which needs the overlap S (the labels' last component; geometry only, the same S as H0's).
A constant V adds c S: a gauge shift, which gauge_invariant_mse already ignores.
"""

from __future__ import annotations

import math

import numpy as np
import torch

#: Coulomb constant, eV Angstrom.
COULOMB = 14.399645


def _kvectors(cell: torch.Tensor, kmax: float) -> torch.Tensor:
    """Reciprocal lattice vectors 0 < |k| <= kmax of ``cell`` (rows are lattice vectors)."""
    recip = 2 * math.pi * torch.linalg.inv(cell).T
    n_max = [int(math.ceil(kmax / float(torch.linalg.norm(b)))) for b in recip]
    ranges = [torch.arange(-n, n + 1, device=cell.device, dtype=cell.dtype) for n in n_max]
    n = torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1).reshape(-1, 3)
    k = n @ recip
    k2 = (k**2).sum(-1)
    return k[(k2 > 0) & (k2 <= kmax**2)]


def ewald_potentials(
    charges: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch: torch.Tensor,
    smearing: float = 2.0,
    tolerance: float = 1e-8,
) -> torch.Tensor:
    """Electrostatic potential (eV per unit charge) at every atom from the Gaussian-smeared
    charges of all the other atoms and their periodic images, for neutral periodic cells.

    The kernel is erf(r / smearing) / r, so the whole sum converges in reciprocal space:
    V_i = 4 pi k / Omega sum_{k != 0} exp(-k^2 s^2 / 4) / k^2 sum_j q_j cos(k (r_i - r_j))
    minus the self term q_i 2 / (s sqrt(pi)). The smearing removes the short range, which the
    local model describes anyway. Differentiable in ``charges``; the geometry is fixed.
    """
    kmax = 2.0 * math.sqrt(-math.log(tolerance)) / smearing
    n_structures = int(cell.shape[0] // 3)
    cells = cell.reshape(n_structures, 3, 3)
    index, out = [], []
    for s in range(n_structures):
        atoms = torch.nonzero(batch == s).reshape(-1)
        q, r = charges[atoms], positions[atoms]
        with torch.no_grad():
            k = _kvectors(cells[s].double(), kmax).to(r.dtype)
            k2 = (k**2).sum(-1)
            volume = torch.abs(torch.linalg.det(cells[s].double())).to(r.dtype)
            weight = 4 * math.pi / volume * torch.exp(-k2 * smearing**2 / 4) / k2
            phase = r @ k.T  # (n_atoms, n_k)
            cos, sin = torch.cos(phase), torch.sin(phase)
        # sum_j q_j cos(k (r_i - r_j)) = cos_i C + sin_i S with C = sum q cos, S = sum q sin
        c_k, s_k = q @ cos, q @ sin
        v = (cos * (weight * c_k)).sum(-1) + (sin * (weight * s_k)).sum(-1)
        v = v - q * 2.0 / (smearing * math.sqrt(math.pi))
        index.append(atoms)
        out.append(COULOMB * v)
    return charges.new_zeros(charges.shape).index_put((torch.cat(index),), torch.cat(out))


def neutralize(charges: torch.Tensor, batch: torch.Tensor, n_structures: int) -> torch.Tensor:
    """Charges minus their mean in each structure (sum zero)."""
    total = charges.new_zeros(n_structures).index_add(0, batch, charges)
    count = charges.new_zeros(n_structures).index_add(0, batch, torch.ones_like(charges))
    return charges - (total / count)[batch]


def label_element_atoms(point_types, edge_types, edge_index, n_edge_elements: int, basis_table):
    """Atom of every node label element, and the two atoms of every edge label element, in the
    order of the flattened labels (and of the readout's output): node blocks of each atom in
    turn; edge blocks of the labelled edges (all of them, or one per pair for a symmetric
    matrix, see ``metrics._label_edge_mask``)."""
    from graph2mat.core.data.metrics import _label_edge_mask

    point_types = np.asarray(point_types, dtype=np.int64)
    edge_types = np.asarray(edge_types, dtype=np.int64)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    node_atoms = np.repeat(np.arange(len(point_types)), basis_table.point_block_size[point_types])
    labelled = _label_edge_mask(np.empty(n_edge_elements), edge_types, edge_index, point_types, basis_table)
    sizes = basis_table.edge_block_size[np.abs(edge_types[labelled])]
    edge_i = np.repeat(edge_index[0, labelled], sizes)
    edge_j = np.repeat(edge_index[1, labelled], sizes)
    as_long = lambda a: torch.as_tensor(a, dtype=torch.long)
    return as_long(node_atoms), as_long(edge_i), as_long(edge_j)
