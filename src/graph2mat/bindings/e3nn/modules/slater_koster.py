"""Optional two-center Slater–Koster readout in the existing RTP basis."""

import torch
from e3nn import o3

from .edge_operations import E3nnLowRankEdgeBlock, _ordered_inputs
from .hamiltonian_readout import _extract_scalars, _scalar_dim

__all__ = ["E3nnSlaterKosterEdgeBlock"]


def _shells(point):
    offset = 0
    for mul, ir in point.e3nn_irreps:
        if ir.l > 3 or ir.p != (-1) ** ir.l:
            raise ValueError("SK requires natural-parity complete s/p/d/f shells.")
        for _ in range(mul):
            yield offset, offset + ir.dim, ir
            offset += ir.dim


def _sk_couplings(i_basis, j_basis, single_irreps_out, change_of_basis):
    """Small (L,q) transforms per radial shell pair, certified against Q.

    Each returned pair contains coefficient segments (start, stop, L) and a
    transform with min(la,lb)+1 columns (sigma, pi, delta, phi). No global KxK
    coefficient map is built. Q must use the existing offsite ``ij`` convention.
    """
    left, right = list(_shells(i_basis)), list(_shells(j_basis))
    irreps = o3.Irreps(single_irreps_out)
    Q = change_of_basis
    shape = (irreps.dim, i_basis.e3nn_irreps.dim, j_basis.e3nn_irreps.dim)
    if (not left or not right or not isinstance(Q, torch.Tensor)
            or Q.shape != shape or Q.dtype not in (torch.float32, torch.float64)
            or Q.requires_grad or not torch.isfinite(Q).all()):
        raise ValueError("SK calibration requires finite fixed real Q with matching shape.")
    Q = Q.detach()
    tolerance = 128 * torch.finfo(Q.dtype).eps
    n0 = Q.new_tensor([0., 1., 0.])
    angles = tuple(Q.new_tensor(value) for value in (.31, .73, -.29))
    instances, start = [], 0
    for mul, ir in irreps:
        for _ in range(mul):
            instances.append((start, start + ir.dim, ir))
            start += ir.dim
    groups = []
    assigned = set()
    for ra, ea, ia in left:
        for rb, eb, ib in right:
            selected = []
            for start, stop, ir in instances:
                local = Q[start:stop, ra:ea, rb:eb]
                if torch.count_nonzero(local):
                    if start in assigned:
                        raise ValueError("Q mixes distinct radial shell pairs.")
                    assigned.add(start)
                    selected.append((start, stop, ir))
            if sum(stop - start for start, stop, _ in selected) != ia.dim * ib.dim:
                raise ValueError("Q does not span the complete shell pair.")
            rows = torch.cat([Q[s:e, ra:ea, rb:eb].flatten(1) for s, e, _ in selected])
            identity = torch.eye(ia.dim * ib.dim, dtype=Q.dtype, device=Q.device)
            if not torch.allclose(rows @ rows.T, identity, atol=tolerance, rtol=tolerance):
                raise ValueError("Q shell-pair basis is not orthonormal.")
            # Axial calibration alone cannot detect permutations of nonaxial m.
            Di, Dj = ia.D_from_angles(*angles), ib.D_from_angles(*angles)
            for s, e, ir in selected:
                local = Q[s:e, ra:ea, rb:eb]
                rotated = torch.einsum("ai,kij,bj->kab", Di, local, Dj)
                expected = torch.einsum("lk,lij->kij", ir.D_from_angles(*angles), local)
                if (ir.p != ia.p * ib.p or not torch.allclose(
                        rotated, expected, atol=tolerance, rtol=tolerance)):
                    raise ValueError("Q is not an orbital/irrep rotation intertwiner.")
            allowed = set(range(abs(ia.l - ib.l), ia.l + ib.l + 1, 2))
            natural = [(s, e, ir.l) for s, e, ir in selected
                       if ir.l in allowed and ir.p == (-1) ** ir.l]
            if {L for _, _, L in natural} != allowed or len(natural) != len(allowed):
                raise ValueError("Q irreps disagree with SK angular selection.")
            transform = Q.new_zeros((len(natural), min(ia.l, ib.l) + 1))
            for q in range(transform.shape[1]):
                axis = Q.new_zeros(shape[1:])
                axis[ra + ia.l - q, rb + ib.l - q] = 1
                axis[ra + ia.l + q, rb + ib.l + q] = 1
                projected = torch.einsum("kij,ij->k", Q, axis)
                rebuilt = torch.zeros_like(projected)
                for k, (s, e, L) in enumerate(natural):
                    y = o3.spherical_harmonics(L, n0, True, normalization="component")
                    amplitude = (projected[s:e] * y).sum() / y.square().sum()
                    transform[k, q] = amplitude
                    rebuilt[s:e] = amplitude * y
                if (not torch.allclose(projected, rebuilt, atol=tolerance, rtol=tolerance)
                        or not torch.allclose(torch.einsum("k,kij->ij", rebuilt, Q),
                                              axis, atol=tolerance, rtol=tolerance)):
                    raise ValueError("Q fails SK axial calibration.")
            if int(torch.linalg.matrix_rank(transform)) != transform.shape[1]:
                raise ValueError("Q produces a rank-deficient SK calibration.")
            groups.append((tuple(natural), transform))
    if len(assigned) != len(instances):
        raise ValueError("Q has unassigned shell-pair coefficients.")
    return groups


class E3nnSlaterKosterEdgeBlock(torch.nn.Module):
    """Experimental neural two-center SK prior plus the existing 7.1 residual.

    Predicts sigma/pi/delta/phi integrals per radial shell pair and real matrix
    component. Angular dependence uses global spherical harmonics, not an
    arbitrary transverse frame. Anisotropic environment corrections remain in
    the low-rank equivariant residual. This is not a general SO(2) convolution
    or an equivalent replacement of the full-width default readout.

    Requires directed periodic ``edge_vectors`` in the feature/orbital basis
    and enveloped 0e ``edge_feats``. Both branches vanish at radial zero. Only
    real C1/C2, natural-parity s/p/d/f and offsite ``ij`` are supported.
    """

    _data_get_edge_args = ("edge_feats", "edge_vectors")
    requires_zero_edge_shift = True

    def __init__(self, irreps_in, irreps_out, edge_feats_irreps,
                 i_basis, j_basis, single_irreps_out, change_of_basis,
                 symmetry="ij", n_matrix_components=1, rank=4):
        super().__init__()
        if (symmetry != "ij" or isinstance(n_matrix_components, bool)
                or not isinstance(n_matrix_components, int) or n_matrix_components not in (1, 2)):
            raise ValueError("SK supports only offsite ij and real C1/C2 (not SOC).")
        self.single_irreps_out = o3.Irreps(single_irreps_out)
        self.irreps_out = o3.Irreps(irreps_out)
        self.n_matrix_components = n_matrix_components
        if self.irreps_out != self.single_irreps_out * n_matrix_components:
            raise ValueError("SK output irreps disagree with component layout.")
        groups = _sk_couplings(i_basis, j_basis, self.single_irreps_out, change_of_basis)
        self._sk_groups = []
        offset = 0
        for index, (segments, transform) in enumerate(groups):
            self.register_buffer(f"sk_transform_{index}", transform.clone())
            end = offset + transform.shape[1]
            self._sk_groups.append((segments, offset, end))
            offset = end
        self.n_integrals = offset
        self.sk_irreps = tuple({o3.Irrep(L, (-1)**L)
                                for segments, _, _ in self._sk_groups for _, _, L in segments})
        self.residual = E3nnLowRankEdgeBlock(irreps_in, irreps_out, edge_feats_irreps, rank=rank)
        self.input_irreps = self.residual.input_irreps
        condition_dim = self.residual.edge_feats_irreps.dim + 2 * sum(
            _scalar_dim(source) for source in self.input_irreps)
        self.sk_net = torch.nn.Sequential(
            torch.nn.Linear(condition_dim, 32), torch.nn.SiLU(),
            torch.nn.Linear(32, n_matrix_components * self.n_integrals),
        )

    def _sk_coefficients(self, integrals, vectors):
        # Scale before taking the norm: tiny nonzero vectors must retain direction.
        scaled = vectors / vectors.abs().amax(dim=-1, keepdim=True)
        unit = scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
        n, C = integrals.shape[:2]
        out = integrals.new_zeros((n, C, self.single_irreps_out.dim))
        harmonics = {}
        for index, (segments, begin, end) in enumerate(self._sk_groups):
            amplitudes = integrals[:, :, begin:end] @ getattr(self, f"sk_transform_{index}").T
            for k, (start, stop, L) in enumerate(segments):
                if L not in harmonics:
                    harmonics[L] = o3.spherical_harmonics(L, unit, False, normalization="component")
                out[:, :, start:stop] = amplitudes[:, :, k:k+1] * harmonics[L][:, None, :]
        return out.reshape(n, C * self.single_irreps_out.dim)

    def forward(self, edge_feats=None, edge_vectors=None, **tuple_kwargs):
        if (set(tuple_kwargs) - {"node_feats", "edge_messages"}
                or len(tuple_kwargs) != len(self.input_irreps)):
            raise ValueError("SK sources must match declared node_feats/edge_messages irreps.")
        pairs = _ordered_inputs(tuple_kwargs)
        if (not isinstance(pairs[0], tuple) or len(pairs[0]) != 2
                or not isinstance(pairs[0][0], torch.Tensor) or pairs[0][0].ndim != 2):
            raise ValueError("SK sources must be pairs of 2D tensors.")
        n = pairs[0][0].shape[0]
        reference = self.sk_net[0].weight
        def validate(name, pair, dim):
            self.residual._validate_pair(name, pair, (n, dim))
            if any(v.dtype != reference.dtype or v.device != reference.device
                   or not torch.isfinite(v).all() for v in pair):
                raise ValueError(f"SK {name} must be finite real tensors matching model dtype/device.")
        validate("edge_feats", edge_feats, self.residual.edge_feats_irreps.dim)
        validate("edge_vectors", edge_vectors, 3)
        for source, pair in zip(self.input_irreps, pairs):
            validate("source", pair, source.dim)
        v, reverse = edge_vectors
        if torch.any(v.abs().amax(-1) == 0):
            raise ValueError("SK edge_vectors must have nonzero length.")
        if not torch.allclose(reverse, -v, rtol=32*torch.finfo(v.dtype).eps, atol=0):
            raise ValueError("SK directed edge_vectors must be opposite paired directions.")
        radial = (edge_feats[0] + edge_feats[1]) * 0.5
        condition = [radial]
        for source, pair in zip(self.input_irreps, pairs):
            for value in pair:
                condition.extend(_extract_scalars(value, source))
        gate = torch.tanh(radial.square().sum(-1, keepdim=True))
        integrals = (self.sk_net(torch.cat(condition, -1)) * gate).reshape(
            n, self.n_matrix_components, self.n_integrals)
        return self._sk_coefficients(integrals, v) + self.residual(edge_feats=edge_feats, **tuple_kwargs)
