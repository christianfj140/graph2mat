"""Optional two-center Slater–Koster readout in the existing RTP basis."""

import torch
from e3nn import o3


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
