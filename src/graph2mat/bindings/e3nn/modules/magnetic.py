"""Node features from per-atom magnetic moments for the matrix readout (roadmap 8.1, magnetism)."""
import torch
from e3nn import nn, o3

__all__ = ["MagneticMomentLayer"]

#: Irreps of one moment: a signed scalar (collinear) or an axial vector (with spin-orbit).
MOMENT_IRREPS = {"collinear": o3.Irreps("1x0e"), "vector": o3.Irreps("1x1e")}


class MagneticMomentLayer(torch.nn.Module):
    """Local features linear in the magnetic moments ``m``, for one structure graph.

    ``out_i = TP(node_feats_i, m_i) + sum_{j -> i} TP_{R(r_ij)}(m_j, Y(r_ij)) / avg_num_neighbors``

    The self term couples ``m_i`` with the (geometric) node features of ``i``; the neighbour term
    adds the moments of the neighbours ``j`` along the edges ``j -> i`` of the graph, weighted by
    a radial network of the edge features. Moments come in the data frame, like the positions:
    shape ``(n,)`` (``"collinear"``, irreps ``0e``) or ``(n, 3)`` (``"vector"``, ``1e``).

    Output irreps: every irrep of ``node_feats x m``, with the largest multiplicity of the node
    features. Linear in ``m`` (no biases): ``layer(-m) = -layer(m)``.
    """

    def __init__(self, kind: str, node_feats_irreps, edge_attrs_irreps, edge_feats_irreps,
                 avg_num_neighbors: float, radial_hidden: int = 64):
        super().__init__()
        if kind not in MOMENT_IRREPS:
            raise ValueError(f"magnetic_moments must be one of {sorted(MOMENT_IRREPS)}, got {kind!r}.")
        self.kind = kind
        moment = MOMENT_IRREPS[kind]
        node_feats_irreps = o3.Irreps(node_feats_irreps)
        irreps = sorted({ir for _, node_ir in node_feats_irreps for ir in node_ir * moment[0].ir})
        self.irreps_out = o3.Irreps([(max(mul for mul, _ in node_feats_irreps), ir) for ir in irreps])
        self.avg_num_neighbors = avg_num_neighbors
        self.self_tp = o3.FullyConnectedTensorProduct(node_feats_irreps, moment, self.irreps_out)
        self.neighbour_tp = o3.FullyConnectedTensorProduct(
            moment, edge_attrs_irreps, self.irreps_out, shared_weights=False, internal_weights=False
        )
        self.radial = nn.FullyConnectedNet(
            [o3.Irreps(edge_feats_irreps).dim, radial_hidden, self.neighbour_tp.weight_numel],
            torch.nn.functional.silu,
        )

    def forward(self, node_feats: torch.Tensor, moments: torch.Tensor, edge_index: torch.Tensor,
                edge_attrs: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        n = len(node_feats)
        if moments.shape != ((n, 3) if self.kind == "vector" else (n,)):
            raise ValueError(
                f"magnetic_moments={self.kind!r} needs moments of shape {(n, 3) if self.kind == 'vector' else (n,)}, "
                f"got {tuple(moments.shape)}."
            )
        m = moments.reshape(n, -1).to(node_feats.dtype)
        sender, receiver = edge_index
        messages = self.neighbour_tp(m[sender], edge_attrs, self.radial(edge_feats))
        neighbours = messages.new_zeros(n, messages.shape[-1]).index_add_(0, receiver, messages)
        return self.self_tp(node_feats, m) + neighbours / self.avg_num_neighbors
