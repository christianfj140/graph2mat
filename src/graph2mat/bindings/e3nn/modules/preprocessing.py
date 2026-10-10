from typing import Tuple, Union, Dict

import torch
from e3nn import o3, nn

from graph2mat.bindings.torch import TorchBasisMatrixData
from ._utils import tp_out_irreps_with_instructions, scatter_sum

__all__ = [
    "E3nnInteraction",
    "E3nnEdgeMessageBlock",
]


class E3nnInteraction(torch.nn.Module):
    """Basically MACE's RealAgnosticResidualInteractionBlock, without reshapes.

    This function takes a graph and returns new states for the nodes. It returns only
    the neighbours' message, without MACE's residual/skip connection (audit ARCH-26):
    used as node preprocessing, the atom's own features are dropped.

    This function can be used for the preprocessing step of both nodes and edges.
    """

    def __init__(
        self,
        irreps: Dict[str, o3.Irreps],
        avg_num_neighbors: float = 10,
    ) -> None:
        super().__init__()

        node_feats_irreps = irreps["node_feats_irreps"]
        # node_attrs_irreps = irreps["node_attrs_irreps"]
        edge_attrs_irreps = irreps["edge_attrs_irreps"]
        edge_feats_irreps = irreps["edge_feats_irreps"]
        target_irreps = irreps["node_feats_irreps"]
        # hidden_irreps = irreps["node_feats_irreps"]

        # First linear
        self.linear_up = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps,
            edge_attrs_irreps,
            target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            node_feats_irreps,
            edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.SiLU(),
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_mji = irreps_mid
        self.linear = o3.Linear(
            irreps_mid, target_irreps, internal_weights=True, shared_weights=True
        )

        self.avg_num_neighbors = avg_num_neighbors

        self.irreps_out = target_irreps

    def forward(
        self,
        data: TorchBasisMatrixData,
        node_feats: torch.Tensor,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        edge_attrs = data["edge_attrs"]
        edge_feats = data["edge_feats"]

        sender, receiver = data["edge_index"]
        num_nodes = node_feats.shape[0]

        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        del tp_weights
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        del mji
        message = self.linear(message) / self.avg_num_neighbors

        return message


class E3nnEdgeMessageBlock(torch.nn.Module):
    """This is basically MACE's RealAgnosticResidualInteractionBlock, but only up to the part
    where it computes the partial mji messages.

    It computes a "message" for each edge in the graph. Note that the message
    is different for the edge (i, j) and the edge (j, i).

    This function can be used for the preprocessing step of edges. It has no effect when used
    as the preprocessing step of nodes.

    If ``irreps["edge_message_mid_irreps"]`` is given, node features are projected to those
    irreps before being copied to every edge (an int means that multiplicity for every irrep
    of the node features). Everything computed per edge (gather, tensor product weights and
    output) scales with this width, so this is the main VRAM knob of the edge readout.

    ``irreps["edge_update_layers"]`` (default 0) adds residual edge update layers
    (roadmap 7.2, audit ARCH-11): ``m <- m + act(Linear(TP(m, P(h_receiver); w(r))))``. The
    message then depends jointly on both ends of the edge (the readout only multiplies
    the two messages of a pair, which are separable). ``P`` projects the receiver features
    to ``irreps["edge_update_node_irreps"]`` (an int: that multiplicity for every irrep of
    the messages; default 8). No biases: messages still vanish at the cutoff.
    """

    def __init__(
        self,
        irreps: Dict[str, o3.Irreps],
    ) -> None:
        super().__init__()

        node_feats_irreps = irreps["node_feats_irreps"]
        edge_attrs_irreps = irreps["edge_attrs_irreps"]
        edge_feats_irreps = irreps["edge_feats_irreps"]
        target_irreps = irreps["edge_hidden_irreps"]
        mid_irreps = self._mid_irreps(
            node_feats_irreps, irreps.get("edge_message_mid_irreps")
        )

        # First linear (on nodes, before the per-edge gather)
        self.linear_up = o3.Linear(
            node_feats_irreps,
            mid_irreps,
        )

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            mid_irreps,
            edge_attrs_irreps,
            target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            mid_irreps,
            edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        assert (
            edge_feats_irreps.lmax == 0
        ), "Edge features must be a scalar array to preserve equivariance"
        input_dim = edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.SiLU(),
        )

        irreps_mid = irreps_mid.simplify()

        self.linear = o3.Linear(irreps_mid, target_irreps)

        n_updates = int(irreps.get("edge_update_layers") or 0)
        update_spec = irreps.get("edge_update_node_irreps") or 8
        if isinstance(update_spec, int):
            update_irreps = o3.Irreps([(update_spec, ir) for ir in sorted({ir for _, ir in target_irreps})])
        else:
            update_irreps = o3.Irreps(update_spec)
        self.update_node_linears = torch.nn.ModuleList()
        self.update_tps = torch.nn.ModuleList()
        self.update_tp_weights = torch.nn.ModuleList()
        self.update_linears = torch.nn.ModuleList()
        self.update_acts = torch.nn.ModuleList()
        for _ in range(n_updates):
            self.update_node_linears.append(o3.Linear(node_feats_irreps, update_irreps))
            tp_irreps, tp_instructions = tp_out_irreps_with_instructions(target_irreps, update_irreps, target_irreps)
            tp = o3.TensorProduct(target_irreps, update_irreps, tp_irreps, instructions=tp_instructions,
                                  shared_weights=False, internal_weights=False)
            self.update_tps.append(tp)
            self.update_tp_weights.append(nn.FullyConnectedNet([input_dim] + 2 * [64] + [tp.weight_numel], torch.nn.SiLU()))
            self.update_linears.append(o3.Linear(tp_irreps.simplify(), target_irreps))
            self.update_acts.append(nn.NormActivation(target_irreps, torch.nn.functional.silu))

        self.irreps_out = (None, target_irreps)

    @staticmethod
    def _mid_irreps(node_feats_irreps: o3.Irreps, spec) -> o3.Irreps:
        """Irreps of the node features copied to each edge (default: node_feats_irreps)."""
        if spec is None:
            return node_feats_irreps
        if isinstance(spec, int):
            present = sorted({ir for _, ir in o3.Irreps(node_feats_irreps)})
            return o3.Irreps([(spec, ir) for ir in present])
        return o3.Irreps(spec)

    def forward(
        self,
        data: TorchBasisMatrixData,
        node_feats: torch.Tensor,
    ) -> Tuple[None, torch.Tensor]:
        sender, receiver = data["edge_index"]

        edge_attrs = data["edge_attrs"]
        edge_feats = data["edge_feats"]

        data_node_feats = node_feats
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        del tp_weights

        messages = self.linear(mji)
        for node_linear, tp, weights, linear, act in zip(
            self.update_node_linears, self.update_tps, self.update_tp_weights, self.update_linears, self.update_acts
        ):
            update = tp(messages, node_linear(data_node_feats)[receiver], weights(edge_feats))
            messages = messages + act(linear(update))

        # The first return is the node features [n_nodes, irreps], which we don't compute
        # The second return are the edge messages [n_edges, irreps]
        return None, messages
