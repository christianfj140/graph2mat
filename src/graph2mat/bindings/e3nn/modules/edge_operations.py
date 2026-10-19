"""E3nn operations to compute edge matrix blocks.

In edge matrix blocks, you tipically will have, for each edge,
a different message coming from each atom in the edge. The edge block
will tipically not be symmetric, but it is common that.

.. math::
    B_{ij} = B_{ji}^T
"""

from e3nn import o3, nn
import torch

from typing import Optional, Tuple

from ._utils import tp_out_irreps_with_instructions

__all__ = [
    "E3nnSimpleEdgeBlock",
    "E3nnLinearEdgeBlock",
    "E3nnLowRankEdgeBlock",
    "E3nnEdgeBlockNodeMix",
]


# Order of the inputs in ``irreps_in`` (see E3nnIrrepsMatrixBlock.get_init_kwargs). Inputs
# arrive as keyword arguments whose order depends on the caller, so they are matched by name.
_INPUT_ORDER = ("node_feats", "edge_messages")


def _ordered_inputs(tuple_kwargs: dict) -> list:
    keys = sorted(tuple_kwargs, key=lambda k: _INPUT_ORDER.index(k) if k in _INPUT_ORDER else len(_INPUT_ORDER))
    return [tuple_kwargs[k] for k in keys]


class E3nnSimpleEdgeBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        if isinstance(irreps_in, (o3.Irreps, str)):
            irreps_in = [irreps_in]

        self.tensor_products = torch.nn.ModuleList(
            [
                o3.FullyConnectedTensorProduct(
                    this_irreps_in, this_irreps_in, irreps_out
                )
                for this_irreps_in in irreps_in
            ]
        )

    def forward(
        self, **tuple_kwargs: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        assert len(tuple_kwargs) == len(
            self.tensor_products
        ), f"Number of input tuples ({len(tuple_kwargs)}) must match number of tensor square operations ({len(self.tensor_products)})."

        tensor_tuples = iter(_ordered_inputs(tuple_kwargs))

        final_value = self.tensor_products[0](*next(tensor_tuples))
        for i, tensor_tuple in enumerate(tensor_tuples):
            final_value = final_value + self.tensor_products[i + 1](*tensor_tuple)

        return final_value


class E3nnLinearEdgeBlock(E3nnSimpleEdgeBlock):
    """E3nnSimpleEdgeBlock plus a linear term of the sum of the two inputs.

    Without it the edge block is a homogeneous quadratic form of the messages.
    There are no biases: edge blocks must decay to zero with distance, as the
    messages do.

    The enclosing matrix block may opt into certified homonuclear weight
    projection; it keeps these original Parameters and their optimizer history.
    """

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__(irreps_in, irreps_out)
        self.linears = torch.nn.ModuleList(
            [o3.Linear(tp.irreps_in1, irreps_out) for tp in self.tensor_products]
        )
        self._homonuclear_ready = False
        for layer in [*self.tensor_products, *self.linears]:
            layer.register_buffer("_homonuclear_index", torch.empty(0, dtype=torch.long), persistent=False)
            layer.register_buffer("_homonuclear_sign", torch.empty(0), persistent=False)

    def _prepare_homonuclear_fusion(self, index, sign) -> Optional[str]:
        """Certify a signed involution of the existing, normalized weight paths."""
        self._homonuclear_ready = False
        layers = [*self.tensor_products, *self.linears]
        for layer in layers:
            layer._homonuclear_index = torch.empty(0, dtype=torch.long, device=layer.weight.device)
            layer._homonuclear_sign = layer.weight.new_empty(0)
        index, sign = index.detach().cpu(), sign.detach().cpu()
        irreps = self.tensor_products[0].irreps_out
        if (index.dtype != torch.long or index.shape != (irreps.dim,) or sign.shape != index.shape
                or not torch.equal(index.sort().values, torch.arange(irreps.dim))
                or not torch.all((sign == 1) | (sign == -1))
                or not torch.equal(index[index], torch.arange(irreps.dim))
                or not torch.all(sign * sign[index] == 1)):
            return "invalid coefficient involution"
        output_maps = []
        offset = 0
        for mul, ir in irreps:
            local = index[offset:offset + mul * ir.dim].reshape(mul, ir.dim) - offset
            signs = sign[offset:offset + mul * ir.dim].reshape(mul, ir.dim)
            permutation = local[:, 0] // ir.dim
            if (torch.any(permutation < 0) or torch.any(permutation >= mul)
                    or not torch.equal(local, permutation[:, None] * ir.dim + torch.arange(ir.dim))
                    or not torch.all(signs == signs[:, :1])):
                return "transpose does not preserve irrep segments and magnetic components"
            output_maps.append((permutation, signs[:, 0]))
            offset += mul * ir.dim

        prepared = []
        for layer in layers:
            if not layer.internal_weights or not layer.shared_weights:
                return "weights must be internal and shared"
            is_tp = isinstance(layer, o3.TensorProduct)
            if not is_tp and layer.bias.numel():
                return "linear bias is unsupported"
            flat = torch.arange(layer.weight_numel)
            lookup = {(r.i_in1, r.i_in2, r.i_out): n for n, r in enumerate(layer.instructions)} if is_tp else {}
            indices, signs = [], []
            for n, instruction in enumerate(layer.instructions):
                permutation, output_sign = output_maps[instruction.i_out]
                phase = 1
                if is_tp:
                    if instruction.connection_mode != "uvw" or not instruction.has_weight:
                        return "only weighted uvw paths are supported"
                    reverse = lookup.get((instruction.i_in2, instruction.i_in1, instruction.i_out))
                    if reverse is None:
                        return "missing reverse tensor product path"
                    other = layer.instructions[reverse]
                    if instruction.path_weight <= 0 or instruction.path_weight != other.path_weight:
                        return "unequal or nonpositive reverse path normalization"
                    view = layer.weight_view_for_instruction(reverse, weight=flat).transpose(0, 1)
                    phase = (-1) ** (layer.irreps_in1[instruction.i_in1].ir.l
                                    + layer.irreps_in2[instruction.i_in2].ir.l
                                    - layer.irreps_out[instruction.i_out].ir.l)
                    signs.append((phase * output_sign)[None, None, :].expand(instruction.path_shape).reshape(-1))
                else:
                    view = layer.weight_view_for_instruction(n, weight=flat)
                    signs.append(output_sign[None, :].expand(instruction.path_shape).reshape(-1))
                indices.append(view[..., permutation].reshape(-1))
            mapping = torch.cat(indices) if indices else flat
            multiplier = torch.cat(signs) if signs else torch.empty(0)
            if (mapping.numel() != layer.weight_numel or not torch.equal(mapping.sort().values, flat)
                    or not torch.equal(mapping[mapping], flat)
                    or not torch.all(multiplier * multiplier[mapping] == 1)):
                return "weight mapping is not a signed involution"
            prepared.append((mapping, multiplier))
        for layer, (mapping, multiplier) in zip(layers, prepared):
            layer._homonuclear_index = mapping.to(layer.weight.device)
            layer._homonuclear_sign = multiplier.to(layer.weight)
        self._homonuclear_ready = True
        return None

    @staticmethod
    def _projected_weight(layer):
        return 0.5 * layer.weight + 0.5 * layer.weight[layer._homonuclear_index] * layer._homonuclear_sign

    def forward(
        self, *, _homonuclear: bool = False, **tuple_kwargs: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if _homonuclear:
            if not self._homonuclear_ready:
                raise RuntimeError("Homonuclear weight fusion has not been certified.")
            assert len(tuple_kwargs) == len(self.tensor_products)
            pairs = _ordered_inputs(tuple_kwargs)
            out = self.tensor_products[0](*pairs[0], weight=self._projected_weight(self.tensor_products[0]))
            for tp, pair in zip(self.tensor_products[1:], pairs[1:]):
                out = out + tp(*pair, weight=self._projected_weight(tp))
            for linear, (a, b) in zip(self.linears, pairs):
                out = out + linear(a + b, weight=self._projected_weight(linear))
            return out
        out = super().forward(**tuple_kwargs)
        for linear, (a, b) in zip(self.linears, _ordered_inputs(tuple_kwargs)):
            out = out + linear(a + b)
        return out


class E3nnLowRankEdgeBlock(torch.nn.Module):
    """Experimental low-rank bilinear readout with pair-specific radial gates.

    Each input irrep is projected to ``rank`` channels separately at the two
    ends. Channel-aligned products retain every allowed angular coupling, but
    share input projections: this is a capacity restriction, not an equivalent
    replacement for ``E3nnLinearEdgeBlock``. A bias-free radial MLP and ``tanh``
    supply per-edge TP weights and a gate for a full-width linear skip. All
    terms vanish when the radial features vanish, including with reused nodes.

    Select with ``edge_operation`` (or Lightning's ``edge_block_readout``).
    The enclosing matrix block handles exchange/transpose symmetry. Separate
    block operations learn separate pair weights, unless basis grouping shares
    those operations. Radial inputs must be even scalars (0e).
    """

    _data_get_edge_args = ("edge_feats",)

    def __init__(self, irreps_in, irreps_out, edge_feats_irreps, rank: int = 4):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer.")
        if isinstance(irreps_in, (o3.Irreps, str)):
            irreps_in = [irreps_in]
        self.input_irreps = [o3.Irreps(value) for value in irreps_in]
        if not 1 <= len(self.input_irreps) <= 2 or any(value.dim == 0 for value in self.input_irreps):
            raise ValueError("Expected one or two nonempty input sources (node_feats, edge_messages).")
        self.irreps_out = o3.Irreps(irreps_out)
        self.edge_feats_irreps = o3.Irreps(edge_feats_irreps)
        if not self.edge_feats_irreps.dim or any(ir != o3.Irrep("0e") for _, ir in self.edge_feats_irreps):
            raise ValueError("edge_feats_irreps must contain only nonempty even scalars (0e).")

        self.left_projections = torch.nn.ModuleList()
        self.right_projections = torch.nn.ModuleList()
        self.tensor_products = torch.nn.ModuleList()
        self.output_linears = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.radial_gates = torch.nn.ModuleList()
        for source in self.input_irreps:
            projected = o3.Irreps([(rank, ir) for ir in sorted({ir for mul, ir in source if mul})])
            mid, instructions = [], []
            for i, (_, left) in enumerate(projected):
                for j, (_, right) in enumerate(projected):
                    for out in left * right:
                        if out in self.irreps_out:
                            instructions.append((i, j, len(mid), "uuu", True))
                            mid.append((rank, out))
            mid = o3.Irreps(mid)
            tp = o3.TensorProduct(projected, projected, mid, instructions,
                                  internal_weights=False, shared_weights=False)
            self.left_projections.append(o3.Linear(source, projected))
            self.right_projections.append(o3.Linear(source, projected))
            self.tensor_products.append(tp)
            self.output_linears.append(o3.Linear(mid, self.irreps_out))
            self.linears.append(o3.Linear(source, self.irreps_out))
            self.radial_gates.append(torch.nn.Sequential(
                torch.nn.Linear(self.edge_feats_irreps.dim, 32, bias=False),
                torch.nn.SiLU(),
                torch.nn.Linear(32, tp.weight_numel + 1, bias=False),
            ))

    @staticmethod
    def _validate_pair(name, pair, shape):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{name} must be a two-tensor tuple.")
        if any(not isinstance(value, torch.Tensor) or value.shape != shape for value in pair):
            raise ValueError(f"{name} input shape must be {tuple(shape)} at both ends.")

    def forward(self, edge_feats=None, **tuple_kwargs):
        if set(tuple_kwargs) - set(_INPUT_ORDER) or len(tuple_kwargs) != len(self.input_irreps):
            raise ValueError("Input sources must match the declared node_feats/edge_messages irreps.")
        pairs = _ordered_inputs(tuple_kwargs)
        if not isinstance(pairs[0], tuple) or len(pairs[0]) != 2 or not isinstance(pairs[0][0], torch.Tensor) or pairs[0][0].ndim != 2:
            raise ValueError("Each input source must be a tuple of two 2D tensors.")
        n = pairs[0][0].shape[0]
        self._validate_pair("edge_feats", edge_feats, (n, self.edge_feats_irreps.dim))
        radial = (edge_feats[0] + edge_feats[1]) * 0.5
        result = radial.new_zeros((n, self.irreps_out.dim))
        for source, pair, left, right, tp, output, linear, gate in zip(
            self.input_irreps, pairs, self.left_projections, self.right_projections,
            self.tensor_products, self.output_linears, self.linears, self.radial_gates,
        ):
            self._validate_pair("source", pair, (n, source.dim))
            a, b = pair
            weights = torch.tanh(gate(radial))
            result = result + output(tp(left(a), right(b), weights[:, :-1]))
            result = result + linear(a + b) * weights[:, -1:]
        return result


class E3nnEdgeBlockNodeMix(torch.nn.Module):
    _data_get_edge_args = ("edge_feats",)

    def __init__(
        self,
        edge_feats_irreps: o3.Irreps,
        edge_messages_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()

        # Network to reduce node representations to scalar features
        self.nodes_linear = o3.Linear(node_feats_irreps, edge_feats_irreps)

        # The weights of the tensor are produced by a fully connected neural network
        # that takes the scalar representations of nodes and edges as input
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            edge_messages_irreps,
            edge_messages_irreps,
            irreps_out,
        )
        # Tensor product between edge features from sender and receiver
        self.edges_tp = o3.TensorProduct(
            edge_messages_irreps,
            edge_messages_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        irreps_mid = irreps_mid.simplify()

        edge_tp_input_irreps = edge_feats_irreps * 3
        assert edge_tp_input_irreps.lmax == 0
        input_dim = edge_tp_input_irreps.num_irreps
        self.edge_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 2 * [128] + [self.edges_tp.weight_numel],
            torch.nn.SiLU(),
        )

        # The final output is produced by a linear layer
        self.output_linear = o3.Linear(irreps_mid, irreps_out)

    def forward(
        self,
        edge_feats: Tuple[torch.Tensor, torch.Tensor],
        edge_messages: Tuple[torch.Tensor, torch.Tensor],
        node_feats: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Convert nodes to scalar features
        scalar_node_feats_sender = self.nodes_linear(node_feats[0])
        scalar_node_feats_receiver = self.nodes_linear(node_feats[1])
        scalar_feats = torch.concatenate(
            (scalar_node_feats_sender, scalar_node_feats_receiver, edge_feats[0]), dim=1
        )
        # Obtain weights for edge tensor product
        edge_tp_weights = self.edge_tp_weights(scalar_feats)

        # Compute edge tensor product
        edges_tp = self.edges_tp(edge_messages[0], edge_messages[1], edge_tp_weights)

        # Compute final output
        output = self.output_linear(edges_tp)

        return output
