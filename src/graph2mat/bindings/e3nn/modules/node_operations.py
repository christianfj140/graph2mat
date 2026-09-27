from e3nn import o3
import torch

__all__ = [
    "E3nnSimpleNodeBlock",
    "E3nnLinearTSQNodeBlock",
    "E3nnProjectedLinearTSQNodeBlock",
    "E3nnSeparateTSQNodeBlock",
]


class E3nnSimpleNodeBlock(torch.nn.Module):
    """Sums all node features and then passes them to a tensor square.

    All node features must have the same irreps.

    Example
    -------
    If we construct a SimpleNodeBlock:

    >>> irreps_in = o3.Irreps("2x0e + 2x1o")
    >>> irreps_out = o3.Irreps("3x2e")
    >>> node_block = SimpleNodeBlock(irreps_in, irreps_out)

    and then use it with 2 different nodewise tensors:

    >>> node_feats = torch.randn(10, irreps_in.dim)
    >>> node_messages = torch.randn(10, irreps_in.dim)
    >>> node_block(node_feats=node_feats, node_messages=node_messages)

    this is equivalent to:

    >>> tsq = o3.TensorSquare(irreps_in, irreps_out)
    >>> output = tsq(node_feats + node_messages)
    """

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        if isinstance(irreps_in, (list, tuple)) and not isinstance(
            irreps_in, o3.Irreps
        ):
            assert all(
                irreps == irreps_in[0] for irreps in irreps_in
            ), "All input irreps must be the same."
            irreps_in = irreps_in[0]

        self.tsq = o3.TensorSquare(irreps_in, irreps_out)

    def forward(self, **node_kwargs: torch.Tensor) -> torch.Tensor:
        node_tensors = iter(node_kwargs.values())

        node_feats = next(node_tensors)
        for other_node_feats in node_tensors:
            node_feats = node_feats + other_node_feats

        return self.tsq(node_feats)


class E3nnLinearTSQNodeBlock(E3nnSimpleNodeBlock):
    """E3nnSimpleNodeBlock plus a linear term with biases on the 0e outputs.

    To first order the crystal field is linear in the neighbour density, and the
    atomic on-site energies are a constant per species (one node operation per
    species with ``basis_grouping="point_type"``). A tensor square alone can only
    produce both indirectly.
    """

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__(irreps_in, irreps_out)
        self.linear = o3.Linear(self.tsq.irreps_in, irreps_out, biases=True)

    def forward(self, **node_kwargs: torch.Tensor) -> torch.Tensor:
        node_feats = sum(node_kwargs.values())
        return self.tsq(node_feats) + self.linear(node_feats)


class E3nnProjectedLinearTSQNodeBlock(torch.nn.Module):
    """E3nnLinearTSQNodeBlock with the tensor square on a projection of the features.

    The tensor square costs (and has parameters) quadratic in the number of channels,
    which with the concatenated features of several MACE interactions is most of the
    readout time. Its input is first projected to ``mid_multiplicity`` channels per
    irrep; the linear term (and the 0e biases) still see the full features.
    """

    mid_multiplicity = 16

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        if isinstance(irreps_in, (list, tuple)) and not isinstance(irreps_in, o3.Irreps):
            assert all(irreps == irreps_in[0] for irreps in irreps_in), "All input irreps must be the same."
            irreps_in = irreps_in[0]
        irreps_in = o3.Irreps(irreps_in)
        mid = o3.Irreps([(self.mid_multiplicity, ir) for ir in sorted({ir for _, ir in irreps_in})])
        self.project = o3.Linear(irreps_in, mid)
        self.tsq = o3.TensorSquare(mid, irreps_out)
        self.linear = o3.Linear(irreps_in, irreps_out, biases=True)

    def forward(self, **node_kwargs: torch.Tensor) -> torch.Tensor:
        node_feats = sum(node_kwargs.values())
        return self.tsq(self.project(node_feats)) + self.linear(node_feats)


class E3nnSeparateTSQNodeBlock(torch.nn.Module):
    """Tensor squares each node features and then sums all outputs.

    Example
    -------
    If we construct a SeparateTSQNodeBlock:

    >>> irreps_in = o3.Irreps("3x0e + 2x1o")
    >>> irreps_out = o3.Irreps("3x2e")
    >>> node_block = SeparateTSQNodeBlock(irreps_in, irreps_out)

    and then use it with 2 different nodewise tensors:

    >>> node_feats = torch.randn(10, irreps_in.dim)
    >>> node_messages = torch.randn(10, irreps_in.dim)
    >>> output = node_block(node_feats=node_feats, node_messages=node_messages)

    this is equivalent to:

    >>> tsq1 = o3.TensorSquare(irreps_in, irreps_out)
    >>> tsq2 = o3.TensorSquare(irreps_in, irreps_out)
    >>> output = tsq1(node_feats) + tsq2(node_messages)
    """

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        if isinstance(irreps_in, (o3.Irreps, str)):
            irreps_in = [irreps_in]

        self.tensor_squares = torch.nn.ModuleList(
            [
                o3.TensorSquare(this_irreps_in, irreps_out)
                for this_irreps_in in irreps_in
            ]
        )

    def forward(self, **node_kwargs: torch.Tensor) -> torch.Tensor:
        assert len(node_kwargs) == len(
            self.tensor_squares
        ), f"Number of input tensors ({len(node_kwargs)}) must match number of tensor square operations ({len(self.tensor_squares)})."

        node_tensors = iter(node_kwargs.values())

        node_feats = self.tensor_squares[0](next(node_tensors))
        for i, other_node_feats in enumerate(node_tensors):
            node_feats = node_feats + self.tensor_squares[i + 1](other_node_feats)

        return node_feats
