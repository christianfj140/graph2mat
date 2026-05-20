"""Experimental Hamiltonian coefficient readouts.

These blocks are opt-in production candidates for Hamiltonian learning. They
use e3nn tensor products, but make the tensor-product weights local by
conditioning them on scalar ``0e`` channels from the readout inputs. When those
scalar channels come from invariant local context, the output coefficients keep
the equivariant transformation behavior of ``irreps_out`` while avoiding the
fixed shared-weight bottleneck of the default readout.
"""

from __future__ import annotations

from typing import Iterable, Optional, Type, Union

import torch
from e3nn import o3

from ._utils import tp_out_irreps_with_instructions

__all__ = [
    "E3nnHamiltonianNodeBlock",
    "E3nnHamiltonianEdgeBlock",
]


IrrepsLike = Union[o3.Irreps, str]


def _as_irreps_list(irreps_in: Union[IrrepsLike, Iterable[IrrepsLike]]):
    if isinstance(irreps_in, (o3.Irreps, str)):
        return [o3.Irreps(irreps_in)]
    return [o3.Irreps(irreps) for irreps in irreps_in]


def _scalar_slices(irreps: o3.Irreps) -> list[slice]:
    return [
        irrep_slice
        for irrep_slice, (_mul, irrep) in zip(irreps.slices(), irreps)
        if irrep.l == 0 and irrep.p == 1
    ]


def _scalar_dim(irreps: o3.Irreps) -> int:
    return sum(
        irrep_slice.stop - irrep_slice.start
        for irrep_slice in _scalar_slices(irreps)
    )


def _extract_scalars(tensor: torch.Tensor, irreps: o3.Irreps) -> list[torch.Tensor]:
    return [tensor[:, irrep_slice] for irrep_slice in _scalar_slices(irreps)]


def _condition_dim(
    irreps: o3.Irreps,
    n_tensors: int,
    require_scalar_context: bool,
    block_name: str,
) -> int:
    dim = n_tensors * _scalar_dim(irreps)
    if dim == 0:
        if require_scalar_context:
            raise ValueError(
                f"{block_name} requires scalar 0e context, but no scalar "
                "channels were found in irreps_in."
            )
        return 1
    return int(dim)


def _condition_from_tensors(
    tensors: list[torch.Tensor],
    irreps: o3.Irreps,
) -> torch.Tensor:
    scalars = []
    for tensor in tensors:
        scalars.extend(_extract_scalars(tensor, irreps))
    if scalars:
        return torch.cat(scalars, dim=-1)
    return tensors[0].new_ones((tensors[0].shape[0], 1))


def _validate_hidden_dim(
    hidden_dim: Optional[int],
    output_dim: int,
    allow_bottleneck: bool,
) -> None:
    if hidden_dim is None or allow_bottleneck:
        return
    if int(hidden_dim) < int(output_dim):
        raise ValueError(
            "hidden_dim is smaller than irreps_out.dim. Pass "
            "allow_bottleneck=True to make this explicit."
        )


def _default_hidden_dim(
    condition_dim: int,
    weight_dim: int,
    output_dim: int,
    hidden_dim: Optional[int],
) -> int:
    if hidden_dim is not None:
        return int(hidden_dim)
    return max(int(condition_dim), int(weight_dim), int(output_dim), 1)


class _ConditionedTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_left: o3.Irreps,
        irreps_right: o3.Irreps,
        irreps_out: o3.Irreps,
        condition_dim: int,
        hidden_dim: Optional[int],
        activation: Type[torch.nn.Module],
        allow_bottleneck: bool,
    ):
        super().__init__()

        _validate_hidden_dim(hidden_dim, irreps_out.dim, allow_bottleneck)

        irreps_mid, instructions = tp_out_irreps_with_instructions(
            irreps_left,
            irreps_right,
            irreps_out,
        )
        self.enabled = len(instructions) > 0
        self.output_dim = int(irreps_out.dim)
        if len(instructions) == 0:
            return

        self.tensor_product = o3.TensorProduct(
            irreps_left,
            irreps_right,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        irreps_mid = irreps_mid.simplify()
        width = _default_hidden_dim(
            condition_dim=condition_dim,
            weight_dim=self.tensor_product.weight_numel,
            output_dim=irreps_out.dim,
            hidden_dim=hidden_dim,
        )
        self.weight_net = torch.nn.Sequential(
            torch.nn.Linear(condition_dim, width),
            activation(),
            torch.nn.Linear(width, self.tensor_product.weight_numel),
        )
        self.output_linear = o3.Linear(irreps_mid, irreps_out)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            return left.new_zeros((left.shape[0], self.output_dim))
        weights = self.weight_net(condition)
        return self.output_linear(self.tensor_product(left, right, weights))


class E3nnHamiltonianNodeBlock(torch.nn.Module):
    """Scalar-conditioned equivariant-compatible node coefficient readout.

    The block maps node-wise tensors to ``irreps_out.dim`` coefficients. Tensor
    product weights are generated from local scalar ``0e`` channels, so enabling
    ``HamiltonianLocalContext`` gives the readout per-atom local control without
    using atom indices or target labels.
    """

    def __init__(
        self,
        irreps_in: Union[IrrepsLike, Iterable[IrrepsLike]],
        irreps_out: IrrepsLike,
        hidden_dim: Optional[int] = None,
        activation: Type[torch.nn.Module] = torch.nn.SiLU,
        require_scalar_context: bool = False,
        include_linear_skip: bool = True,
        allow_bottleneck: bool = False,
    ):
        super().__init__()
        self.irreps_in = _as_irreps_list(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.require_scalar_context = bool(require_scalar_context)
        self.include_linear_skip = bool(include_linear_skip)

        self.paths = torch.nn.ModuleList(
            [
                _ConditionedTensorProduct(
                    irreps_left=irreps,
                    irreps_right=irreps,
                    irreps_out=self.irreps_out,
                    condition_dim=_condition_dim(
                        irreps,
                        n_tensors=1,
                        require_scalar_context=self.require_scalar_context,
                        block_name=self.__class__.__name__,
                    ),
                    hidden_dim=hidden_dim,
                    activation=activation,
                    allow_bottleneck=allow_bottleneck,
                )
                for irreps in self.irreps_in
            ]
        )
        self.linear_skips = torch.nn.ModuleList(
            [o3.Linear(irreps, self.irreps_out) for irreps in self.irreps_in]
            if self.include_linear_skip
            else []
        )
        has_tensor_product_path = any(path.enabled for path in self.paths)
        if not self.include_linear_skip and not has_tensor_product_path:
            raise ValueError(
                "No tensor-product path connects irreps_in to irreps_out. Enable "
                "include_linear_skip or choose compatible irreps."
            )

    def forward(self, **node_kwargs: torch.Tensor) -> torch.Tensor:
        if not node_kwargs:
            raise ValueError("E3nnHamiltonianNodeBlock requires node inputs.")
        values = list(node_kwargs.values())
        if len(values) != len(self.irreps_in):
            raise ValueError(
                "Number of node inputs must match the initialized irreps_in "
                f"list: {len(values)} != {len(self.irreps_in)}."
            )
        self._validate_leading_dims(values)

        output = values[0].new_zeros((values[0].shape[0], self.irreps_out.dim))
        for index, (path, value) in enumerate(zip(self.paths, values)):
            condition = _condition_from_tensors([value], self.irreps_in[index])
            output = output + path(value, value, condition)
            if self.include_linear_skip:
                output = output + self.linear_skips[index](value)
        return output

    @staticmethod
    def _validate_leading_dims(values: list[torch.Tensor]) -> None:
        n_samples = values[0].shape[0]
        if any(value.shape[0] != n_samples for value in values):
            raise ValueError("All node inputs must have the same leading dimension.")


class E3nnHamiltonianEdgeBlock(torch.nn.Module):
    """Scalar-conditioned equivariant-compatible edge coefficient readout.

    Inputs follow the same tuple convention as ``E3nnSimpleEdgeBlock``. The
    tensor product uses both edge directions, while local scalar channels from
    both directions condition the tensor-product weights per physical edge.
    """

    def __init__(
        self,
        irreps_in: Union[IrrepsLike, Iterable[IrrepsLike]],
        irreps_out: IrrepsLike,
        node_feats_irreps: Optional[IrrepsLike] = None,
        edge_messages_irreps: Optional[IrrepsLike] = None,
        hidden_dim: Optional[int] = None,
        activation: Type[torch.nn.Module] = torch.nn.SiLU,
        require_scalar_context: bool = False,
        include_linear_skip: bool = True,
        allow_bottleneck: bool = False,
    ):
        super().__init__()
        fallback_irreps = _as_irreps_list(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.require_scalar_context = bool(require_scalar_context)
        self.include_linear_skip = bool(include_linear_skip)
        self.keyed_irreps = {}
        if edge_messages_irreps is not None:
            self.keyed_irreps["edge_messages"] = o3.Irreps(edge_messages_irreps)
        if node_feats_irreps is not None:
            self.keyed_irreps["node_feats"] = o3.Irreps(node_feats_irreps)

        if self.keyed_irreps:
            self.path_irreps = dict(self.keyed_irreps)
            self.uses_keyword_irreps = True
        else:
            self.path_irreps = {
                f"input_{index}": irreps for index, irreps in enumerate(fallback_irreps)
            }
            self.uses_keyword_irreps = False

        self.paths = torch.nn.ModuleDict(
            {
                key: _ConditionedTensorProduct(
                    irreps_left=irreps,
                    irreps_right=irreps,
                    irreps_out=self.irreps_out,
                    condition_dim=_condition_dim(
                        irreps,
                        n_tensors=2,
                        require_scalar_context=self.require_scalar_context,
                        block_name=self.__class__.__name__,
                    ),
                    hidden_dim=hidden_dim,
                    activation=activation,
                    allow_bottleneck=allow_bottleneck,
                )
                for key, irreps in self.path_irreps.items()
            }
        )
        if self.include_linear_skip:
            self.left_linear_skips = torch.nn.ModuleDict(
                {
                    key: o3.Linear(irreps, self.irreps_out)
                    for key, irreps in self.path_irreps.items()
                }
            )
            self.right_linear_skips = torch.nn.ModuleDict(
                {
                    key: o3.Linear(irreps, self.irreps_out)
                    for key, irreps in self.path_irreps.items()
                }
            )
        else:
            self.left_linear_skips = torch.nn.ModuleDict()
            self.right_linear_skips = torch.nn.ModuleDict()
        if not self.include_linear_skip and not any(
            path.enabled for path in self.paths.values()
        ):
            raise ValueError(
                "No tensor-product path connects irreps_in to irreps_out. Enable "
                "include_linear_skip or choose compatible irreps."
            )

    def forward(self, **tuple_kwargs) -> torch.Tensor:
        if not tuple_kwargs:
            raise ValueError("E3nnHamiltonianEdgeBlock requires edge inputs.")
        keys = self._forward_keys(list(tuple_kwargs))
        tuples = self._ordered_tuples(tuple_kwargs)
        self._validate_edge_tuples(tuples)

        output = tuples[0][0].new_zeros((tuples[0][0].shape[0], self.irreps_out.dim))
        for key, (left, right) in zip(keys, tuples):
            irreps = self.path_irreps[key]
            condition = _condition_from_tensors([left, right], irreps)
            path = self.paths[key]
            output = output + path(left, right, condition)
            if self.include_linear_skip:
                output = output + self.left_linear_skips[key](left)
                output = output + self.right_linear_skips[key](right)
        return output

    def _forward_keys(self, keys: list[str]) -> list[str]:
        if self.uses_keyword_irreps:
            missing = [key for key in keys if key not in self.path_irreps]
            if missing:
                raise ValueError(
                    "Received edge inputs without initialized irreps: "
                    f"{', '.join(missing)}."
                )
            return keys
        if len(keys) > len(self.path_irreps):
            raise ValueError(
                "Number of edge inputs exceeds the initialized irreps_in list: "
                f"{len(keys)} > {len(self.path_irreps)}."
            )
        return [f"input_{index}" for index in range(len(keys))]

    @staticmethod
    def _ordered_tuples(tuple_kwargs) -> list[tuple[torch.Tensor, torch.Tensor]]:
        tuples = []
        for value in tuple_kwargs.values():
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError(
                    "E3nnHamiltonianEdgeBlock inputs must be two-tensor tuples."
                )
            tuples.append(value)
        return tuples

    def _condition(
        self,
        tuples: list[tuple[torch.Tensor, torch.Tensor]],
        irreps_list: list[o3.Irreps],
    ) -> torch.Tensor:
        scalars = []
        for (left, right), irreps in zip(tuples, irreps_list):
            scalars.extend(_extract_scalars(left, irreps))
            scalars.extend(_extract_scalars(right, irreps))
        if scalars:
            return torch.cat(scalars, dim=-1)
        return tuples[0][0].new_ones((tuples[0][0].shape[0], 1))

    @staticmethod
    def _validate_edge_tuples(tuples: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        n_samples = tuples[0][0].shape[0]
        for left, right in tuples:
            if left.shape[0] != n_samples or right.shape[0] != n_samples:
                raise ValueError(
                    "All edge input tensors must have the same leading dimension."
                )
