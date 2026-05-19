"""Experimental dense coefficient readouts for diagnostics.

These modules are intentionally non-equivariant. They are useful as opt-in
probes to test whether a less constrained coefficient head can fit a target
that the equivariant readout cannot memorize.
"""

from __future__ import annotations

from typing import Iterable, Optional, Type, Union

import torch
from e3nn import o3

__all__ = [
    "E3nnDiagnosticDenseNodeBlock",
    "E3nnDiagnosticDenseEdgeBlock",
]


IrrepsLike = Union[o3.Irreps, str]


def _as_irreps_list(irreps_in: Union[IrrepsLike, Iterable[IrrepsLike]]):
    if isinstance(irreps_in, (o3.Irreps, str)):
        return [o3.Irreps(irreps_in)]
    return [o3.Irreps(irreps) for irreps in irreps_in]


def _hidden_dim(input_dim: int, output_dim: int, hidden_dim: Optional[int]) -> int:
    if hidden_dim is not None:
        return int(hidden_dim)
    return max(int(input_dim), int(output_dim))


class E3nnDiagnosticDenseNodeBlock(torch.nn.Module):
    """Non-equivariant dense node coefficient readout for diagnostics.

    The module concatenates node-wise readout tensors and maps them through a
    small MLP to ``irreps_out.dim`` coefficients. It is deliberately opt-in and
    should be used to diagnose capacity/rank issues, not as a default equivariant
    architecture.
    """

    def __init__(
        self,
        irreps_in: Union[IrrepsLike, Iterable[IrrepsLike]],
        irreps_out: IrrepsLike,
        hidden_dim: Optional[int] = None,
        activation: Type[torch.nn.Module] = torch.nn.SiLU,
    ):
        super().__init__()
        self.irreps_in = _as_irreps_list(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        input_dim = sum(irreps.dim for irreps in self.irreps_in)
        output_dim = self.irreps_out.dim
        width = _hidden_dim(input_dim, output_dim, hidden_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, width),
            activation(),
            torch.nn.Linear(width, output_dim),
        )

    def forward(self, **node_kwargs: torch.Tensor) -> torch.Tensor:
        if not node_kwargs:
            raise ValueError("E3nnDiagnosticDenseNodeBlock requires node inputs.")
        values = list(node_kwargs.values())
        n_samples = values[0].shape[0]
        if any(value.shape[0] != n_samples for value in values):
            raise ValueError("All node inputs must have the same leading dimension.")
        return self.mlp(torch.cat(values, dim=-1))


class E3nnDiagnosticDenseEdgeBlock(torch.nn.Module):
    """Non-equivariant dense edge coefficient readout for diagnostics.

    The module concatenates both directions of each edge input tuple and maps
    them through an MLP to ``irreps_out.dim`` coefficients. It follows the same
    call convention as ``E3nnSimpleEdgeBlock`` and is intended only as an
    experimental full-rank probe.
    """

    def __init__(
        self,
        irreps_in: Union[IrrepsLike, Iterable[IrrepsLike]],
        irreps_out: IrrepsLike,
        hidden_dim: Optional[int] = None,
        activation: Type[torch.nn.Module] = torch.nn.SiLU,
    ):
        super().__init__()
        self.irreps_in = _as_irreps_list(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        input_dim = 2 * sum(irreps.dim for irreps in self.irreps_in)
        output_dim = self.irreps_out.dim
        width = _hidden_dim(input_dim, output_dim, hidden_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, width),
            activation(),
            torch.nn.Linear(width, output_dim),
        )

    def forward(
        self, **tuple_kwargs
    ) -> torch.Tensor:
        if not tuple_kwargs:
            raise ValueError("E3nnDiagnosticDenseEdgeBlock requires edge inputs.")
        pieces = []
        n_samples = None
        for value in tuple_kwargs.values():
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError(
                    "E3nnDiagnosticDenseEdgeBlock inputs must be two-tensor tuples."
                )
            left, right = value
            if n_samples is None:
                n_samples = left.shape[0]
            if left.shape[0] != n_samples or right.shape[0] != n_samples:
                raise ValueError(
                    "All edge input tensors must have the same leading dimension."
                )
            pieces.extend([left, right])
        return self.mlp(torch.cat(pieces, dim=-1))
