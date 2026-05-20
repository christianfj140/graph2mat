"""Local scalar context features for Hamiltonian matrix readouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from e3nn import o3

__all__ = ["HamiltonianLocalContext", "HamiltonianLocalContextOutput"]


@dataclass(frozen=True)
class HamiltonianLocalContextOutput:
    """Context tensors produced for a graph.

    Attributes
    ----------
    node
        Tensor with shape ``[n_nodes, node_context_dim]``.
    edge
        Tensor with shape ``[n_edges, edge_context_dim]``.
    """

    node: torch.Tensor
    edge: torch.Tensor


class HamiltonianLocalContext(torch.nn.Module):
    """Build deterministic scalar local context for Hamiltonian readouts.

    The module uses only species, edge distances, and rotation-invariant
    directional moments around each local pair. It intentionally returns scalar
    ``0e`` channels so they can be concatenated to MACE/Graph2Mat readout
    features without introducing a local-frame convention.
    """

    def __init__(
        self,
        num_types: int,
        r_max: float,
        num_radial: int = 4,
        radial_width: Optional[float] = None,
        include_species: bool = True,
        include_directional_moments: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        if num_types <= 0:
            raise ValueError("num_types must be positive.")
        if r_max <= 0:
            raise ValueError("r_max must be positive.")
        if num_radial <= 0:
            raise ValueError("num_radial must be positive.")
        if radial_width is not None and radial_width <= 0:
            raise ValueError("radial_width must be positive when provided.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.num_types = int(num_types)
        self.r_max = float(r_max)
        self.num_radial = int(num_radial)
        self.radial_width = (
            float(radial_width)
            if radial_width is not None
            else float(max(num_radial, 1) ** 2)
        )
        self.include_species = bool(include_species)
        self.include_directional_moments = bool(include_directional_moments)
        self.eps = float(eps)

    @property
    def node_context_dim(self) -> int:
        dim = self.num_radial + self.num_types * self.num_radial
        if self.include_species:
            dim += self.num_types
        return dim

    @property
    def edge_context_dim(self) -> int:
        dim = self.num_radial + 2 * self.node_context_dim
        if self.include_species:
            dim += 2 * self.num_types
        if self.include_directional_moments:
            dim += 2 * self.num_radial
        return dim

    @property
    def node_context_irreps(self) -> o3.Irreps:
        return o3.Irreps(f"{self.node_context_dim}x0e")

    @property
    def edge_context_irreps(self) -> o3.Irreps:
        return o3.Irreps(f"{self.edge_context_dim}x0e")

    def forward(
        self,
        data,
        edge_vectors: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> HamiltonianLocalContextOutput:
        """Compute local context.

        Parameters
        ----------
        data
            Graph data with ``point_types`` and ``edge_index``.
        edge_vectors
            Directed edge vectors with shape ``[n_edges, 3]``.
        edge_lengths
            Directed edge lengths with shape ``[n_edges]`` or ``[n_edges, 1]``.
        """

        point_types = data["point_types"].long().to(edge_vectors.device)
        edge_index = data["edge_index"].long().to(edge_vectors.device)

        if point_types.ndim != 1:
            raise ValueError("point_types must be a one-dimensional tensor.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges].")
        if edge_vectors.ndim != 2 or edge_vectors.shape[1] != 3:
            raise ValueError("edge_vectors must have shape [n_edges, 3].")

        n_nodes = int(point_types.shape[0])
        n_edges = int(edge_index.shape[1])
        if edge_vectors.shape[0] != n_edges:
            raise ValueError(
                "edge_vectors and edge_index disagree on the number of edges: "
                f"{edge_vectors.shape[0]} != {n_edges}."
            )

        edge_lengths = edge_lengths.reshape(-1)
        if edge_lengths.shape[0] != n_edges:
            raise ValueError(
                "edge_lengths and edge_index disagree on the number of edges: "
                f"{edge_lengths.shape[0]} != {n_edges}."
            )
        if n_nodes and (
            int(point_types.min().item()) < 0
            or int(point_types.max().item()) >= self.num_types
        ):
            raise ValueError(
                "point_types contains values outside the configured "
                f"[0, {self.num_types}) range."
            )

        dtype = edge_vectors.dtype
        device = edge_vectors.device
        edge_lengths = edge_lengths.to(dtype=dtype, device=device)
        species = torch.nn.functional.one_hot(
            point_types, num_classes=self.num_types
        ).to(dtype=dtype, device=device)

        sender, receiver = edge_index
        radial = self._radial_features(edge_lengths)
        unit_vectors = edge_vectors / edge_lengths.clamp_min(self.eps).reshape(-1, 1)

        node_radial = self._scatter_sum(radial, sender, n_nodes)
        neighbor_species = species[receiver]
        species_radial = (radial[:, :, None] * neighbor_species[:, None, :]).reshape(
            n_edges, self.num_radial * self.num_types
        )
        node_species_radial = self._scatter_sum(species_radial, sender, n_nodes)

        node_parts = []
        if self.include_species:
            node_parts.append(species)
        node_parts.extend([node_radial, node_species_radial])
        node_context = torch.cat(node_parts, dim=-1)

        edge_parts = []
        if self.include_species:
            edge_parts.extend([species[sender], species[receiver]])
        edge_parts.extend([radial, node_context[sender], node_context[receiver]])

        if self.include_directional_moments:
            radial_vectors = radial[:, :, None] * unit_vectors[:, None, :]
            vector_moments = self._scatter_sum(radial_vectors, sender, n_nodes)
            sender_moments = (
                vector_moments[sender] * unit_vectors[:, None, :]
            ).sum(dim=-1)
            receiver_moments = (
                vector_moments[receiver] * (-unit_vectors[:, None, :])
            ).sum(dim=-1)
            edge_parts.extend([sender_moments, receiver_moments])

        edge_context = torch.cat(edge_parts, dim=-1)

        return HamiltonianLocalContextOutput(node=node_context, edge=edge_context)

    def _radial_features(self, edge_lengths: torch.Tensor) -> torch.Tensor:
        scaled = edge_lengths.reshape(-1, 1) / self.r_max
        if self.num_radial == 1:
            centers = torch.full(
                (1,),
                0.5,
                dtype=edge_lengths.dtype,
                device=edge_lengths.device,
            )
        else:
            centers = torch.linspace(
                0.0,
                1.0,
                self.num_radial,
                dtype=edge_lengths.dtype,
                device=edge_lengths.device,
            )
        radial = torch.exp(-self.radial_width * (scaled - centers.reshape(1, -1)) ** 2)
        cutoff = (1.0 - scaled).clamp_min(0.0) ** 2
        return radial * cutoff

    @staticmethod
    def _scatter_sum(
        values: torch.Tensor,
        index: torch.Tensor,
        dim_size: int,
    ) -> torch.Tensor:
        output = values.new_zeros((dim_size, *values.shape[1:]))
        expand_shape = (index.shape[0], *([1] * (values.ndim - 1)))
        expanded_index = index.reshape(expand_shape).expand_as(values)
        return output.scatter_add(0, expanded_index, values)
