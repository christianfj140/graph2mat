"""Torch wrappers for Graph2Mat."""
import numpy as np
import torch
from types import ModuleType

from graph2mat import Graph2Mat
from graph2mat.core.modules.graph2mat import LAYOUT_ATTR, type_layout

__all__ = ["TorchGraph2Mat", "torch_type_layout", "layout_to"]


def _map_layout(layout: dict, fn) -> dict:
    """Applies ``fn`` to every array of a ``type_layout`` (other entries are kept)."""

    def apply(value):
        if isinstance(value, dict):
            return {k: apply(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(apply(v) for v in value)
        if isinstance(value, list):
            return [apply(v) for v in value]
        if isinstance(value, (np.ndarray, torch.Tensor)):
            return fn(value)
        return value

    return apply(layout)


def torch_type_layout(point_types, edge_types, basis_table) -> dict:
    """``type_layout`` of a batch as CPU torch tensors (roadmap 6.3)."""
    return _map_layout(
        type_layout(point_types, edge_types, basis_table),
        lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64)),
    )


def layout_to(layout: dict, device) -> dict:
    """Moves a ``torch_type_layout`` to ``device``."""
    return _map_layout(layout, lambda a: a.to(device, non_blocking=True))


class TorchGraph2Mat(Graph2Mat, torch.nn.Module):
    """Wrapper for Graph2Mat to make it use torch instead of numpy.

    It also makes `Graph2Mat` a `torch.nn.Module`,  and it makes it
    store the list of node block functions as a `torch.nn.ModuleList`
    and the dictionary of edge block functions as a `torch.nn.ModuleDict`.

    Parameters
    ----------
    **kwargs:
        Additional arguments passed to the `Graph2Mat` class.

    See Also
    --------
    Graph2Mat
        The class that `TorchGraph2Mat` extends. Its documentation contains a more
        detailed explanation of the inner workings of the class.
    """

    numpy = torch

    def __init__(
        self,
        *args,
        numpy: ModuleType = torch,
        self_interactions_list=torch.nn.ModuleList,
        interactions_dict=torch.nn.ModuleDict,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            numpy=numpy,
            self_interactions_list=self_interactions_list,
            interactions_dict=interactions_dict,
        )

    def _empty_labels(self, like: torch.Tensor) -> torch.Tensor:
        shape = (0,) if self.n_matrix_components == 1 else (0, self.n_matrix_components)
        return torch.zeros(shape, device=like.device)

    def _pair_swap_index(self, n_edges: int, like: torch.Tensor) -> torch.Tensor:
        return torch.arange(n_edges, device=like.device) ^ 1

    def _init_center_types(self, basis_grouping):
        super()._init_center_types(basis_grouping)

        for k in ("types_to_graph2mat", "edge_types_to_graph2mat"):
            array = getattr(self, k, None)
            if isinstance(array, np.ndarray):
                # Register the buffer as a torch tensor
                tensor = torch.from_numpy(getattr(self, k))
                delattr(self, k)
                self.register_buffer(k, tensor, persistent=False)

    def _get_labels_resort_index(
        self, types: torch.Tensor, original_types: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Wrapping of the method to use torch instead of numpy."""
        types = types.numpy(force=True)
        original_types = original_types.numpy(force=True)

        indices = super()._get_labels_resort_index(
            types, original_types=original_types, **kwargs
        )

        if self.basis_grouping != "max":
            indices = self.numpy.from_numpy(indices)

        return indices
    
    def _forward_self_interactions(self, *args, **kwargs):
        with torch.profiler.record_function("G2M::self_interactions_forward"):
            return super()._forward_self_interactions(*args, **kwargs)

    def _forward_interactions(self, *args, **kwargs):
        with torch.profiler.record_function("G2M::interactions_forward"):
            return super()._forward_interactions(*args, **kwargs)
