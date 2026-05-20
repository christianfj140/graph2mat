"""Functions to assess performance.

When answering "How similar are these two matrices?", there is no
perfectly right answer. Depending on what you are most interested in,
you might use any of the functions implemented here.

Functions are wrapped into an ``OrbitalMatrixMetric`` class to make
sure they share the same interface and that they are all registered.
"""

from copy import copy

from typing import Any, Tuple, Dict, Type, Callable, Union, Optional
import numpy as np
import torch

from .formats import Formats
from .processing import MatrixDataProcessor

__all__ = [
    "OrbitalMatrixMetric",
    "block_type_mse",
    "block_type_huber",
    "block_normalized_huber",
    "block_type_smooth_l1",
    "hamiltonian_composite_loss",
    "project_block_to_coefficients",
    "target_coefficients_from_labels",
    "coefficients_to_labels",
    "coefficient_space_mse",
    "coefficient_space_mae",
    "block_type_mae",
    "block_type_mape",
    "block_type_mapemaemix",
    "block_type_mapemsemix",
    "block_type_mapestdmix",
    "elementwise_mse",
    "node_mse",
    "edge_mse",
    "block_type_mse_threshold",
    "block_type_mse_sigmoid_thresh",
    "block_type_mae_sigmoid_thresh",
    "normalized_density_error",
]


def _isnan(values):
    """NaN checking compatible with both torch and numpy"""
    return values != values


def get_predictions_error(
    nodes_pred, nodes_ref, edges_pred, edges_ref, remove_nan=True
):
    """Returns errors for both nodes and edges, removing NaN values."""
    node_error = nodes_pred - nodes_ref

    if remove_nan:
        if getattr(edges_ref, "ndim", 1) == 2:
            notnan = ~_isnan(edges_ref).any(axis=1)
            edge_error = edges_ref[notnan] - edges_pred[notnan]
        else:
            notnan = ~_isnan(edges_ref)
            edge_error = edges_ref[notnan] - edges_pred[notnan]
    else:
        edge_error = edges_ref - edges_pred

    return node_error, edge_error


def _smooth_l1(values, beta: float):
    if beta <= 0:
        raise ValueError(f"Smooth L1/Huber beta must be positive, got {beta!r}.")
    abs_values = abs(values)
    quadratic = 0.5 * values**2 / beta
    linear = abs_values - 0.5 * beta
    if torch.is_tensor(values):
        return torch.where(abs_values < beta, quadratic, linear)
    return np.where(abs_values < beta, quadratic, linear)


def _spin_channel_stats(node_error, edge_error):
    if getattr(node_error, "ndim", 1) != 2:
        return {}

    stats = {}
    for i in range(node_error.shape[1]):
        stats[f"node_rmse_spin{i}"] = (node_error[:, i] ** 2).mean() ** (1 / 2)
    if getattr(edge_error, "ndim", 1) == 2:
        for i in range(edge_error.shape[1]):
            stats[f"edge_rmse_spin{i}"] = (edge_error[:, i] ** 2).mean() ** (1 / 2)
    return stats


def _validate_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return value


def _as_stat_tensor(reference, value):
    if torch.is_tensor(reference):
        return reference.new_tensor(value)
    return value


def _as_numpy_int(values) -> np.ndarray:
    if torch.is_tensor(values):
        return values.detach().cpu().numpy().astype(int)
    return np.asarray(values).astype(int)


def _map_types(type_map, values) -> np.ndarray:
    values = _as_numpy_int(values)
    if torch.is_tensor(type_map):
        mapped = type_map[
            torch.as_tensor(values, dtype=torch.long, device=type_map.device)
        ]
        return mapped.detach().cpu().numpy().astype(int)

    return np.asarray(type_map)[values].astype(int)


def project_block_to_coefficients(block, change_of_basis):
    """Project a dense block into irreducible coefficients."""
    if block.ndim == 2:
        block = block.unsqueeze(0)

    if block.ndim == 4:
        coefficients = torch.einsum(
            "zij,nijc->ncz", change_of_basis.to(block), block
        )
        return coefficients.reshape(coefficients.shape[0], -1)

    return torch.einsum("zij,nij->nz", change_of_basis.to(block), block)


_project_block_to_coefficients = project_block_to_coefficients


def _coefficients_to_block(operation, coefficients):
    if not hasattr(operation, "coefficients_to_block"):
        raise ValueError(
            "coefficient reconstruction requires matrix blocks to expose "
            "coefficients_to_block()."
        )

    return operation.coefficients_to_block(coefficients)


def _flatten_reconstructed_block(block):
    if block.ndim == 3:
        return block.reshape(-1)
    if block.ndim == 4:
        return block.reshape(-1, block.shape[-1])
    raise ValueError(f"Unsupported reconstructed block dimensions: {block.ndim}.")


def _edge_module_for_unique_type(readout, edge_type: int):
    graph_edge_type = int(
        abs(_map_types(readout.edge_types_to_graph2mat, [edge_type])[0])
    )
    for module_key, operation in readout.interactions.items():
        _, _, op_edge_type = map(int, module_key[1:-1].split(","))
        if abs(op_edge_type) == graph_edge_type:
            return module_key, operation
    raise KeyError(f"No readout edge operation found for edge type {edge_type}")


def _readout_coefficient_metadata(readout) -> dict:
    if hasattr(readout, "coefficient_metadata"):
        return readout.coefficient_metadata()
    return {}


def _coefficient_values(coefficients: dict, kind: str) -> dict:
    if kind not in coefficients:
        raise ValueError(f"Missing {kind} coefficient outputs.")
    values = coefficients[kind]
    if not isinstance(values, dict):
        raise TypeError(f"{kind} coefficient outputs must be a dictionary.")
    return values


def _batch_field(batch, name: str, default=None):
    if batch is None:
        return default
    if isinstance(batch, dict):
        return batch.get(name, default)
    return getattr(batch, name, default)


def _flat_length(values) -> int:
    if torch.is_tensor(values):
        return int(values.reshape(-1).shape[0])
    return int(np.asarray(values).reshape(-1).shape[0])


def _edge_label_count_from_coefficients(readout, coefficients: dict) -> int:
    edge_coeffs = _coefficient_values(coefficients, "edge")
    total = 0
    for op_key, coeff in edge_coeffs.items():
        module_key = op_key[len("edge:") :] if op_key.startswith("edge:") else op_key
        if module_key not in readout.interactions:
            raise ValueError(
                f"Cannot infer edge label length from coefficients for {op_key}: "
                "the readout has no matching edge operation."
            )
        operation = readout.interactions[module_key]
        if not hasattr(operation, "change_of_basis"):
            raise ValueError(
                f"Cannot infer edge label length from coefficients for {op_key}: "
                "the edge operation has no change_of_basis tensor."
            )
        total += int(coeff.shape[0]) * int(np.prod(operation.change_of_basis.shape[1:]))
    return total


def _symmetric_edge_types_for_labels(
    *,
    readout,
    batch,
    basis_table,
    expected_nlabels: Optional[int],
) -> np.ndarray:
    edge_types = _as_numpy_int(_batch_field(batch, "edge_types"))
    if not getattr(readout, "symmetric", False):
        return edge_types

    if expected_nlabels is None:
        edge_labels = _batch_field(batch, "edge_labels")
        if edge_labels is not None:
            expected_nlabels = _flat_length(edge_labels)

    if expected_nlabels is None:
        raise ValueError(
            "Cannot select symmetric edge representatives without the expected "
            "flat edge-label length. Provide edge labels or coefficient outputs "
            "with matching edge operations."
        )

    point_types = _batch_field(batch, "point_types")
    point_types = None if point_types is None else _as_numpy_int(point_types)
    edge_index = _batch_field(batch, "edge_index")
    edge_index = None if edge_index is None else _as_numpy_int(edge_index)

    selector = MatrixDataProcessor(
        basis_table=basis_table,
        symmetric_matrix=True,
        sub_point_matrix=False,
    )
    unique_edge_mask = selector._get_symmetric_unique_edge_mask(
        edge_types,
        expected_nlabels=int(expected_nlabels),
        edge_index=edge_index,
        point_types=point_types,
    )
    return edge_types[unique_edge_mask]


def target_coefficients_from_labels(
    readout,
    batch,
    basis_table,
    nodes_ref,
    edges_ref,
) -> dict:
    """Project flat target labels into the readout coefficient space.

    Returns a dictionary with ``node`` and ``edge`` coefficient tensors keyed by
    readout operation, plus ``metadata`` describing those operations.
    """
    if nodes_ref.ndim != 1 or edges_ref.ndim != 1:
        raise ValueError(
            "coefficient-space metrics currently support only flat single-component "
            "labels. Use n_matrix_components=1 with an H-only target."
        )

    point_types = _as_numpy_int(batch.point_types)
    point_labels = nodes_ref.reshape(-1)
    point_pointers = basis_table.point_block_pointer(point_types)
    graph_node_types = _map_types(readout.types_to_graph2mat, point_types)

    node_coeffs = {}
    for atom_index, point_type in enumerate(point_types):
        op_key = f"node:{graph_node_types[atom_index]}"
        operation = readout.self_interactions[int(graph_node_types[atom_index])]
        shape = tuple(int(x) for x in basis_table.point_block_shape[:, point_type])
        block = point_labels[
            point_pointers[atom_index] : point_pointers[atom_index + 1]
        ].reshape(shape)
        coeff = project_block_to_coefficients(
            block, operation.change_of_basis
        ).squeeze(0)
        node_coeffs.setdefault(op_key, []).append(coeff)

    edge_labels = edges_ref.reshape(-1)
    edge_types = _symmetric_edge_types_for_labels(
        readout=readout,
        batch=batch,
        basis_table=basis_table,
        expected_nlabels=_flat_length(edge_labels),
    )
    edge_pointers = basis_table.edge_block_pointer(np.abs(edge_types))
    edge_coeffs = {}
    for unique_edge_index, edge_type in enumerate(edge_types):
        module_key, operation = _edge_module_for_unique_type(readout, int(edge_type))
        op_key = f"edge:{module_key}"
        shape = tuple(int(x) for x in basis_table.edge_block_shape[:, abs(edge_type)])
        block = edge_labels[
            edge_pointers[unique_edge_index] : edge_pointers[unique_edge_index + 1]
        ].reshape(shape)
        coeff = project_block_to_coefficients(
            block, operation.change_of_basis
        ).squeeze(0)
        edge_coeffs.setdefault(op_key, []).append(coeff)

    return {
        "node": {key: torch.stack(value) for key, value in node_coeffs.items()},
        "edge": {key: torch.stack(value) for key, value in edge_coeffs.items()},
        "metadata": _readout_coefficient_metadata(readout),
    }


_target_coefficients_from_labels = target_coefficients_from_labels


def coefficients_to_labels(readout, batch, basis_table, coefficients: dict) -> dict:
    """Map operation-wise coefficients back to flat node and edge labels."""

    point_types = _as_numpy_int(batch.point_types)
    graph_node_types = _map_types(readout.types_to_graph2mat, point_types)
    node_coeffs = _coefficient_values(coefficients, "node")
    node_offsets = {key: 0 for key in node_coeffs}
    node_labels = []

    for graph_node_type in graph_node_types:
        op_key = f"node:{graph_node_type}"
        if op_key not in node_coeffs:
            raise ValueError(f"Missing predicted node coefficients for {op_key}.")

        operation = readout.self_interactions[int(graph_node_type)]
        offset = node_offsets[op_key]
        coeff = node_coeffs[op_key][offset : offset + 1]
        node_offsets[op_key] += 1
        node_labels.append(
            _flatten_reconstructed_block(_coefficients_to_block(operation, coeff))
        )

    edge_coeffs = _coefficient_values(coefficients, "edge")
    edge_types = _symmetric_edge_types_for_labels(
        readout=readout,
        batch=batch,
        basis_table=basis_table,
        expected_nlabels=_edge_label_count_from_coefficients(readout, coefficients),
    )
    edge_offsets = {key: 0 for key in edge_coeffs}
    edge_labels = []

    for edge_type in edge_types:
        module_key, operation = _edge_module_for_unique_type(readout, int(edge_type))
        op_key = f"edge:{module_key}"
        if op_key not in edge_coeffs:
            raise ValueError(f"Missing predicted edge coefficients for {op_key}.")

        offset = edge_offsets[op_key]
        coeff = edge_coeffs[op_key][offset : offset + 1]
        edge_offsets[op_key] += 1
        edge_labels.append(
            _flatten_reconstructed_block(_coefficients_to_block(operation, coeff))
        )

    for kind, offsets, values in (
        ("node", node_offsets, node_coeffs),
        ("edge", edge_offsets, edge_coeffs),
    ):
        for key, offset in offsets.items():
            if offset != len(values[key]):
                raise ValueError(
                    f"Unused {kind} coefficients for {key}: consumed={offset}, "
                    f"total={len(values[key])}."
                )

    node_labels = torch.cat(node_labels, dim=0) if node_labels else None
    if edge_labels:
        edge_labels = torch.cat(edge_labels, dim=0)
    else:
        device = node_labels.device if torch.is_tensor(node_labels) else None
        edge_labels = torch.empty(0, dtype=node_labels.dtype, device=device)

    return {"node_labels": node_labels, "edge_labels": edge_labels}


def _coefficient_space_metric(
    nodes_ref,
    edges_ref,
    batch,
    basis_table,
    out,
    model,
    *,
    reduction,
):
    if out is None or "node_coefficients" not in out or "edge_coefficients" not in out:
        raise ValueError(
            "coefficient-space loss requires model outputs to include "
            "'node_coefficients' and 'edge_coefficients'. Instantiate the model with "
            "return_coefficients=True."
        )

    readout = getattr(model, "matrix_readouts", None)
    if readout is None:
        raise ValueError(
            "coefficient-space loss requires a model with a "
            "'matrix_readouts' attribute."
        )

    target = target_coefficients_from_labels(
        readout=readout,
        batch=batch,
        basis_table=basis_table,
        nodes_ref=nodes_ref,
        edges_ref=edges_ref,
    )
    predicted = {"node": out["node_coefficients"], "edge": out["edge_coefficients"]}

    losses = []
    stats = {}
    for kind in ("node", "edge"):
        predicted_values = _coefficient_values(predicted, kind)
        for key, target_value in _coefficient_values(target, kind).items():
            if key not in predicted_values:
                raise ValueError(f"Missing predicted {kind} coefficients for {key}.")
            error = predicted_values[key] - target_value.to(predicted_values[key])
            if reduction == "mse":
                loss = (error**2).mean()
                stats[f"{kind}_{key}_coeff_rmse"] = loss ** (1 / 2)
            elif reduction == "mae":
                loss = abs(error).mean()
                stats[f"{kind}_{key}_coeff_mae"] = loss
            else:
                raise ValueError(f"Unknown coefficient reduction {reduction!r}")
            losses.append(loss)

    if not losses:
        raise ValueError(
            "No coefficient blocks were available for coefficient-space loss."
        )

    loss = torch.stack(losses).mean()
    stats["coefficient_blocks"] = torch.as_tensor(
        len(losses), dtype=loss.dtype, device=loss.device
    )
    return loss, stats


def _loss_mean(values, pointers=None):
    if pointers is None:
        return values.mean()
    losses = []
    for start, end in zip(pointers[:-1], pointers[1:]):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        losses.append(values[start:end].mean())
    if not losses:
        return values.mean()
    if torch.is_tensor(values):
        return torch.stack(losses).mean()
    return np.stack(losses).mean()


def _loss_blocks(values, pointers):
    losses = []
    total = int(values.shape[0])
    if int(pointers[-1]) != total:
        raise ValueError(
            "Block-normalized loss pointer mismatch: "
            f"last pointer is {int(pointers[-1])}, but labels have length {total}."
        )
    for start, end in zip(pointers[:-1], pointers[1:]):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        losses.append(values[start:end].mean())
    if not losses:
        raise ValueError("Block-normalized loss received no non-empty blocks.")
    return losses


def _node_edge_block_pointers(
    batch,
    basis_table,
    model=None,
    expected_edge_nlabels: Optional[int] = None,
):
    if batch is None or basis_table is None:
        raise ValueError(
            "per_block_normalization requires both batch and basis_table."
        )

    point_types = _as_numpy_int(batch.point_types)
    node_pointers = basis_table.point_block_pointer(point_types)

    readout = getattr(model, "matrix_readouts", None)
    edge_types = _symmetric_edge_types_for_labels(
        readout=readout,
        batch=batch,
        basis_table=basis_table,
        expected_nlabels=expected_edge_nlabels,
    )
    edge_pointers = basis_table.edge_block_pointer(np.abs(edge_types))
    return node_pointers, edge_pointers


def _single_component_labels_or_raise(nodes_ref, edges_ref, loss_name: str):
    if getattr(nodes_ref, "ndim", 1) != 1 or getattr(edges_ref, "ndim", 1) != 1:
        raise ValueError(
            f"{loss_name} supports only flat single-component H-only labels. "
            "Use matrix_component_policy='h_only' and n_matrix_components=1."
        )


def _block_type_stats(values, pointers, types, prefix: str):
    if types is None or len(types) != len(pointers) - 1:
        return {}

    block_losses = _loss_blocks(values, pointers)
    stats = {}
    for type_id in sorted({int(type_value) for type_value in types}):
        selected = [
            loss
            for loss, type_value in zip(block_losses, types)
            if int(type_value) == type_id
        ]
        if not selected:
            continue
        if torch.is_tensor(values):
            stats[f"{prefix}_type{type_id}_block_huber"] = torch.stack(selected).mean()
        else:
            stats[f"{prefix}_type{type_id}_block_huber"] = np.stack(selected).mean()
    return stats


def _weighted_label_huber(
    node_error,
    edge_error,
    *,
    beta: float,
    node_weight: float,
    edge_weight: float,
    per_block_normalization: bool,
    batch=None,
    basis_table=None,
    model=None,
):
    node_losses = _smooth_l1(node_error, beta)
    edge_losses = _smooth_l1(edge_error, beta)
    node_pointers = edge_pointers = None
    if per_block_normalization:
        node_pointers, edge_pointers = _node_edge_block_pointers(
            batch=batch,
            basis_table=basis_table,
            model=model,
            expected_edge_nlabels=_flat_length(edge_error),
        )

    node_loss = _loss_mean(node_losses, node_pointers)
    edge_loss = _loss_mean(edge_losses, edge_pointers)
    return node_weight * node_loss + edge_weight * edge_loss, node_loss, edge_loss


class Meta(type):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        metric_keys = {"nodes_pred", "nodes_ref", "edges_pred", "edges_ref"}
        if len(args) == 0 and not (set(kwds) & metric_keys):
            return super().__call__(*args, **kwds)

        return super().__call__().get_metric(*args, **kwds)


class OrbitalMatrixMetric(metaclass=Meta):
    requires_model_output = False

    def __call__(self, *args, **kwargs):
        return self.get_metric(*args, **kwargs)

    def get_metric(
        self, config_resolved=False, **kwargs
    ) -> Union[Tuple[float, Dict[str, float]], np.ndarray]:
        """Get the value for the metric.

        Parameters
        ----------
        config_resolved : bool, optional
            Whether the metric should be computed individually for each configuration in the batch, by default False.

            If False, a single float is returned, which is the metric for the whole batch.
            If True, a numpy array is returned, which contains the metric for each configuration in the batch.

        Returns
        -------
        Union[Tuple[float, Dict[str, float]], np.ndarray]
            The metric value(s), as specified by config_resolved.

            If config_resolved is False, a dictionary is also returned containing additional stats related to the metric.
        """

        if not config_resolved:
            return self.compute_metric(**kwargs)
        else:
            # We need to loop through the batch
            if "batch" not in kwargs:
                raise ValueError(
                    "A batch is required to compute metrics individually for each configuration."
                )

            batch = kwargs["batch"]

            if "symmetric_matrix" not in kwargs:
                raise ValueError(
                    "symmetric_matrix is required to compute metrics individually for each configuration."
                )
            if "basis_table" not in kwargs:
                raise ValueError(
                    "basis_table is required to compute metrics individually for each configuration."
                )

            processor = MatrixDataProcessor(
                basis_table=kwargs["basis_table"],
                symmetric_matrix=kwargs["symmetric_matrix"],
            )

            target_iterator = processor.yield_from_batch(batch)
            pred_iterator = processor.yield_from_batch(
                batch,
                predictions={
                    "node_labels": kwargs["nodes_pred"],
                    "edge_labels": kwargs["edges_pred"],
                },
            )

            metrics_array = None

            for i, (pred, target) in enumerate(zip(pred_iterator, target_iterator)):
                kwargs["nodes_pred"] = pred.point_labels
                kwargs["edges_pred"] = pred.edge_labels
                kwargs["nodes_ref"] = target.point_labels
                kwargs["edges_ref"] = target.edge_labels

                kwargs["batch"] = target

                config_metric, _ = self.compute_metric(**kwargs)

                if metrics_array is None:
                    metrics_array = np.zeros(batch.num_graphs, dtype=np.float64)

                metrics_array[i] = config_metric

            return metrics_array, {}

    @staticmethod
    def compute_metric(
        nodes_pred, nodes_ref, edges_pred, edges_ref, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """Function that actually computes the metric.
        This function should return the metric and a dictionary of other intermediate metrics that have
        been computed as intermediate steps (or are easy to compute from intermediate steps), if applicable.
        This other metrics might be, for example, logged during training.
        """
        raise NotImplementedError

    @classmethod
    def from_metric_func(cls, fn: Callable) -> Type["OrbitalMatrixMetric"]:
        """Creates an OrbitalMatrixMetric class from a function that computes the loss."""
        return type(fn.__name__, (cls,), {"compute_metric": staticmethod(fn)})


@OrbitalMatrixMetric.from_metric_func
def block_type_mse(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = (node_error**2).mean()
    edge_loss = (edge_error**2).mean()

    stats = {
        "node_rmse": node_loss ** (1 / 2),
        "edge_rmse": edge_loss ** (1 / 2),
    }
    stats.update(_spin_channel_stats(node_error, edge_error))

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def coefficient_space_mse(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    return _coefficient_space_metric(
        nodes_ref=nodes_ref,
        edges_ref=edges_ref,
        reduction="mse",
        **kwargs,
    )


coefficient_space_mse.requires_model_output = True


@OrbitalMatrixMetric.from_metric_func
def coefficient_space_mae(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    return _coefficient_space_metric(
        nodes_ref=nodes_ref,
        edges_ref=edges_ref,
        reduction="mae",
        **kwargs,
    )


coefficient_space_mae.requires_model_output = True


@OrbitalMatrixMetric.from_metric_func
def block_type_mae(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = abs(node_error).mean()
    edge_loss = abs(edge_error).mean()

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


class block_type_huber(OrbitalMatrixMetric):
    """Block-type Smooth L1/Huber loss for matrix labels.

    This is an opt-in Hamiltonian training candidate between the current
    block-type MAE and MSE objectives. ``beta`` is the Smooth L1 transition
    width in label units; ``delta`` is accepted as an alias for Huber-style
    configuration files.
    """

    def __init__(self, beta: float = 1.0, delta: Optional[float] = None):
        if delta is not None:
            beta = delta
        beta = float(beta)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError(f"block_type_huber beta must be positive, got {beta!r}.")
        self.beta = beta

    def compute_metric(
        self, nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        node_error, edge_error = get_predictions_error(
            nodes_pred, nodes_ref, edges_pred, edges_ref
        )

        node_loss = _smooth_l1(node_error, self.beta).mean()
        edge_loss = _smooth_l1(edge_error, self.beta).mean()

        stats = {
            "node_smooth_l1": node_loss,
            "edge_smooth_l1": edge_loss,
            "node_rmse": (node_error**2).mean() ** (1 / 2),
            "edge_rmse": (edge_error**2).mean() ** (1 / 2),
        }
        stats.update(_spin_channel_stats(node_error, edge_error))

        if log_verbose:
            abs_node_error = abs(node_error)
            abs_edge_error = abs(edge_error)
            stats.update(
                {
                    "node_mean": abs_node_error.mean(),
                    "edge_mean": abs_edge_error.mean(),
                    "node_std": abs_node_error.std(),
                    "edge_std": abs_edge_error.std(),
                    "node_max": abs_node_error.max(),
                    "edge_max": abs_edge_error.max(),
                    "smooth_l1_beta": node_error.new_tensor(self.beta)
                    if torch.is_tensor(node_error)
                    else self.beta,
                }
            )

        return node_loss + edge_loss, stats


class block_type_smooth_l1(block_type_huber):
    """Alias for ``block_type_huber`` using PyTorch Smooth L1 terminology."""


class block_normalized_huber(OrbitalMatrixMetric):
    """Opt-in H-only Huber loss averaged by physical matrix blocks.

    Unlike ``block_type_huber``, this objective first averages the Huber errors
    inside each physical node/edge block and then averages those block losses.
    This prevents large blocks from dominating only because they contain more
    matrix elements. It currently supports only flat single-component H-only
    labels.
    """

    def __init__(
        self,
        beta: float = 0.01,
        delta: Optional[float] = None,
        node_weight: float = 1.0,
        edge_weight: float = 1.0,
        log_type_stats: bool = True,
    ):
        if delta is not None:
            beta = delta
        beta = float(beta)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError(
                f"block_normalized_huber beta must be positive, got {beta!r}."
            )
        self.beta = beta
        self.node_weight = _validate_nonnegative(node_weight, "node_weight")
        self.edge_weight = _validate_nonnegative(edge_weight, "edge_weight")
        if self.node_weight == 0 and self.edge_weight == 0:
            raise ValueError("At least one of node_weight or edge_weight must be > 0.")
        self.log_type_stats = bool(log_type_stats)

    def compute_metric(
        self,
        nodes_pred,
        nodes_ref,
        edges_pred,
        edges_ref,
        log_verbose=False,
        **kwargs,
    ) -> Tuple[float, Dict[str, float]]:
        _single_component_labels_or_raise(
            nodes_ref,
            edges_ref,
            "block_normalized_huber",
        )
        node_error, edge_error = get_predictions_error(
            nodes_pred, nodes_ref, edges_pred, edges_ref
        )
        node_pointers, edge_pointers = _node_edge_block_pointers(
            batch=kwargs.get("batch"),
            basis_table=kwargs.get("basis_table"),
            model=kwargs.get("model"),
            expected_edge_nlabels=_flat_length(edge_error),
        )

        node_losses = _smooth_l1(node_error, self.beta)
        edge_losses = _smooth_l1(edge_error, self.beta)
        node_block_losses = _loss_blocks(node_losses, node_pointers)
        edge_block_losses = _loss_blocks(edge_losses, edge_pointers)

        if torch.is_tensor(node_losses):
            node_loss = torch.stack(node_block_losses).mean()
        else:
            node_loss = np.stack(node_block_losses).mean()
        if torch.is_tensor(edge_losses):
            edge_loss = torch.stack(edge_block_losses).mean()
        else:
            edge_loss = np.stack(edge_block_losses).mean()
        loss = self.node_weight * node_loss + self.edge_weight * edge_loss

        stats = {
            "node_block_huber": node_loss,
            "edge_block_huber": edge_loss,
            "node_rmse": (node_error**2).mean() ** (1 / 2),
            "edge_rmse": (edge_error**2).mean() ** (1 / 2),
            "block_huber_beta": _as_stat_tensor(node_error, self.beta),
            "node_blocks": _as_stat_tensor(node_error, len(node_block_losses)),
            "edge_blocks": _as_stat_tensor(edge_error, len(edge_block_losses)),
        }
        stats.update(_spin_channel_stats(node_error, edge_error))

        if self.log_type_stats:
            batch = kwargs.get("batch")
            model = kwargs.get("model")
            point_types = _as_numpy_int(batch.point_types) if batch is not None else None
            edge_types = None
            if batch is not None:
                readout = getattr(model, "matrix_readouts", None)
                edge_types = _symmetric_edge_types_for_labels(
                    readout=readout,
                    batch=batch,
                    basis_table=kwargs.get("basis_table"),
                    expected_nlabels=_flat_length(edge_error),
                )
                edge_types = np.abs(edge_types)
            stats.update(
                _block_type_stats(
                    node_losses,
                    node_pointers,
                    point_types,
                    "node",
                )
            )
            stats.update(
                _block_type_stats(
                    edge_losses,
                    edge_pointers,
                    edge_types,
                    "edge",
                )
            )

        if log_verbose:
            abs_node_error = abs(node_error)
            abs_edge_error = abs(edge_error)
            stats.update(
                {
                    "node_mean": abs_node_error.mean(),
                    "edge_mean": abs_edge_error.mean(),
                    "node_std": abs_node_error.std(),
                    "edge_std": abs_edge_error.std(),
                    "node_max": abs_node_error.max(),
                    "edge_max": abs_edge_error.max(),
                }
            )

        return loss, stats


class hamiltonian_composite_loss(OrbitalMatrixMetric):
    """Opt-in Hamiltonian loss combining label and coefficient objectives.

    The label component defaults to Smooth L1/Huber with ``beta=0.01`` because
    that was the strongest production-compatible candidate in the H2O loss
    hardening sweep. The coefficient component is disabled by default and uses
    the explicit coefficient-space API when ``coefficient_mse_weight > 0``.
    """

    requires_model_output = True

    def __init__(
        self,
        label_loss: str = "huber",
        beta: float = 0.01,
        coefficient_mse_weight: float = 0.0,
        node_weight: float = 1.0,
        edge_weight: float = 1.0,
        per_block_normalization: bool = False,
    ):
        if label_loss != "huber":
            raise ValueError(
                "hamiltonian_composite_loss currently supports only "
                "label_loss='huber'."
            )
        beta = float(beta)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError(
                f"hamiltonian_composite_loss beta must be positive, got {beta!r}."
            )
        self.label_loss = label_loss
        self.beta = beta
        self.coefficient_mse_weight = _validate_nonnegative(
            coefficient_mse_weight,
            "coefficient_mse_weight",
        )
        self.node_weight = _validate_nonnegative(node_weight, "node_weight")
        self.edge_weight = _validate_nonnegative(edge_weight, "edge_weight")
        if self.node_weight == 0 and self.edge_weight == 0:
            raise ValueError("At least one of node_weight or edge_weight must be > 0.")
        self.per_block_normalization = bool(per_block_normalization)

    def compute_metric(
        self,
        nodes_pred,
        nodes_ref,
        edges_pred,
        edges_ref,
        log_verbose=False,
        **kwargs,
    ) -> Tuple[float, Dict[str, float]]:
        node_error, edge_error = get_predictions_error(
            nodes_pred, nodes_ref, edges_pred, edges_ref
        )
        label_loss, node_loss, edge_loss = _weighted_label_huber(
            node_error,
            edge_error,
            beta=self.beta,
            node_weight=self.node_weight,
            edge_weight=self.edge_weight,
            per_block_normalization=self.per_block_normalization,
            batch=kwargs.get("batch"),
            basis_table=kwargs.get("basis_table"),
            model=kwargs.get("model"),
        )

        loss = label_loss
        stats = {
            "label_huber": label_loss,
            "node_smooth_l1": node_loss,
            "edge_smooth_l1": edge_loss,
            "node_rmse": (node_error**2).mean() ** (1 / 2),
            "edge_rmse": (edge_error**2).mean() ** (1 / 2),
            "coefficient_mse_weight": _as_stat_tensor(
                node_error,
                self.coefficient_mse_weight,
            ),
        }
        stats.update(_spin_channel_stats(node_error, edge_error))

        if self.coefficient_mse_weight > 0:
            coefficient_loss, coefficient_stats = _coefficient_space_metric(
                nodes_ref=nodes_ref,
                edges_ref=edges_ref,
                reduction="mse",
                **kwargs,
            )
            loss = loss + self.coefficient_mse_weight * coefficient_loss
            stats["coefficient_mse"] = coefficient_loss
            stats.update(
                {
                    f"coefficient_{key}": value
                    for key, value in coefficient_stats.items()
                }
            )

        if log_verbose:
            abs_node_error = abs(node_error)
            abs_edge_error = abs(edge_error)
            stats.update(
                {
                    "node_mean": abs_node_error.mean(),
                    "edge_mean": abs_edge_error.mean(),
                    "node_std": abs_node_error.std(),
                    "edge_std": abs_edge_error.std(),
                    "node_max": abs_node_error.max(),
                    "edge_max": abs_edge_error.max(),
                    "smooth_l1_beta": _as_stat_tensor(node_error, self.beta),
                }
            )

        return loss, stats


# @OrbitalMatrixMetric.from_metric_func
# def O2_d(
#     nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
# ) -> Tuple[float, Dict[str, float]]:
#     node_error, edge_error = get_predictions_error(
#         nodes_pred, nodes_ref, edges_pred, edges_ref
#     )

#     loss = 0.
#     for i in range(8, 13):
#         for j in range(8, 13):
#             loss = loss + (node_error[i * 13 + j::13**2 + 50]**2).sum()

#     return loss, {}

#     node_loss = abs(node_error / nodes_ref).mean()
#     edge_loss = abs(edge_error / edges_ref).mean()

#     stats = {
#         # "node_rmse": node_loss ** (1 / 2),
#         # "edge_rmse": edge_loss ** (1 / 2),
#     }

#     if log_verbose:
#         abs_node_error = abs(node_error)
#         abs_edge_error = abs(edge_error)

#         stats.update(
#             {
#                 "node_mean": abs_node_error.mean(),
#                 "edge_mean": abs_edge_error.mean(),
#                 "node_std": abs_node_error.std(),
#                 "edge_std": abs_edge_error.std(),
#                 "node_max": abs_node_error.max(),
#                 "edge_max": abs_edge_error.max(),
#             }
#         )

#     return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mape(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = abs(node_error / nodes_ref).mean()
    edge_loss = abs(edge_error / edges_ref[~_isnan(edges_ref)]).mean()

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mapemaemix(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_mape = abs(node_error / nodes_ref)
    edge_mape = abs(edge_error / edges_ref[~_isnan(edges_ref)])

    node_loss = node_mape[abs(nodes_ref) > 1e-6].mean() + (100 * abs(node_error).mean())
    edge_loss = edge_mape[abs(edges_ref[~_isnan(edges_ref)]) > 1e-6].mean() + (
        100 * abs(edge_error).mean()
    )

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mapemsemix(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_mape = abs(node_error / nodes_ref)
    edge_mape = abs(edge_error / edges_ref[~_isnan(edges_ref)])

    node_loss = (
        node_mape[abs(nodes_ref) > 1e-6].mean() + ((100 * node_error) ** 2).mean()
    )
    edge_loss = (
        edge_mape[abs(edges_ref[~_isnan(edges_ref)]) > 1e-6].mean()
        + ((100 * edge_error) ** 2).mean()
    )

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mapestdmix(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_mape = abs(node_error / nodes_ref)
    edge_mape = abs(edge_error / edges_ref[~_isnan(edges_ref)])

    node_mape = node_mape[abs(nodes_ref) > 1e-6]
    edge_mape = edge_mape[abs(edges_ref[~_isnan(edges_ref)]) > 1e-6]

    node_loss = node_mape.mean() + node_mape.std()
    edge_loss = node_mape.mean() + node_mape.std()

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def elementwise_mse(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = (node_error**2).mean()
    edge_loss = (edge_error**2).mean()

    n_node_els = node_error.shape[0]
    n_edge_els = edge_error.shape[0]

    loss = (n_node_els * node_loss + edge_loss * n_edge_els) / (n_node_els + n_edge_els)

    stats = {"node_rmse": node_loss ** (1 / 2), "edge_rmse": edge_loss ** (1 / 2)}

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return loss, stats


@OrbitalMatrixMetric.from_metric_func
def node_mse(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = (node_error**2).mean()
    edge_loss = (edge_error**2).mean()

    stats = {"node_rmse": node_loss ** (1 / 2), "edge_rmse": edge_loss ** (1 / 2)}

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss, stats


@OrbitalMatrixMetric.from_metric_func
def edge_mse(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = (node_error**2).mean()
    edge_loss = (edge_error**2).mean()

    stats = {"node_rmse": node_loss ** (1 / 2), "edge_rmse": edge_loss ** (1 / 2)}

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mse_threshold(
    nodes_pred,
    nodes_ref,
    edges_pred,
    edges_ref,
    threshold=1e-4,
    log_verbose=False,
    **kwargs,
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    n_node_els = node_error.shape[0]
    n_edge_els = edge_error.shape[0]

    abs_node_error = abs(node_error)
    abs_edge_error = abs(edge_error)

    node_error_above_thresh = node_error[abs_node_error > threshold]
    edge_error_above_thresh = edge_error[abs_edge_error > threshold]

    # We do the sum instead of the mean here so that we reward putting
    # elements below the threshold
    node_loss = (node_error_above_thresh**2).sum()
    edge_loss = (edge_error_above_thresh**2).sum()

    stats = {
        "node_rmse": (node_error**2).mean() ** (1 / 2),
        "edge_rmse": (edge_error**2).mean() ** (1 / 2),
        "node_above_threshold_frac": node_error_above_thresh.shape[0] / n_node_els,
        "edge_above_threshold_frac": edge_error_above_thresh.shape[0] / n_edge_els,
        "node_above_threshold_mean": abs_node_error[abs_node_error > threshold].mean(),
        "edge_above_threshold_mean": abs_edge_error[abs_edge_error > threshold].mean(),
    }

    if log_verbose:
        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mapemaemix(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_mape = abs(node_error / nodes_ref)
    edge_mape = abs(edge_error / edges_ref[~_isnan(edges_ref)])

    node_loss = node_mape[abs(nodes_ref) > 1e-6].mean() + (100 * abs(node_error).mean())
    edge_loss = edge_mape[abs(edges_ref[~_isnan(edges_ref)]) > 1e-6].mean() + (
        100 * abs(edge_error).mean()
    )

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mapemsemix(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_mape = abs(node_error / nodes_ref)
    edge_mape = abs(edge_error / edges_ref[~_isnan(edges_ref)])

    node_loss = (
        node_mape[abs(nodes_ref) > 1e-6].mean() + ((100 * node_error) ** 2).mean()
    )
    edge_loss = (
        edge_mape[abs(edges_ref[~_isnan(edges_ref)]) > 1e-6].mean()
        + ((100 * edge_error) ** 2).mean()
    )

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mse_sigmoid_thresh(
    nodes_pred,
    nodes_ref,
    edges_pred,
    edges_ref,
    threshold=1e-4,
    sigmoid_factor=1e-5,
    log_verbose=False,
    **kwargs,
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    n_node_els = node_error.shape[0]
    n_edge_els = edge_error.shape[0]

    abs_node_error = abs(node_error)
    abs_edge_error = abs(edge_error)

    def sigmoid_threshold(abs_errors, threshold):
        x = abs_errors - threshold
        sigmoid = 1 / (1 + np.e ** (-x / sigmoid_factor))
        return abs_errors * sigmoid

    node_error_thresholded = sigmoid_threshold(abs_node_error, threshold)
    edge_error_thresholded = sigmoid_threshold(abs_edge_error, threshold)

    # We do the sum instead of the mean here so that we reward putting
    # elements below the threshold
    node_loss = (node_error_thresholded**2).sum()
    edge_loss = (edge_error_thresholded**2).sum()

    abs_node_error_above_thresh = abs_node_error[abs_node_error > threshold]
    abs_edge_error_above_thresh = abs_edge_error[abs_edge_error > threshold]

    stats = {
        "node_rmse": (node_error**2).mean() ** (1 / 2),
        "edge_rmse": (edge_error**2).mean() ** (1 / 2),
        "node_above_threshold_frac": abs_node_error_above_thresh.shape[0] / n_node_els,
        "edge_above_threshold_frac": abs_edge_error_above_thresh.shape[0] / n_edge_els,
        "node_above_threshold_mean": abs_node_error_above_thresh.mean(),
        "edge_above_threshold_mean": abs_edge_error_above_thresh.mean(),
    }

    if log_verbose:
        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mae_sigmoid_thresh(
    nodes_pred,
    nodes_ref,
    edges_pred,
    edges_ref,
    threshold=1e-4,
    sigmoid_factor=1e-5,
    log_verbose=False,
    **kwargs,
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    n_node_els = node_error.shape[0]
    n_edge_els = edge_error.shape[0]

    abs_node_error = abs(node_error)
    abs_edge_error = abs(edge_error)

    def sigmoid_threshold(abs_errors, threshold):
        x = abs_errors - threshold
        sigmoid = 1 / (1 + np.e ** (-x / sigmoid_factor))
        return abs_errors * sigmoid

    node_error_thresholded = sigmoid_threshold(abs_node_error, threshold)
    edge_error_thresholded = sigmoid_threshold(abs_edge_error, threshold)

    # We do the sum instead of the mean here so that we reward putting
    # elements below the threshold
    node_loss = node_error_thresholded.sum()
    edge_loss = edge_error_thresholded.sum()

    abs_node_error_above_thresh = abs_node_error[abs_node_error > threshold]
    abs_edge_error_above_thresh = abs_edge_error[abs_edge_error > threshold]

    stats = {
        "node_rmse": (node_error**2).mean() ** (1 / 2),
        "edge_rmse": (edge_error**2).mean() ** (1 / 2),
        "node_above_threshold_frac": abs_node_error_above_thresh.shape[0] / n_node_els,
        "edge_above_threshold_frac": abs_edge_error_above_thresh.shape[0] / n_edge_els,
        "node_above_threshold_mean": abs_node_error_above_thresh.mean(),
        "edge_above_threshold_mean": abs_edge_error_above_thresh.mean(),
    }

    if log_verbose:
        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def normalized_density_error(
    nodes_pred,
    nodes_ref,
    edges_pred,
    edges_ref,
    batch,
    basis_table,
    grid_spacing: float = 0.1,
    log_verbose=False,
    **kwargs,
) -> Tuple[float, Dict[str, float]]:
    """Computes the normalized density error.

    This is the error of the density in real space divided by the number of electrons.
    """
    import sisl
    from graph2mat import BasisMatrixDataBase

    # Get the errors in the density matrix. Make sure that NaNs are set to 0, which
    # basically means that they will have no influence on the error.
    errors = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref, remove_nan=False
    )
    errors[1][_isnan(errors[1])] = 0

    if isinstance(batch, BasisMatrixDataBase):
        # We haven't really received a batch, but just a single structure.
        # Do as if we received a batch of size 1.
        matrix_error = copy(batch)
        matrix_error.point_labels = errors[0]
        matrix_error.edge_labels = errors[1]
        matrix_errors = [matrix_error.convert_to(Formats.SISL_DM)]
    else:
        # Create an iterator that returns the error for each structure in the batch
        # as a sisl DensityMatrix.
        # Note here that we assume that the model was trained under the assumption
        # that the density matrix is symmetric, so we set symmetric_matrix=True. (Might not be the case?)
        processor = MatrixDataProcessor(
            basis_table=basis_table, symmetric_matrix=True, out_matrix="density_matrix"
        )

        matrix_errors = processor.yield_from_batch(
            batch,
            predictions={"node_labels": errors[0], "edge_labels": errors[1]},
            as_matrix=True,
        )

    # Initialize counters
    per_config_total_error = 0
    total_error = 0
    total_electrons = 0
    num_configs = 0

    # Loop through structures in the batch
    for matrix_error in matrix_errors:
        # Initialize a grid to project the error
        error_grid = sisl.Grid(grid_spacing, geometry=matrix_error.geometry)

        # Project the error onto the grid
        matrix_error.density(error_grid)

        # We need the absolute value of the error
        grid_abs_error = abs(error_grid)

        # Aggregate all the errors and normalize by the number of electrons
        this_config_error = grid_abs_error.grid.sum() * error_grid.dvolume
        this_config_electrons = matrix_error.geometry.q0
        this_config_norm_error = this_config_error / this_config_electrons

        # Update counters
        per_config_total_error += this_config_norm_error
        total_error += this_config_error
        total_electrons += this_config_electrons
        num_configs += 1

    # Compute average errors
    avg_per_config_error = per_config_total_error / num_configs
    avg_error = total_error / total_electrons

    stats = {
        "avg_per_config_error_percent": avg_per_config_error * 100,
        "avg_error_percent": avg_error * 100,
    }

    # If the verbose stats are requested, we return also the counters.
    # This might be useful e.g. if we want to calculate the error over
    # epochs.
    if log_verbose:
        stats.update(
            {
                "per_config_total_error": per_config_total_error,
                "total_error": total_error,
                "total_electrons": total_electrons,
            }
        )

    return avg_per_config_error, stats
