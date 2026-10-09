"""Functions to assess performance.

When answering "How similar are these two matrices?", there is no
perfectly right answer. Depending on what you are most interested in,
you might use any of the functions implemented here.

Functions are wrapped into an ``OrbitalMatrixMetric`` class to make
sure they share the same interface and that they are all registered.
"""

from copy import copy

from typing import Any, Tuple, Dict, Type, Callable, Union, Optional, Iterable, TYPE_CHECKING, cast
import numpy as np
from scipy.special import expit

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None

from .formats import Formats
from .processing import MatrixDataProcessor
from .table import IdentityTypeMap

__all__ = [
    "OrbitalMatrixMetric",
    "block_type_mse",
    "block_type_mse_per_structure",
    "block_type_mse_per_domain",
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
    "pair_difference_mse",
    "gauge_invariant_mse",
    "elementwise_mae",
    "node_mse",
    "edge_mse",
    "block_type_mse_threshold",
    "block_type_mse_sigmoid_thresh",
    "block_type_mae_sigmoid_thresh",
    "normalized_density_error",
]


def _is_torch_tensor(value):
    return torch is not None and torch.is_tensor(value)


def _isnan(values):
    """NaN checking compatible with both torch and numpy"""
    return values != values


def _valid_label_rows(values):
    invalid = _isnan(values)
    return ~invalid.any(axis=1) if getattr(values, "ndim", 1) == 2 else ~invalid


def _numel(values):
    return values.numel() if _is_torch_tensor(values) else values.size


def _mean(values):
    return values.mean() if _numel(values) else values.sum()


def _std(values):
    return values.std() if _numel(values) > 1 else values.sum() * 0


def _max(values):
    return values.max() if _numel(values) else values.sum()


def _sigmoid_threshold(abs_errors, threshold, sigmoid_factor):
    x = (abs_errors - threshold) / sigmoid_factor
    return abs_errors * (torch.sigmoid(x) if _is_torch_tensor(x) else expit(x))


def get_predictions_error(
    nodes_pred, nodes_ref, edges_pred, edges_ref, remove_nan=True
):
    """Returns errors (pred - ref) for both nodes and edges, removing NaN values."""
    if remove_nan:
        nodes = _valid_label_rows(nodes_ref)
        edges = _valid_label_rows(edges_ref)
        return nodes_pred[nodes] - nodes_ref[nodes], edges_pred[edges] - edges_ref[edges]
    return nodes_pred - nodes_ref, edges_pred - edges_ref


def _relative_error(error, ref, eps: float = 1e-6):
    """|error / ref| for the elements with |ref| > eps.

    Dividing by the whole reference and filtering afterwards is not enough: exact zeros
    (e.g. symmetry-zero elements of H) give inf, and 0 * inf = NaN in the backward pass.
    """
    ref = ref[_valid_label_rows(ref)]
    mask = abs(ref) > eps
    ones = torch.ones_like(ref) if _is_torch_tensor(ref) else np.ones_like(ref)
    safe = ref * mask + ones * ~mask
    return abs(error / safe)[mask]


def _smooth_l1(values, beta: float):
    if beta <= 0:
        raise ValueError(f"Smooth L1/Huber beta must be positive, got {beta!r}.")
    abs_values = abs(values)
    quadratic = 0.5 * values**2 / beta
    linear = abs_values - 0.5 * beta
    if _is_torch_tensor(values):
        return torch.where(abs_values < beta, quadratic, linear)
    return np.where(abs_values < beta, quadratic, linear)


def _spin_channel_stats(node_error, edge_error):
    if getattr(node_error, "ndim", 1) != 2:
        return {}

    stats = {}
    for i in range(node_error.shape[1]):
        stats[f"node_rmse_spin{i}"] = _mean(node_error[:, i] ** 2) ** (1 / 2)
    if getattr(edge_error, "ndim", 1) == 2:
        for i in range(edge_error.shape[1]):
            stats[f"edge_rmse_spin{i}"] = _mean(edge_error[:, i] ** 2) ** (1 / 2)
    return stats


def _validate_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return value


def _as_stat_tensor(reference, value):
    if _is_torch_tensor(reference):
        return reference.new_tensor(value)
    return value


def _as_numpy_int(values) -> np.ndarray:
    if _is_torch_tensor(values):
        return values.detach().cpu().numpy().astype(int)
    return np.asarray(values).astype(int)


def _map_types(type_map, values) -> np.ndarray:
    values = _as_numpy_int(values)
    if _is_torch_tensor(type_map):
        mapped = type_map[
            torch.as_tensor(values, dtype=torch.long, device=type_map.device)
        ]
        return mapped.detach().cpu().numpy().astype(int)
    if isinstance(type_map, IdentityTypeMap):
        return values

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
    graph_edge_type = int(_map_types(readout.edge_types_to_graph2mat, [edge_type])[0])
    if readout.symmetric:
        graph_edge_type = abs(graph_edge_type)
    for module_key, operation in readout.interactions.items():
        _, _, op_edge_type = map(int, module_key[1:-1].split(","))
        if op_edge_type == graph_edge_type:
            return module_key, operation
    raise KeyError(f"No readout edge operation found for edge type {edge_type}")


def _readout_coefficient_metadata(readout) -> dict:
    if hasattr(readout, "coefficient_metadata"):
        return readout.coefficient_metadata()
    return {}


def _coefficient_readout(readout):
    """Resolve the common output coordinates of per-interaction readouts."""
    if torch is None or not isinstance(readout, torch.nn.ModuleList):
        return readout
    if len(readout) == 0:
        raise ValueError("Coefficient targets require at least one readout.")
    first = readout[0]
    metadata = _readout_coefficient_metadata(first)
    if not metadata:
        raise ValueError("Coefficient targets require readout coefficient metadata.")
    for other in readout[1:]:
        if (_readout_coefficient_metadata(other) != metadata
                or other.basis_table != first.basis_table
                or other.basis_grouping != first.basis_grouping
                or other.symmetric != first.symmetric):
            raise ValueError("Per-interaction readouts must have compatible coefficient coordinates.")
    return first


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
    if _is_torch_tensor(values):
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
    symmetric = getattr(readout, "symmetric", None)
    if symmetric is False:
        return edge_types

    if expected_nlabels is None:
        edge_labels = _batch_field(batch, "edge_labels")
        if edge_labels is not None:
            expected_nlabels = _flat_length(edge_labels)

    if symmetric is None and expected_nlabels is not None:
        # No model to ask: labels for every directed edge mean non-symmetric storage.
        all_edges = int(basis_table.edge_block_pointer(np.abs(edge_types))[-1])
        if int(expected_nlabels) == all_edges:
            return edge_types

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
    """Project target label rows into the readout coefficient space.

    Returns a dictionary with ``node`` and ``edge`` coefficient tensors keyed by
    readout operation, plus ``metadata`` describing those operations.
    """
    readout = _coefficient_readout(readout)
    basis_filters = getattr(readout, "basis_filters", None)
    if basis_filters is not None and not np.all(basis_filters):
        raise ValueError(
            "Coefficient targets with padding require a masked matrix loss; "
            "unobserved grouped elements cannot be assigned zero targets."
        )
    n_components = getattr(readout, "n_matrix_components", 1)
    component_shape = () if n_components == 1 else (n_components,)
    for reference in (nodes_ref, edges_ref):
        if (reference.ndim not in (1, 2)
                or (1 if reference.ndim == 1 else reference.shape[1]) != n_components):
            raise ValueError(
                f"Coefficient targets must have {n_components} matrix components "
                f"matching the readout, got shape {tuple(reference.shape)}."
            )

    point_types = _as_numpy_int(batch.point_types)
    point_labels = nodes_ref
    point_pointers = basis_table.point_block_pointer(point_types)
    graph_node_types = _map_types(readout.types_to_graph2mat, point_types)

    node_coeffs: Dict[str, list[Any]] = {}
    for atom_index, point_type in enumerate(point_types):
        op_key = f"node:{graph_node_types[atom_index]}"
        operation = readout.self_interactions[int(graph_node_types[atom_index])]
        if operation is None:
            continue
        shape = tuple(int(x) for x in basis_table.point_block_shape[:, point_type])
        block = point_labels[
            point_pointers[atom_index] : point_pointers[atom_index + 1]
        ].reshape(shape + component_shape)
        coeff = project_block_to_coefficients(
            block.unsqueeze(0), operation.change_of_basis
        ).squeeze(0)
        node_coeffs.setdefault(op_key, []).append(coeff)

    edge_labels = edges_ref
    edge_types = _symmetric_edge_types_for_labels(
        readout=readout,
        batch=batch,
        basis_table=basis_table,
        expected_nlabels=len(edge_labels),
    )
    edge_pointers = basis_table.edge_block_pointer(np.abs(edge_types))
    edge_coeffs: Dict[str, list[Any]] = {}
    for unique_edge_index, edge_type in enumerate(edge_types):
        module_key, operation = _edge_module_for_unique_type(readout, int(edge_type))
        if operation is None:
            continue
        op_key = f"edge:{module_key}"
        shape = tuple(int(x) for x in basis_table.edge_block_shape[:, abs(edge_type)])
        if edge_type < 0:
            shape = shape[::-1]
        block = edge_labels[
            edge_pointers[unique_edge_index] : edge_pointers[unique_edge_index + 1]
        ].reshape(shape + component_shape)
        if readout.symmetric and _map_types(readout.edge_types_to_graph2mat, [edge_type])[0] < 0:
            block = block.transpose(0, 1)
        coeff = project_block_to_coefficients(
            block.unsqueeze(0), operation.change_of_basis
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

    readout = _coefficient_readout(readout)
    point_types = _as_numpy_int(batch.point_types)
    graph_node_types = _map_types(readout.types_to_graph2mat, point_types)
    node_coeffs = _coefficient_values(coefficients, "node")
    node_offsets = {key: 0 for key in node_coeffs}
    node_parts = []
    node_filters = getattr(readout, "node_filters", None)

    for point_type, graph_node_type in zip(point_types, graph_node_types):
        op_key = f"node:{graph_node_type}"
        operation = readout.self_interactions[int(graph_node_type)]
        if operation is None:
            continue
        if op_key not in node_coeffs:
            raise ValueError(f"Missing predicted node coefficients for {op_key}.")

        offset = node_offsets[op_key]
        coeff = node_coeffs[op_key][offset : offset + 1]
        node_offsets[op_key] += 1
        labels = _flatten_reconstructed_block(_coefficients_to_block(operation, coeff))
        if node_filters is not None:
            labels = labels[node_filters[point_type].ravel()]
        node_parts.append(labels)

    edge_coeffs = _coefficient_values(coefficients, "edge")
    edge_filters = getattr(readout, "edge_filters", None)
    if (getattr(readout, "symmetric", None) is True
            and edge_filters is not None and not np.all(edge_filters)):
        # Counts must describe physical rows, not padded grouped block elements.
        original_types = _as_numpy_int(_batch_field(batch, "edge_types"))
        reference = _batch_field(batch, "edge_labels")
        expected_nlabels = (len(reference) if reference is not None else
                            int(basis_table.edge_block_pointer(np.abs(original_types[::2]))[-1]))
    else:
        expected_nlabels = _edge_label_count_from_coefficients(readout, coefficients)
    edge_types = _symmetric_edge_types_for_labels(
        readout=readout,
        batch=batch,
        basis_table=basis_table,
        expected_nlabels=expected_nlabels,
    )
    edge_offsets = {key: 0 for key in edge_coeffs}
    edge_parts = []

    for edge_type in edge_types:
        module_key, operation = _edge_module_for_unique_type(readout, int(edge_type))
        if operation is None:
            continue
        op_key = f"edge:{module_key}"
        if op_key not in edge_coeffs:
            raise ValueError(f"Missing predicted edge coefficients for {op_key}.")

        offset = edge_offsets[op_key]
        coeff = edge_coeffs[op_key][offset : offset + 1]
        edge_offsets[op_key] += 1
        block = _coefficients_to_block(operation, coeff)
        if readout.symmetric and _map_types(readout.edge_types_to_graph2mat, [edge_type])[0] < 0:
            block = block.transpose(1, 2)
        labels = _flatten_reconstructed_block(block)
        if edge_filters is not None:
            labels = labels[edge_filters[edge_type].ravel()]
        edge_parts.append(labels)

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

    template = next(iter(node_coeffs.values()), next(iter(edge_coeffs.values()), None))
    n_components = getattr(readout, "n_matrix_components", 1)
    empty_shape = (0,) if n_components == 1 else (0, n_components)
    node_labels = torch.cat(node_parts, dim=0) if node_parts else (
        template.new_empty(empty_shape) if template is not None else torch.empty(empty_shape)
    )
    if edge_parts:
        edge_labels = torch.cat(edge_parts, dim=0)
    else:
        edge_labels = node_labels.new_empty(empty_shape)

    return {"node_labels": node_labels, "edge_labels": edge_labels}


def _coefficient_space_errors(
    nodes_ref,
    edges_ref,
    batch,
    basis_table,
    out,
    model,
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

    errors = {}
    for kind in ("node", "edge"):
        predicted_values = _coefficient_values(predicted, kind)
        for key, target_value in _coefficient_values(target, kind).items():
            if key not in predicted_values:
                raise ValueError(f"Missing predicted {kind} coefficients for {key}.")
            prediction = predicted_values[key]
            valid = _valid_label_rows(target_value)
            errors[f"{kind}_{key}"] = prediction[valid] - target_value.to(prediction)[valid]
    return errors


def _coefficient_space_metric(nodes_ref, edges_ref, batch, basis_table, out, model, *, reduction):
    errors = _coefficient_space_errors(nodes_ref, edges_ref, batch, basis_table, out, model)
    losses = []
    stats = {}
    for key, error in errors.items():
        if reduction == "mse":
            loss = _mean(error**2)
            stats[f"{key}_coeff_rmse"] = loss ** (1 / 2)
        elif reduction == "mae":
            loss = _mean(abs(error))
            stats[f"{key}_coeff_mae"] = loss
        else:
            raise ValueError(f"Unknown coefficient reduction {reduction!r}")
        if error.numel():
            losses.append(loss)

    # Keep empty/unobserved outputs connected with zero gradient, without ever
    # multiplying a NaN prediction by zero.
    predictions = [value for kind in ("node", "edge") for value in out[f"{kind}_coefficients"].values()]
    predictions.extend(out[key] for key in ("node_labels", "edge_labels") if key in out)
    zero = (predictions[0] if predictions else nodes_ref).new_zeros(())
    for prediction in predictions:
        zero = zero + prediction.reshape(-1)[:0].sum()
    loss = torch.stack(losses).mean() + zero if losses else zero
    stats["coefficient_blocks"] = torch.as_tensor(
        len(losses), dtype=loss.dtype, device=loss.device
    )
    return loss, stats


def _loss_mean(values, pointers=None):
    if pointers is None:
        return _mean(values)
    losses = []
    for start, end in zip(pointers[:-1], pointers[1:]):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        losses.append(values[start:end].mean())
    if not losses:
        return _mean(values)
    if _is_torch_tensor(values):
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
    return losses


def _valid_block_pointers(pointers, reference):
    """Keep physical block boundaries aligned after dropping missing label rows."""
    if reference is None:
        return pointers
    valid = _as_numpy_int(_valid_label_rows(reference))
    if int(pointers[-1]) != len(valid):
        raise ValueError(
            "Block-normalized loss pointer mismatch: "
            f"last pointer is {int(pointers[-1])}, but labels have length {len(valid)}."
        )
    prefix = np.concatenate(([0], np.cumsum(valid)))
    return prefix[np.asarray(pointers, dtype=int)]


def _node_edge_block_pointers(
    batch,
    basis_table,
    model=None,
    expected_edge_nlabels: Optional[int] = None,
    nodes_ref=None,
    edges_ref=None,
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
        expected_nlabels=len(edges_ref) if edges_ref is not None else expected_edge_nlabels,
    )
    edge_pointers = basis_table.edge_block_pointer(np.abs(edge_types))
    return (_valid_block_pointers(node_pointers, nodes_ref),
            _valid_block_pointers(edge_pointers, edges_ref))


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
    types = np.asarray(types)[np.diff(pointers) > 0]
    stats = {}
    for type_id in sorted({int(type_value) for type_value in types}):
        selected = [
            loss
            for loss, type_value in zip(block_losses, types)
            if int(type_value) == type_id
        ]
        if not selected:
            continue
        if _is_torch_tensor(values):
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
    nodes_ref=None,
    edges_ref=None,
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
            nodes_ref=nodes_ref,
            edges_ref=edges_ref,
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
    requires_overlap_labels = False
    overlap_prediction_optional = False

    def __init__(self, **metric_kwargs):
        # Constructor kwargs (``loss_kwargs`` in the lightning models) become
        # defaults for every metric evaluation, e.g. per-domain weights.
        from .physical import validate_physical_weights

        weights = validate_physical_weights(**{name: metric_kwargs.get(name, 0.)
            for name in ("electron_weight", "mulliken_weight", "energy_weight")})
        if any(weights) and type(self) not in (elementwise_mse, gauge_invariant_mse):
            raise ValueError("physical objectives are supported only by elementwise_mse and gauge_invariant_mse")
        self.metric_kwargs = dict(metric_kwargs)

    def __call__(self, *args, **kwargs):
        return self.get_metric(*args, **kwargs)

    def get_metric(
        self, config_resolved=False, **kwargs
    ) -> Tuple[Union[float, np.ndarray], Dict[str, float]]:
        """Get the value for the metric.

        Parameters
        ----------
        config_resolved : bool, optional
            Whether the metric should be computed individually for each configuration in the batch, by default False.

            If False, the first tuple element is the metric for the whole batch.
            If True, it is an array with a metric for each configuration.

        Returns
        -------
        Tuple[Union[float, np.ndarray], Dict[str, float]]
            The metric value(s), as specified by config_resolved.

            The second tuple element contains additional stats, or an empty dict
            when config_resolved is True.
        """

        if getattr(self, "metric_kwargs", None):
            kwargs = {**self.metric_kwargs, **kwargs}

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

            metrics_array = np.zeros(batch.num_graphs, dtype=np.float64)

            for i, (pred, target) in enumerate(zip(pred_iterator, target_iterator)):
                kwargs["nodes_pred"] = pred.point_labels
                kwargs["edges_pred"] = pred.edge_labels
                kwargs["nodes_ref"] = target.point_labels
                kwargs["edges_ref"] = target.edge_labels

                kwargs["batch"] = target

                config_metric, _ = self.compute_metric(**kwargs)

                metrics_array[i] = config_metric

            return metrics_array, {}

    if TYPE_CHECKING:
        # Evaluated on instances; generated metrics also expose static functions.
        def compute_metric(
            self, nodes_pred, nodes_ref, edges_pred, edges_ref, **kwargs
        ) -> Tuple[float, Dict[str, float]]:
            ...
    else:
        @staticmethod
        def compute_metric(
            nodes_pred, nodes_ref, edges_pred, edges_ref, **kwargs
        ) -> Tuple[float, Dict[str, float]]:
            """Compute the metric and optionally return intermediate statistics."""
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

    node_loss = _mean(node_error**2)
    edge_loss = _mean(edge_error**2)

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
                "node_mean": _mean(abs_node_error),
                "edge_mean": _mean(abs_edge_error),
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
            }
        )

    return node_loss + edge_loss, stats


def _node_graph(batch) -> np.ndarray:
    """Structure of every node (batch.batch); an unbatched data object is one structure."""
    node_graph = _batch_field(batch, "batch")
    if node_graph is None:
        return np.zeros(len(_as_numpy_int(_batch_field(batch, "point_types"))), dtype=int)
    return _as_numpy_int(node_graph)


def _num_structures(batch, node_graph) -> int:
    count = _batch_field(batch, "num_graphs")
    if count is not None:
        return int(count)
    ptr = _batch_field(batch, "ptr")
    return len(ptr) - 1 if ptr is not None else (int(node_graph.max()) + 1 if node_graph.size else 1)


def _label_edge_mask(edges_ref, edge_types, edge_index, point_types, basis_table) -> np.ndarray:
    """Edges that carry labels: all of them, or one per pair for a symmetric matrix."""
    edge_sizes = basis_table.edge_block_size[np.abs(edge_types)]
    n_components = 1 if getattr(edges_ref, "ndim", 1) == 1 else int(edges_ref.shape[1])
    actual = _flat_length(edges_ref) // max(n_components, 1)
    if actual == int(edge_sizes.sum()):
        return np.ones(len(edge_types), dtype=bool)
    selector = MatrixDataProcessor(basis_table=basis_table, symmetric_matrix=True, sub_point_matrix=False)
    return selector._get_symmetric_unique_edge_mask(
        edge_types, expected_nlabels=int(actual), edge_index=edge_index, point_types=point_types
    )


def spectral_window_layout(edges_ref, batch, basis_table) -> dict:
    """Where every label element goes in its structure's Gamma-point matrix (roadmap 4.4).

    ``node_row``/``node_col`` and ``edge_row``/``edge_col``: orbital indices inside the
    structure. Node blocks are flattened row-major (orbitals of the node x orbitals of the
    node), edge blocks row-major (orbitals of ``edge_index[0]`` x of ``edge_index[1]``).
    With symmetric labels (one edge per pair) each edge element also stands for its
    transpose (``symmetric``). ``orb_struct``/``orb_local``: structure and local index of
    every orbital of the batch, to scatter per-orbital data such as ``spectral_C``.
    """
    point_types = _as_numpy_int(_batch_field(batch, "point_types"))
    edge_types = _as_numpy_int(_batch_field(batch, "edge_types"))
    edge_index = _as_numpy_int(_batch_field(batch, "edge_index"))
    node_graph = _node_graph(batch)
    sizes = basis_table.basis_size[point_types].astype(np.int64)
    n_structures = _num_structures(batch, node_graph)
    n_orb = np.bincount(node_graph, weights=sizes, minlength=n_structures).astype(np.int64)
    start = np.cumsum(n_orb) - n_orb  # the nodes of a structure are contiguous in a batch
    local = np.cumsum(sizes) - sizes - start[node_graph]

    def block_pairs(row_nodes, col_nodes):
        n_rows, n_cols = sizes[row_nodes], sizes[col_nodes]
        n = n_rows * n_cols
        owner = np.repeat(np.arange(len(n)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        return local[row_nodes][owner] + k // n_cols[owner], local[col_nodes][owner] + k % n_cols[owner]

    nodes = np.arange(len(point_types))
    node_row, node_col = block_pairs(nodes, nodes)
    labelled = _label_edge_mask(edges_ref, edge_types, edge_index, point_types, basis_table)
    edge_row, edge_col = block_pairs(edge_index[0, labelled], edge_index[1, labelled])
    orb_struct = np.repeat(node_graph, sizes)
    orb_local = np.arange(int(sizes.sum())) - start[orb_struct]
    as_long = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long)
    return {
        "node_row": as_long(node_row), "node_col": as_long(node_col),
        "edge_row": as_long(edge_row), "edge_col": as_long(edge_col),
        "orb_struct": as_long(orb_struct), "orb_local": as_long(orb_local),
        "no_max": int(n_orb.max()) if n_orb.size else 0,
        "symmetric": not bool(labelled.all()),
    }


def _spectral_window_loss(d_nodes, d_edges, node_struct, edge_struct, n_structures, batch, edges_ref, basis_table):
    """``|| C^T dH(Gamma) C - mean(diag) I ||^2`` per structure (roadmap 4.4, audit LOSS_TRAIN-15).

    ``C`` (``batch.spectral_C``): reference eigenvectors of a window around E_F at Gamma,
    S-orthonormal, one row per orbital, zero columns as padding. The diagonal is the
    first-order error of each eigenvalue and the off-diagonal the coupling inside the
    window. ``C^T S C = I``, so the gauge shift ``c S`` is the mean of the diagonal and is
    removed. It only needs dH = prediction - label, so it holds for Delta-learning too.
    """
    from graph2mat.core.modules.graph2mat import LAYOUT_ATTR

    C = _batch_field(batch, "spectral_C")
    if C is None:
        raise ValueError("spectral_weight needs batch.spectral_C (MatrixDataModule(spectral_window=True)).")
    if not torch.isfinite(C).all():
        raise ValueError("spectral_weight needs finite batch.spectral_C.")
    layout = getattr(batch, LAYOUT_ATTR, None) or {}
    if "node_row" not in layout:
        layout = spectral_window_layout(edges_ref, batch, basis_table)
    dev = d_nodes.device
    get = lambda key: layout[key].to(dev)
    rows, cols = [get("node_row"), get("edge_row")], [get("node_col"), get("edge_col")]
    structs, values = [node_struct, edge_struct], [d_nodes, d_edges]
    if layout["symmetric"]:
        rows.append(get("edge_col")); cols.append(get("edge_row"))
        structs.append(edge_struct); values.append(d_edges)
    no = layout["no_max"]
    d_h = d_nodes.new_zeros((n_structures, no, no)).index_put(
        (torch.cat(structs), torch.cat(rows), torch.cat(cols)), torch.cat(values), accumulate=True
    )
    C = C.to(device=dev, dtype=d_nodes.dtype)
    c_pad = C.new_zeros((n_structures, no, C.shape[1])).index_put((get("orb_struct"), get("orb_local")), C)
    m = c_pad.transpose(1, 2) @ d_h @ c_pad
    mask = (c_pad.abs().sum(1) > 0).to(m.dtype)  # real (non-padding) window columns
    k = mask.sum(1)
    shift = (torch.diagonal(m, dim1=1, dim2=2) * mask).sum(1) / k.clamp_min(1)
    residual = m - shift[:, None, None] * torch.diag_embed(mask)
    count = (k**2).sum()
    return (residual**2).sum() / count.clamp_min(1), count


def _per_element_structure_ids(nodes_ref, edges_ref, batch, basis_table):
    """Structure id of every flat node/edge label element (torch tensors).

    Handles symmetric-matrix edge selection (labels stored for one direction
    only) and the NaN-edge filtering applied by ``get_predictions_error``. Batches from
    the Lightning DataModule carry them already (their type layout, roadmap 6.3), which
    avoids a device -> host copy per step.
    """
    from graph2mat.core.modules.graph2mat import LAYOUT_ATTR

    layout = batch.get(LAYOUT_ATTR) if isinstance(batch, dict) else getattr(batch, LAYOUT_ATTR, None)
    if layout is not None and "node_struct" in layout:
        return layout["node_struct"], layout["edge_struct"], layout["n_structures"]

    point_types = _batch_field(batch, "point_types")
    if point_types is None:
        raise ValueError("per-structure losses need batch.point_types.")
    point_types = _as_numpy_int(point_types)
    node_graph = _node_graph(batch)
    node_pointers = basis_table.point_block_pointer(point_types)
    node_sizes = np.diff(node_pointers)
    node_struct = np.repeat(node_graph, node_sizes)

    edge_types = _as_numpy_int(_batch_field(batch, "edge_types"))
    edge_index = _as_numpy_int(_batch_field(batch, "edge_index"))
    labelled = _label_edge_mask(edges_ref, edge_types, edge_index, point_types, basis_table)
    edge_graph = node_graph[edge_index[0]][labelled]
    edge_sizes = basis_table.edge_block_size[np.abs(edge_types)][labelled]
    edge_struct = np.repeat(edge_graph, edge_sizes)

    n_structures = _num_structures(batch, node_graph)
    return (
        torch.as_tensor(node_struct, dtype=torch.long),
        torch.as_tensor(edge_struct, dtype=torch.long),
        int(n_structures),
    )


def _structure_mean_sq(error, struct_ids, n_structures):
    """Differentiable per-structure mean of squared errors. NaN-safe counts."""
    error = error if _is_torch_tensor(error) else torch.as_tensor(np.asarray(error))
    n_components = int(np.prod(error.shape[1:]))
    squared = error.reshape(error.shape[0], n_components).pow(2).sum(dim=1)
    sums = torch.zeros(n_structures, dtype=squared.dtype, device=squared.device)
    sums = sums.index_add(0, struct_ids.to(squared.device), squared)
    counts = torch.zeros(n_structures, dtype=squared.dtype, device=squared.device)
    counts = counts.index_add(
        0, struct_ids.to(squared.device), torch.ones_like(squared)
    )
    counts = counts * n_components
    return sums / counts.clamp(min=1), counts > 0


def _per_structure_losses(
    nodes_pred, nodes_ref, edges_pred, edges_ref, batch, basis_table,
    node_weight, edge_weight,
):
    node_struct, edge_struct, n_structures = _per_element_structure_ids(
        nodes_ref, edges_ref, batch, basis_table
    )
    # struct ids are built on CPU (torch.as_tensor with no device); move them to
    # the data device so indexing/index_add against CUDA tensors does not raise
    # "indices should be either on cpu or on the same device".
    node_struct = node_struct.to(nodes_pred.device)
    edge_struct = edge_struct.to(edges_pred.device)
    node_error, edge_error = get_predictions_error(nodes_pred, nodes_ref, edges_pred, edges_ref)
    node_struct = node_struct[torch.as_tensor(_valid_label_rows(nodes_ref), device=node_struct.device)]
    edge_struct = edge_struct[torch.as_tensor(_valid_label_rows(edges_ref), device=edge_struct.device)]
    node_loss_s, node_present = _structure_mean_sq(node_error, node_struct, n_structures)
    edge_loss_s, edge_present = _structure_mean_sq(edge_error, edge_struct, n_structures)
    loss_s = node_weight * node_loss_s + edge_weight * edge_loss_s
    stats = {
        "node_rmse": float(node_loss_s[node_present].mean() ** 0.5) if node_present.any() else 0.0,
        "edge_rmse": float(edge_loss_s[edge_present].mean() ** 0.5) if edge_present.any() else 0.0,
    }
    return loss_s, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mse_per_structure(
    nodes_pred, nodes_ref, edges_pred, edges_ref,
    batch=None, basis_table=None,
    node_weight: float = 1.0, edge_weight: float = 1.0,
    log_verbose=False, **kwargs,
) -> Tuple[float, Dict[str, float]]:
    """MSE normalized inside each structure, then averaged over structures.

    Every structure contributes the same weight to the loss regardless of its
    size — a 50-atom snapshot no longer dominates a 2-atom one just by having
    more matrix elements (training_weighting_policy='per_structure').
    """
    loss_s, stats = _per_structure_losses(
        nodes_pred, nodes_ref, edges_pred, edges_ref, batch, basis_table,
        node_weight, edge_weight,
    )
    stats["n_structures"] = int(loss_s.shape[0])
    return _mean(loss_s), stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mse_per_domain(
    nodes_pred, nodes_ref, edges_pred, edges_ref,
    batch=None, basis_table=None,
    domain_threshold_atoms: Optional[int] = None,
    small_domain_weight: float = 0.5,
    large_domain_weight: float = 0.5,
    node_weight: float = 1.0, edge_weight: float = 1.0,
    log_verbose=False, **kwargs,
) -> Tuple[float, Dict[str, float]]:
    """Per-structure MSE averaged inside each size domain, then across domains.

    ``domain_threshold_atoms`` is required (structures with at least that many
    atoms are 'large'); domain weights are configurable
    (training_weighting_policy='per_domain').
    """
    if domain_threshold_atoms is None:
        raise ValueError(
            "block_type_mse_per_domain requires domain_threshold_atoms "
            "(pass it via loss_kwargs)."
        )
    loss_s, stats = _per_structure_losses(
        nodes_pred, nodes_ref, edges_pred, edges_ref, batch, basis_table,
        node_weight, edge_weight,
    )
    atoms_per_structure = np.bincount(_node_graph(batch), minlength=loss_s.shape[0])
    large_mask = torch.as_tensor(
        atoms_per_structure >= int(domain_threshold_atoms), device=loss_s.device
    )
    parts = []
    weights = []
    if (~large_mask).any():
        parts.append(loss_s[~large_mask].mean())
        weights.append(float(small_domain_weight))
    if large_mask.any():
        parts.append(loss_s[large_mask].mean())
        weights.append(float(large_domain_weight))
    total_weight = sum(weights)
    loss = sum(w * p for w, p in zip(weights, parts)) / total_weight if parts else _mean(loss_s)
    stats.update(
        {
            "n_structures": int(loss_s.shape[0]),
            "n_large_structures": int(large_mask.sum()),
            "small_domain_weight": float(small_domain_weight),
            "large_domain_weight": float(large_domain_weight),
        }
    )
    return loss, stats


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

    node_loss = _mean(abs(node_error))
    edge_loss = _mean(abs(edge_error))

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": _mean(abs_node_error),
                "edge_mean": _mean(abs_edge_error),
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
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

        node_loss = _mean(_smooth_l1(node_error, self.beta))
        edge_loss = _mean(_smooth_l1(edge_error, self.beta))

        stats = {
            "node_smooth_l1": node_loss,
            "edge_smooth_l1": edge_loss,
            "node_rmse": _mean(node_error**2) ** (1 / 2),
            "edge_rmse": _mean(edge_error**2) ** (1 / 2),
        }
        stats.update(_spin_channel_stats(node_error, edge_error))

        if log_verbose:
            abs_node_error = abs(node_error)
            abs_edge_error = abs(edge_error)
            stats.update(
                {
                    "node_mean": _mean(abs_node_error),
                    "edge_mean": _mean(abs_edge_error),
                    "node_std": _std(abs_node_error),
                    "edge_std": _std(abs_edge_error),
                    "node_max": _max(abs_node_error),
                    "edge_max": _max(abs_edge_error),
                    "smooth_l1_beta": node_error.new_tensor(self.beta)
                    if _is_torch_tensor(node_error)
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
            nodes_ref=nodes_ref,
            edges_ref=edges_ref,
        )

        node_losses = _smooth_l1(node_error, self.beta)
        edge_losses = _smooth_l1(edge_error, self.beta)
        node_block_losses = _loss_blocks(node_losses, node_pointers)
        edge_block_losses = _loss_blocks(edge_losses, edge_pointers)

        if not node_block_losses:
            node_loss = node_losses.sum()
        elif _is_torch_tensor(node_losses):
            node_loss = torch.stack(node_block_losses).mean()
        else:
            node_loss = np.stack(node_block_losses).mean()
        if not edge_block_losses:
            edge_loss = edge_losses.sum()
        elif _is_torch_tensor(edge_losses):
            edge_loss = torch.stack(edge_block_losses).mean()
        else:
            edge_loss = np.stack(edge_block_losses).mean()
        loss = self.node_weight * node_loss + self.edge_weight * edge_loss

        stats = {
            "node_block_huber": node_loss,
            "edge_block_huber": edge_loss,
            "node_rmse": _mean(node_error**2) ** (1 / 2),
            "edge_rmse": _mean(edge_error**2) ** (1 / 2),
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
                    expected_nlabels=len(edges_ref),
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
                    "node_mean": _mean(abs_node_error),
                    "edge_mean": _mean(abs_edge_error),
                    "node_std": _std(abs_node_error),
                    "edge_std": _std(abs_edge_error),
                    "node_max": _max(abs_node_error),
                    "edge_max": _max(abs_edge_error),
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
            nodes_ref=nodes_ref,
            edges_ref=edges_ref,
        )

        loss = label_loss
        stats = {
            "label_huber": label_loss,
            "node_smooth_l1": node_loss,
            "edge_smooth_l1": edge_loss,
            "node_rmse": _mean(node_error**2) ** (1 / 2),
            "edge_rmse": _mean(edge_error**2) ** (1 / 2),
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
                    "node_mean": _mean(abs_node_error),
                    "edge_mean": _mean(abs_edge_error),
                    "node_std": _std(abs_node_error),
                    "edge_std": _std(abs_edge_error),
                    "node_max": _max(abs_node_error),
                    "edge_max": _max(abs_edge_error),
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

    node_loss = _mean(_relative_error(node_error, nodes_ref))
    edge_loss = _mean(_relative_error(edge_error, edges_ref))

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": _mean(abs_node_error),
                "edge_mean": _mean(abs_edge_error),
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
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

    node_mape = _relative_error(node_error, nodes_ref)
    edge_mape = _relative_error(edge_error, edges_ref)

    node_loss = _mean(node_mape) + (100 * _mean(abs(node_error)))
    edge_loss = _mean(edge_mape) + (100 * _mean(abs(edge_error)))

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": _mean(abs_node_error),
                "edge_mean": _mean(abs_edge_error),
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
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

    node_mape = _relative_error(node_error, nodes_ref)
    edge_mape = _relative_error(edge_error, edges_ref)

    node_loss = _mean(node_mape) + _mean((100 * node_error) ** 2)
    edge_loss = _mean(edge_mape) + _mean((100 * edge_error) ** 2)

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": _mean(abs_node_error),
                "edge_mean": _mean(abs_edge_error),
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
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

    node_mape = _relative_error(node_error, nodes_ref)
    edge_mape = _relative_error(edge_error, edges_ref)

    node_loss = _mean(node_mape) + _std(node_mape)
    edge_loss = _mean(edge_mape) + _std(edge_mape)

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": _mean(abs_node_error),
                "edge_mean": _mean(abs_edge_error),
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def elementwise_mse(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, edge_weight: float = 1.0,
    electron_weight: float = 0.0, mulliken_weight: float = 0.0, energy_weight: float = 0.0, **kwargs
) -> Tuple[float, Dict[str, float]]:
    """MSE over all matrix elements.

    ``edge_weight`` counts each edge element that many times. With symmetric storage
    each stored edge block also stands for its transpose, so ``edge_weight=2`` gives
    the MSE over the full matrix.
    """
    from .physical import physical_loss_terms, validate_physical_weights

    weights = validate_physical_weights(electron_weight, mulliken_weight, energy_weight)
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = _mean(node_error**2)
    edge_loss = _mean(edge_error**2)

    n_node_els = _numel(node_error)
    n_edge_els = _numel(edge_error) * edge_weight

    loss = (n_node_els * node_loss + edge_loss * n_edge_els) / (n_node_els + n_edge_els or 1)

    stats = {"node_rmse": node_loss ** (1 / 2), "edge_rmse": edge_loss ** (1 / 2)}

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": _mean(abs_node_error),
                "edge_mean": _mean(abs_edge_error),
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
            }
        )

    if any(weights):
        physical, physical_stats = physical_loss_terms(nodes_pred, edges_pred,
            batch=kwargs.get("batch"), basis_table=kwargs.get("basis_table"),
            electron_weight=weights[0], mulliken_weight=weights[1], energy_weight=weights[2])
        loss = loss + physical
        stats.update(physical_stats)
    return loss, stats


def find_displacement_pairs(batch, max_displacement: float, min_displacement: float = 0.0) -> list[tuple[int, int]]:
    """Structures of a batch that are +/- finite-difference partners.

    Two structures pair when they have the same number of atoms and the largest atom
    displacement between them (minimum image, Ang) is in ``(min_displacement,
    max_displacement]``. Partners of a +/-delta pair differ by exactly 2 delta, so a narrow
    window around it separates them from other structures that share the template. Atoms must be in
    the same order (the pairs come from the same template). Ambiguous matches raise.
    """
    positions = _batch_field(batch, "positions")
    positions = np.asarray(positions.detach().cpu() if _is_torch_tensor(positions) else positions, dtype=float)
    node_graph = _node_graph(batch)
    n_structures = _num_structures(batch, node_graph)
    cell = _batch_field(batch, "cell")
    cell = None if cell is None else np.asarray(cell.detach().cpu() if _is_torch_tensor(cell) else cell,
                                                dtype=float).reshape(n_structures, 3, 3)
    atoms = [np.flatnonzero(node_graph == s) for s in range(n_structures)]
    pairs, used = [], set()
    for a in range(n_structures):
        for b in range(a + 1, n_structures):
            if not len(atoms[a]) or len(atoms[a]) != len(atoms[b]):
                continue
            delta = positions[atoms[a]] - positions[atoms[b]]
            if cell is not None and np.allclose(cell[a], cell[b]):
                periodic = np.abs(np.linalg.det(cell[a])) > 1e-12
                if periodic:
                    frac = np.linalg.solve(cell[a].T, delta.T).T
                    delta = (frac - np.round(frac)) @ cell[a]
            moved = np.linalg.norm(delta, axis=1).max()
            if max(min_displacement, 0.0) < moved <= max_displacement:
                if a in used or b in used:
                    raise ValueError(f"structures {a} and {b}: ambiguous +/- pairing")
                pairs.append((a, b))
                used |= {a, b}
    return pairs


def _gamma_matrices(nodes, edges, edges_ref, batch, basis_table):
    """Gamma matrices, with any label-component axes after (n_structures, no, no)."""
    from graph2mat.core.modules.graph2mat import LAYOUT_ATTR

    layout = getattr(batch, LAYOUT_ATTR, None) or {}
    if "node_row" not in layout:
        layout = spectral_window_layout(edges_ref, batch, basis_table)
    node_struct, edge_struct, n_structures = _per_element_structure_ids(nodes, edges_ref, batch, basis_table)
    dev = nodes.device
    get = lambda key: layout[key].to(dev)
    rows, cols = [get("node_row"), get("edge_row")], [get("node_col"), get("edge_col")]
    structs, values = [node_struct.to(dev), edge_struct.to(dev)], [nodes, edges]
    if layout["symmetric"]:
        rows.append(get("edge_col")); cols.append(get("edge_row"))
        structs.append(edge_struct.to(dev)); values.append(edges)
    no = layout["no_max"]
    return nodes.new_zeros((n_structures, no, no) + tuple(nodes.shape[1:])).index_put(
        (torch.cat(structs), torch.cat(rows), torch.cat(cols)), torch.cat(values), accumulate=True
    ), node_struct, edge_struct


@OrbitalMatrixMetric.from_metric_func
def pair_difference_mse(
    nodes_pred, nodes_ref, edges_pred, edges_ref, batch=None, basis_table=None,
    edge_weight: float = 1.0, pair_weight: float = 1.0, pair_max_displacement: float = 0.05,
    pair_min_displacement: float = 0.0, expected_pairs: Optional[int] = None, log_verbose=False, **kwargs,
) -> Tuple[float, Dict[str, float]]:
    """``elementwise_mse`` plus the error of the +/- pair differences at Gamma.

    For every +/- finite-difference pair in the batch (``find_displacement_pairs``) the
    term is the mean square of ``(H_a - H_b)_pred - (H_a - H_b)_ref`` at the Gamma point,
    i.e. of ``E_a(Gamma) - E_b(Gamma)`` with ``E = pred - ref``, per label element of the
    structure. Plain MSE lets the errors of both partners be equal and large; this term
    penalises their antisymmetric part, the error of the slope. It has the units and
    normalisation of the H loss, so ``pair_weight=1`` weighs a difference element like a
    value element. ``expected_pairs`` makes a training batch with any other number of pairs
    (including none) an error, so a pairing failure never trains silently without the term.
    Paired structures need complete reference matrices; a missing element cannot
    define their Gamma-point difference. Unpaired structures retain NaN-safe MSE.
    Multiple label components are averaged as scalar elements, as in the base MSE.

    The term is a training signal only: with ``model`` in eval mode (validation) the loss
    is the plain ``elementwise_mse``, so ``val_loss`` and checkpoint selection match runs
    without pairs, and small-amplitude validation structures are never mistaken for pairs.
    """
    loss, stats = cast(Callable[..., Tuple[float, dict]], elementwise_mse.compute_metric)(
        nodes_pred=nodes_pred, nodes_ref=nodes_ref, edges_pred=edges_pred, edges_ref=edges_ref,
        edge_weight=edge_weight, log_verbose=log_verbose,
    )
    model = kwargs.get("model")
    if model is not None and not getattr(model, "training", True):
        return loss, stats
    pairs = find_displacement_pairs(batch, pair_max_displacement, pair_min_displacement)
    if expected_pairs is not None and len(pairs) != int(expected_pairs):  # training batches only
        raise ValueError(f"found {len(pairs)} +/- pairs in the batch, expected {expected_pairs}")
    stats["n_pairs"] = len(pairs)
    if not pairs:
        stats["pair_rmse"] = 0.0
        return loss, stats
    errors, node_struct, edge_struct = _gamma_matrices(
        nodes_pred - nodes_ref, edges_pred - edges_ref, edges_ref, batch, basis_table
    )
    partners = torch.as_tensor(pairs, device=errors.device).reshape(-1)
    for ref, ids in ((nodes_ref, node_struct), (edges_ref, edge_struct)):
        unknown = ids.to(ref.device)[~_valid_label_rows(ref)]
        if torch.isin(unknown, partners.to(ref.device)).any():
            raise ValueError("pair_difference_mse requires complete reference labels for paired structures.")
    counts = torch.bincount(node_struct, minlength=errors.shape[0]) + edge_weight * torch.bincount(
        edge_struct, minlength=errors.shape[0]
    )
    a, b = (torch.as_tensor([p[i] for p in pairs], device=errors.device) for i in (0, 1))
    components = int(np.prod(nodes_pred.shape[1:]))
    per_pair = ((errors[a] - errors[b]) ** 2).sum(dim=tuple(range(1, errors.ndim))) / (
        counts.to(errors.device)[a].clamp_min(1) * components
    )
    pair_loss = per_pair.mean()
    stats["pair_rmse"] = pair_loss.detach() ** 0.5
    return loss + pair_weight * pair_loss, stats


@OrbitalMatrixMetric.from_metric_func
def gauge_invariant_mse(
    nodes_pred, nodes_ref, edges_pred, edges_ref, batch=None, basis_table=None,
    edge_weight: float = 1.0, overlap_weight: float = 1.0, spectral_weight: float = 0.0,
    log_verbose=False, electron_weight: float = 0.0, mulliken_weight: float = 0.0,
    energy_weight: float = 0.0, **kwargs,
) -> Tuple[float, Dict[str, float]]:
    """Elementwise MSE of H that ignores a per-structure energy shift H -> H + c S.

    The energy zero of a structure (E_F, mean electrostatic potential) cannot be
    inferred by a local model (roadmap 4.2, audit PHYS_H-03). Labels must carry the
    overlap as their last component (``matrix_component_policy="h_and_overlap"``).
    For each structure and H component, ``c = sum(w dH S) / sum(w S^2)`` (closed form,
    differentiable) and the H loss is the MSE of ``dH - c S``. The model may predict
    (H, S), and then the predicted overlap is fitted with a plain MSE weighted by
    ``overlap_weight``, or only H (one component fewer than the labels): S is then just
    the label that defines the shift, and the readout is half the size.
    ``edge_weight`` as in ``elementwise_mse``. Rows with any NaN H/S label
    component are unobserved and excluded before alignment and subtraction.
    ``spectral_weight`` > 0 adds the Gamma-point window loss of ``_spectral_window_loss``
    (one H component; needs finite ``batch.spectral_C`` and complete H/S labels).
    """
    from .physical import physical_loss_terms, validate_physical_weights

    weights = validate_physical_weights(electron_weight, mulliken_weight, energy_weight)
    physical_nodes_pred, physical_edges_pred = nodes_pred, edges_pred
    node_struct, edge_struct, n_structures = _per_element_structure_ids(
        nodes_ref, edges_ref, batch, basis_table
    )
    node_struct = node_struct.to(nodes_pred.device)
    edge_struct = edge_struct.to(edges_pred.device)
    valid_nodes, valid_edges = _valid_label_rows(nodes_ref), _valid_label_rows(edges_ref)
    if spectral_weight and (not valid_nodes.all() or not valid_edges.all()):
        raise ValueError("spectral_weight requires complete H/S reference labels.")
    node_struct, edge_struct = node_struct[valid_nodes], edge_struct[valid_edges]
    nodes_pred, nodes_ref = nodes_pred[valid_nodes], nodes_ref[valid_nodes]
    edges_pred, edges_ref = edges_pred[valid_edges], edges_ref[valid_edges]
    n_h = nodes_ref.shape[1] - 1
    if nodes_pred.ndim == 1:  # a single predicted component: H only
        nodes_pred, edges_pred = nodes_pred[:, None], edges_pred[:, None]
    predicts_overlap = nodes_pred.shape[1] == nodes_ref.shape[1]
    d_nodes = nodes_pred[:, :n_h] - nodes_ref[:, :n_h]
    d_edges = edges_pred[:, :n_h] - edges_ref[:, :n_h]
    s_nodes, s_edges = nodes_ref[:, -1:], edges_ref[:, -1:]

    def per_structure(values, struct):
        out = torch.zeros((n_structures, values.shape[1]), dtype=values.dtype, device=values.device)
        return out.index_add(0, struct, values)

    num = per_structure(d_nodes * s_nodes, node_struct) + edge_weight * per_structure(
        d_edges * s_edges, edge_struct
    )
    den = per_structure(s_nodes**2, node_struct) + edge_weight * per_structure(s_edges**2, edge_struct)
    shift = num / den.clamp_min(1e-12)  # (n_structures, n_h)

    aligned_nodes = d_nodes - shift[node_struct] * s_nodes
    aligned_edges = d_edges - shift[edge_struct] * s_edges
    n_elements = nodes_ref.shape[0] + edge_weight * edges_ref.shape[0]
    denominator = n_elements if n_elements > 0 else 1
    h_loss = ((aligned_nodes**2).sum() + edge_weight * (aligned_edges**2).sum()) / (denominator * n_h)
    if predicts_overlap:
        s_loss = (((nodes_pred[:, -1] - s_nodes[:, 0]) ** 2).sum()
                  + edge_weight * ((edges_pred[:, -1] - s_edges[:, 0]) ** 2).sum()) / denominator
    else:
        s_loss = h_loss.new_zeros(())

    stats = {
        "gauge_h_rmse": h_loss.detach() ** 0.5,
        "overlap_rmse": s_loss.detach() ** 0.5,
        "gauge_shift_mean_abs": _mean(shift.detach().abs()),
    }
    loss = h_loss + overlap_weight * s_loss
    if spectral_weight:
        if n_h != 1:
            raise ValueError("spectral_weight supports a single H component.")
        spectral, count = _spectral_window_loss(
            d_nodes[:, 0], d_edges[:, 0], node_struct, edge_struct, n_structures, batch, edges_ref, basis_table
        )
        stats["spectral_rmse"] = spectral.detach() ** 0.5
        stats["spectral_elements"] = count.detach()
        loss = loss + spectral_weight * spectral
    if any(weights):
        aligned_nodes_full = aligned_nodes.new_zeros((physical_nodes_pred.shape[0], n_h)).index_copy(
            0, valid_nodes.nonzero(as_tuple=True)[0], aligned_nodes)
        aligned_edges_full = aligned_edges.new_zeros((physical_edges_pred.shape[0], n_h)).index_copy(
            0, valid_edges.nonzero(as_tuple=True)[0], aligned_edges)
        physical, physical_stats = physical_loss_terms(physical_nodes_pred, physical_edges_pred,
            batch=batch, basis_table=basis_table, electron_weight=weights[0],
            mulliken_weight=weights[1], energy_weight=weights[2],
            aligned_nodes=aligned_nodes_full, aligned_edges=aligned_edges_full)
        loss = loss + physical
        stats.update(physical_stats)
    return loss, stats


# Only meaningful with (H, S) labels: not part of the metrics run on any matrix by default.
gauge_invariant_mse.requires_overlap_labels = True
# The prediction may omit S (n_output_components one fewer than the labels).
gauge_invariant_mse.overlap_prediction_optional = True


@OrbitalMatrixMetric.from_metric_func
def elementwise_mae(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    abs_node_error = abs(node_error)
    abs_edge_error = abs(edge_error)
    node_loss = _mean(abs_node_error)
    edge_loss = _mean(abs_edge_error)

    n_node_els = _numel(node_error)
    n_edge_els = _numel(edge_error)

    loss = (n_node_els * node_loss + edge_loss * n_edge_els) / (n_node_els + n_edge_els or 1)

    stats = {
        "node_rmse": _mean(node_error**2) ** (1 / 2),
        "edge_rmse": _mean(edge_error**2) ** (1 / 2),
    }

    if log_verbose:
        stats.update(
            {
                "node_mean": node_loss,
                "edge_mean": edge_loss,
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
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

    node_loss = _mean(node_error**2)
    edge_loss = _mean(edge_error**2)

    stats = {"node_rmse": node_loss ** (1 / 2), "edge_rmse": edge_loss ** (1 / 2)}

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": _mean(abs_node_error),
                "edge_mean": _mean(abs_edge_error),
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
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

    node_loss = _mean(node_error**2)
    edge_loss = _mean(edge_error**2)

    stats = {"node_rmse": node_loss ** (1 / 2), "edge_rmse": edge_loss ** (1 / 2)}

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": _mean(abs_node_error),
                "edge_mean": _mean(abs_edge_error),
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
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

    n_node_els = _numel(node_error)
    n_edge_els = _numel(edge_error)

    abs_node_error = abs(node_error)
    abs_edge_error = abs(edge_error)

    node_error_above_thresh = node_error[abs_node_error > threshold]
    edge_error_above_thresh = edge_error[abs_edge_error > threshold]

    # We do the sum instead of the mean here so that we reward putting
    # elements below the threshold
    node_loss = (node_error_above_thresh**2).sum()
    edge_loss = (edge_error_above_thresh**2).sum()

    stats = {
        "node_rmse": _mean(node_error**2) ** (1 / 2),
        "edge_rmse": _mean(edge_error**2) ** (1 / 2),
        "node_above_threshold_frac": _numel(node_error_above_thresh) / (n_node_els or 1),
        "edge_above_threshold_frac": _numel(edge_error_above_thresh) / (n_edge_els or 1),
        "node_above_threshold_mean": _mean(abs_node_error[abs_node_error > threshold]),
        "edge_above_threshold_mean": _mean(abs_edge_error[abs_edge_error > threshold]),
    }

    if log_verbose:
        stats.update(
            {
                "node_mean": _mean(abs_node_error),
                "edge_mean": _mean(abs_edge_error),
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
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

    n_node_els = _numel(node_error)
    n_edge_els = _numel(edge_error)

    abs_node_error = abs(node_error)
    abs_edge_error = abs(edge_error)

    node_error_thresholded = _sigmoid_threshold(abs_node_error, threshold, sigmoid_factor)
    edge_error_thresholded = _sigmoid_threshold(abs_edge_error, threshold, sigmoid_factor)

    # We do the sum instead of the mean here so that we reward putting
    # elements below the threshold
    node_loss = (node_error_thresholded**2).sum()
    edge_loss = (edge_error_thresholded**2).sum()

    abs_node_error_above_thresh = abs_node_error[abs_node_error > threshold]
    abs_edge_error_above_thresh = abs_edge_error[abs_edge_error > threshold]

    stats = {
        "node_rmse": _mean(node_error**2) ** (1 / 2),
        "edge_rmse": _mean(edge_error**2) ** (1 / 2),
        "node_above_threshold_frac": _numel(abs_node_error_above_thresh) / (n_node_els or 1),
        "edge_above_threshold_frac": _numel(abs_edge_error_above_thresh) / (n_edge_els or 1),
        "node_above_threshold_mean": _mean(abs_node_error_above_thresh),
        "edge_above_threshold_mean": _mean(abs_edge_error_above_thresh),
    }

    if log_verbose:
        stats.update(
            {
                "node_mean": _mean(abs_node_error),
                "edge_mean": _mean(abs_edge_error),
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
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

    n_node_els = _numel(node_error)
    n_edge_els = _numel(edge_error)

    abs_node_error = abs(node_error)
    abs_edge_error = abs(edge_error)

    node_error_thresholded = _sigmoid_threshold(abs_node_error, threshold, sigmoid_factor)
    edge_error_thresholded = _sigmoid_threshold(abs_edge_error, threshold, sigmoid_factor)

    # We do the sum instead of the mean here so that we reward putting
    # elements below the threshold
    node_loss = node_error_thresholded.sum()
    edge_loss = edge_error_thresholded.sum()

    abs_node_error_above_thresh = abs_node_error[abs_node_error > threshold]
    abs_edge_error_above_thresh = abs_edge_error[abs_edge_error > threshold]

    stats = {
        "node_rmse": _mean(node_error**2) ** (1 / 2),
        "edge_rmse": _mean(edge_error**2) ** (1 / 2),
        "node_above_threshold_frac": _numel(abs_node_error_above_thresh) / (n_node_els or 1),
        "edge_above_threshold_frac": _numel(abs_edge_error_above_thresh) / (n_edge_els or 1),
        "node_above_threshold_mean": _mean(abs_node_error_above_thresh),
        "edge_above_threshold_mean": _mean(abs_edge_error_above_thresh),
    }

    if log_verbose:
        stats.update(
            {
                "node_mean": _mean(abs_node_error),
                "edge_mean": _mean(abs_edge_error),
                "node_std": _std(abs_node_error),
                "edge_std": _std(abs_edge_error),
                "node_max": _max(abs_node_error),
                "edge_max": _max(abs_edge_error),
            }
        )

    return node_loss + edge_loss, stats


def _error_processor(batch, basis_table) -> MatrixDataProcessor:
    """Processor of ``batch`` (or a symmetric density-matrix default) with sub_point_matrix off."""
    metadata = getattr(batch, "metadata", None)
    processor = metadata.get("data_processor") if isinstance(metadata, dict) else None
    if isinstance(processor, (list, tuple)):
        processor = processor[0] if len(processor) > 0 else None
    if processor is None:
        processor = MatrixDataProcessor(
            basis_table=basis_table, symmetric_matrix=True, out_matrix="density_matrix"
        )
    return processor.copy(sub_point_matrix=False)


def _error_to_sisl(data, processor: MatrixDataProcessor, predictions=None):
    # Conversions read the processor from the data's metadata, so replace it there.
    data.metadata = {**data.metadata, "data_processor": processor}
    return processor.matrix_from_data(data, predictions=predictions, out_format=Formats.SISL_DM)


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

    observed_errors = get_predictions_error(nodes_pred, nodes_ref, edges_pred, edges_ref)
    errors = []
    for pred, ref, values in zip((nodes_pred, edges_pred), (nodes_ref, edges_ref), observed_errors):
        finite = torch.isfinite(values).all() if _is_torch_tensor(values) else np.isfinite(values).all()
        if not finite:
            raise ValueError("normalized_density_error requires finite errors for observed reference labels.")
        # Unobserved rows contribute no density error; keep the original layout.
        padded = torch.zeros_like(pred) if _is_torch_tensor(pred) else np.zeros_like(pred)
        padded[_valid_label_rows(ref)] = values
        errors.append(padded)

    # The labels are errors, not density matrices: convert them with the data's own
    # processor (same symmetric convention as training) but without adding back the
    # isolated-atom reference that sub_point_matrix subtracted.
    processor = _error_processor(batch, basis_table)

    matrix_errors: Iterable[Any]
    if isinstance(batch, BasisMatrixDataBase) and _batch_field(batch, "ptr") is None:
        # We haven't really received a batch, but just a single structure.
        # Do as if we received a batch of size 1.
        matrix_error = copy(batch)
        matrix_errors = [_error_to_sisl(
            matrix_error, processor,
            predictions={"node_labels": errors[0], "edge_labels": errors[1]},
        )]
    else:
        # Create an iterator that returns the error for each structure in the batch
        # as a sisl DensityMatrix.
        matrix_errors = (
            _error_to_sisl(example, processor)
            for example in processor.yield_from_batch(
                batch,
                predictions={"node_labels": errors[0], "edge_labels": errors[1]},
            )
        )

    # Initialize counters
    per_config_total_error = 0.
    total_error = 0.
    total_electrons = 0.
    num_configs = 0

    # Loop through structures in the batch
    for matrix_error in matrix_errors:
        if matrix_error.geometry.na == 0:
            continue
        this_config_electrons = matrix_error.geometry.q0
        if not np.isfinite(this_config_electrons) or this_config_electrons <= 0:
            raise ValueError("normalized_density_error requires a positive finite electron count.")

        this_config_error = 0.
        if matrix_error.nnz:
            error_grid = sisl.Grid(grid_spacing, geometry=matrix_error.geometry)
            matrix_error.density(error_grid)
            this_config_error = abs(error_grid).grid.sum() * error_grid.dvolume

        this_config_norm_error = this_config_error / this_config_electrons

        # Update counters
        per_config_total_error += this_config_norm_error
        total_error += this_config_error
        total_electrons += this_config_electrons
        num_configs += 1

    # Compute average errors
    avg_per_config_error = per_config_total_error / (num_configs or 1)
    avg_error = total_error / (total_electrons or 1)

    stats = {
        "avg_per_config_error_percent": avg_per_config_error * 100,
        "avg_error_percent": avg_error * 100,
        "n_structures": num_configs,
        "total_electrons": total_electrons,
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
