"""CPU-prepared, element-weighted matrix MSE without dense predicted blocks."""

import copy
import math

import numpy as np
import torch

from graph2mat.core.data.metrics import (
    _as_numpy_int, _coefficient_readout, _edge_module_for_unique_type,
    _map_types, _symmetric_edge_types_for_labels, _valid_label_rows,
)
from graph2mat.core.data.table import BasisTableWithEdges

COEFFICIENT_TARGETS_ATTR = "_graph2mat_coefficient_mse_targets"


class CoefficientMSEUnsupported(ValueError):
    """This batch/readout needs the ordinary matrix objective instead."""


def _local_factors(operation, mask):
    if operation.change_of_basis.requires_grad:
        raise CoefficientMSEUnsupported("Coefficient-only matrix MSE requires fixed Q, not a trainable change of basis.")
    groups = getattr(operation, "_coefficient_groups", None)
    if groups is None:
        raise CoefficientMSEUnsupported("Matrix MSE requires certified local RTP coefficient groups.")
    if operation.change_of_basis.is_complex():
        raise CoefficientMSEUnsupported("The real matrix MSE certificate does not support complex Q.")
    q = operation.change_of_basis.detach().cpu().double().flatten(1).clone()
    if not torch.isfinite(q).all():
        raise ValueError("The change of basis must be finite.")
    indices = [index for group in groups for index in group]
    if sorted(indices) != list(range(len(q))):
        raise CoefficientMSEUnsupported("RTP groups must cover each coefficient exactly once.")
    mask = torch.as_tensor(mask, dtype=torch.bool).ravel()
    if len(mask) != q.shape[1]:
        raise ValueError("Physical mask and change-of-basis shape disagree.")
    owner = torch.zeros(q.shape[1], dtype=torch.bool)
    factors = []
    for group in groups:
        columns = torch.tensor(group, dtype=torch.long)
        if getattr(operation, "symm_transpose", False):
            index = getattr(operation, "_coefficient_transpose_index", None)
            sign = getattr(operation, "_coefficient_transpose_sign", None)
            if index is None or sign is None:
                raise CoefficientMSEUnsupported("Coefficient-only MSE requires a certified transpose map.")
            index, sign = index.detach().cpu(), sign.detach().cpu().double()
            shaped = q.reshape(*operation.change_of_basis.shape)
            if not torch.allclose(shaped[columns].transpose(1, 2),
                                  shaped[index[columns]] * sign[columns, None, None], rtol=0,
                                  atol=64 * torch.finfo(operation.change_of_basis.dtype).eps):
                raise CoefficientMSEUnsupported("The transpose map does not match the effective Q.")
        local = q[columns]
        support = (local != 0).any(0)
        if (owner & support).any():
            raise CoefficientMSEUnsupported("Local RTP groups have overlapping matrix supports.")
        owner |= support
        observed = support & mask
        if not observed.any():
            continue
        # shortcut: partial angular supports reconstruct; certify masked rank before extending this route.
        if (support & ~mask).any():
            raise CoefficientMSEUnsupported("Padding cuts a local RTP support; use the masked matrix loss.")
        local = local[:, observed]
        if local.shape[0] > local.shape[1]:
            raise CoefficientMSEUnsupported("The observed local RTP is rank deficient.")
        u, r = torch.linalg.qr(local.T, mode="reduced")
        if (r.diagonal().abs() <= 64 * torch.finfo(q.dtype).eps).any():
            raise CoefficientMSEUnsupported("The observed local RTP is rank deficient.")
        factors.append({"columns": columns, "elements": observed[mask].nonzero().ravel(),
                        "u": u, "factor": r.T.contiguous()})
    return factors, (~owner & mask).nonzero().ravel(), len(q)


class CoefficientMatrixMSEProjector:
    """Immutable CPU snapshot for a fixed readout Q; rebuild it if Q changes.

    The object contains no model/module/parameter and can be sent to collate
    workers. QR uses the effective Q in float64, not assumed orthonormality.
    Whole missing blocks are supported; partial NaN labels request fallback.
    """

    def __init__(self, readout):
        first = _coefficient_readout(readout)
        if isinstance(readout, torch.nn.ModuleList):
            for other in readout[1:]:
                pairs = list(zip(first.self_interactions, other.self_interactions))
                pairs.extend((operation, other.interactions[key]) for key, operation in first.interactions.items())
                for operation, peer in pairs:
                    if operation is None:
                        continue
                    if peer.change_of_basis.requires_grad:
                        raise CoefficientMSEUnsupported("Coefficient-only matrix MSE requires fixed Q in every readout.")
                    if getattr(peer, "symm_transpose", False):
                        for name in ("_coefficient_transpose_index", "_coefficient_transpose_sign"):
                            expected, actual = getattr(operation, name, None), getattr(peer, name, None)
                            if expected is None or actual is None or not torch.equal(
                                expected.detach().cpu(), actual.detach().cpu(),
                            ):
                                raise CoefficientMSEUnsupported("Every readout requires the same certified transpose map.")
                    if not torch.equal(
                        operation.change_of_basis.detach().cpu(), peer.change_of_basis.detach().cpu(),
                    ):
                        raise ValueError("Readout coefficient coordinates require identical change of basis.")
        self.basis_table = BasisTableWithEdges(copy.deepcopy(first.basis_table.basis))
        self.symmetric = first.symmetric
        self.n_components = first.n_matrix_components
        self.node_specs, self.edge_specs = {}, {}
        node_map = _map_types(first.types_to_graph2mat, np.arange(len(first.basis_table.basis)))
        basis_filters = getattr(first, "basis_filters", None)
        for point_type, graph_type in enumerate(node_map):
            operation = first.self_interactions[int(graph_type)]
            if operation is None:
                self.node_specs[point_type] = None
                continue
            mask = (np.ones(operation.change_of_basis.shape[1:], dtype=bool) if basis_filters is None else
                    np.outer(basis_filters[point_type], basis_filters[point_type]))
            self.node_specs[point_type] = self._spec(operation, mask, f"node:{graph_type}", False)
        edge_types = range(len(first.basis_table.edge_type_to_point_types))
        # Reverse edges carry -t: between different species, and between atoms of the same
        # species too in non-symmetric matrices (both directions are stored).
        reverse_types = [-t for t, (i, j) in enumerate(first.basis_table.edge_type_to_point_types)
                         if t != 0 and (i != j or not self.symmetric)]
        for edge_type in [*edge_types, *reverse_types]:
            key, operation = _edge_module_for_unique_type(first, edge_type)
            if operation is None:
                self.edge_specs[edge_type] = None
                continue
            graph_type = int(_map_types(first.edge_types_to_graph2mat, [edge_type])[0])
            transpose = self.symmetric and graph_type < 0
            if basis_filters is None:
                mask = np.ones(operation.change_of_basis.shape[1:], dtype=bool)
            else:
                i, j = first.basis_table.edge_type_to_point_types[abs(edge_type)]
                if edge_type < 0:
                    i, j = j, i
                mask = np.outer(basis_filters[i], basis_filters[j])
                if transpose:
                    mask = mask.T
            self.edge_specs[edge_type] = self._spec(operation, mask, f"edge:{key}", transpose)

    @staticmethod
    def _spec(operation, mask, key, transpose):
        factors, uncovered, size = _local_factors(operation, mask)
        # Uncovered entries can only contribute target residual, never prediction.
        mask = torch.as_tensor(mask, dtype=torch.bool).ravel()
        uncovered = torch.searchsorted(mask.nonzero().ravel(), uncovered)
        return {"key": key, "size": size, "factors": factors,
                "uncovered": uncovered, "transpose": transpose}

    @torch.no_grad()
    def __call__(self, batch, nodes_ref, edges_ref):
        for reference in (nodes_ref, edges_ref):
            if not torch.is_tensor(reference) or reference.device.type != "cpu":
                raise ValueError("Coefficient targets must be prepared from CPU tensors.")
            if reference.is_complex():
                raise CoefficientMSEUnsupported("The real matrix MSE certificate does not support complex labels.")
            if reference.ndim not in (1, 2) or (1 if reference.ndim == 1 else reference.shape[1]) != self.n_components:
                raise ValueError(f"Labels must have {self.n_components} components with shape (rows[, components]).")
            if torch.isinf(reference).any():
                raise ValueError("Observed coefficient targets must be finite or NaN.")
        point_types = _as_numpy_int(batch.point_types)
        edge_types = _symmetric_edge_types_for_labels(
            readout=self, batch=batch, basis_table=self.basis_table, expected_nlabels=len(edges_ref),
        )
        result = {"node": {}, "edge": {}}
        for kind, types, reference, specs, shapes in (
            ("node", point_types, nodes_ref, self.node_specs, self.basis_table.point_block_shape),
            ("edge", edge_types, edges_ref, self.edge_specs, self.basis_table.edge_block_shape),
        ):
            lengths = np.prod(shapes[:, np.abs(types)], axis=0)
            if int(lengths.sum()) != len(reference):
                raise ValueError(f"{kind} label length does not match physical block rows.")
            # Count missing rows once; per-block Tensor reductions dominate CPU collate.
            missing = np.concatenate(([0], np.cumsum((~_valid_label_rows(reference)).numpy())))
            buckets, offsets = {}, {}
            start = 0
            for physical_type, length in zip(types, lengths):
                end = start + int(length)
                spec = specs[int(physical_type)]
                if spec is not None:
                    key = spec["key"]
                    offset = offsets.get(key, 0)
                    offsets[key] = offset + 1
                    shape = tuple(shapes[:, abs(physical_type)])
                    if kind == "edge" and physical_type < 0:
                        shape = shape[::-1]
                    if spec["transpose"]:
                        block = reference[start:end].reshape(*shape, self.n_components).transpose(0, 1)
                    else:
                        block = reference[start:end]
                    block = block.reshape(-1, self.n_components)
                    n_missing = int(missing[end] - missing[start])
                    if 0 < n_missing < length:
                        raise CoefficientMSEUnsupported("partial NaN blocks require the matrix loss.")
                    if key not in result[kind]:
                        result[kind][key] = {"shape": None, "terms": [], "count": 0,
                                            "residual": torch.zeros((), dtype=torch.float64)}
                    entry = result[kind][key]
                    entry["shape"] = (offset + 1, self.n_components * spec["size"])
                    if n_missing == 0 and length:
                        indices, blocks = buckets.setdefault(int(physical_type), ([], []))
                        indices.append(offset)
                        blocks.append(block)
                        entry["count"] += block.numel()
                start = end
            for physical_type, (rows, blocks) in buckets.items():
                spec = specs[physical_type]
                entry = result[kind][spec["key"]]
                target = torch.stack(blocks).double().transpose(1, 2)
                rows = torch.tensor(rows, dtype=torch.long)
                for local in spec["factors"]:
                    observed = target[:, :, local["elements"]]
                    projected = observed @ local["u"]
                    entry["residual"] += (observed - projected @ local["u"].T).square().sum()
                    entry["terms"].append({"rows": rows, "columns": local["columns"],
                                           "factor": local["factor"], "target": projected})
                entry["residual"] += target[:, :, spec["uncovered"]].square().sum()
        return result


def coefficient_matrix_mse(predictions, targets, edge_weight=1.0):
    """Equivalent to elementwise_mse for the prepared physical observations.

    ``edge_weight`` multiplies both edge SSE and physical element count. This
    is not coefficient_space_mse's equal mean across readout operations.
    """
    if not math.isfinite(edge_weight) or edge_weight < 0:
        raise ValueError("edge_weight must be finite and nonnegative.")
    values = [value for kind in ("node", "edge") for value in predictions[kind].values()]
    template = values[0] if values else torch.empty((), dtype=torch.float64)
    zero = template.new_zeros(())
    for value in values:
        zero = zero + value.reshape(-1)[:0].sum()
    stats = {}
    for kind in ("node", "edge"):
        sse, count = zero, 0
        for key, entry in targets[kind].items():
            if key not in predictions[kind]:
                raise ValueError(f"Missing predicted {kind} coefficients for {key}.")
            value = predictions[kind][key]
            if tuple(value.shape) != entry["shape"]:
                raise ValueError(f"Coefficient shape for {key}: got {tuple(value.shape)}, expected {entry['shape']}.")
            size = entry["shape"][1]
            for term in entry["terms"]:
                n_components = term["target"].shape[1]
                coefficients = value.reshape(len(value), n_components, size // n_components)
                local = coefficients[term["rows"].to(value.device)][:, :, term["columns"].to(value.device)]
                # Outer Lightning AMP must not round an otherwise float32 matrix loss.
                with torch.autocast(device_type=value.device.type, enabled=False):
                    error = local @ term["factor"].to(value) - term["target"].to(value)
                sse = sse + error.square().sum()
            sse = sse + entry["residual"].to(value)
            count += entry["count"]
        for key, value in predictions[kind].items():
            if key not in targets[kind] and len(value):
                raise ValueError(f"Unused predicted {kind} coefficients for {key}.")
        stats[f"{kind}_sse"] = sse
        stats[f"{kind}_count"] = count
        stats[f"{kind}_rmse"] = (sse / (count or 1)).sqrt()
    count = stats["node_count"] + edge_weight * stats["edge_count"]
    return (stats["node_sse"] + edge_weight * stats["edge_sse"]) / (count or 1), stats
