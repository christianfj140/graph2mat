"""Explicit weighted objectives for semantic matrix targets, with no mixed-unit mean."""

import math

import torch

from graph2mat.core.data import metrics
from graph2mat.core.data.multitarget import target_data_view
from graph2mat.core.data.physical import validate_physical_weights


def validate_layout_signature(data, layout):
    signature = metrics._batch_field(data, "matrix_target_signature")
    if (not torch.is_tensor(signature) or signature.ndim != 2 or signature.shape[1] != 6
            or signature.shape[0] == 0 or signature.dtype not in (torch.int32, torch.int64)):
        raise ValueError("missing/invalid multitarget layout signature")
    expected = torch.as_tensor(layout.signature, device=signature.device, dtype=signature.dtype)
    if not torch.equal(signature, expected.expand_as(signature)):
        raise ValueError("multitarget semantic layout signature mismatch")


class MultitargetMatrixLoss:
    overlap_prediction_optional = False

    def __init__(self, layout, *, weights, losses):
        if not isinstance(weights, dict) or set(weights) != set(layout.targets):
            raise ValueError("matrix target weights must name exactly the selected targets")
        if not isinstance(losses, dict) or set(losses) != set(layout.targets):
            raise ValueError("matrix target losses must name exactly the selected targets")
        self.layout = layout
        try:
            self.weights = {name: float(value) for name, value in weights.items()}
        except (ValueError, TypeError) as error:
            raise ValueError("matrix target weights must be finite and nonnegative") from error
        if (any(not math.isfinite(v) or v < 0 for v in self.weights.values())
                or not any(self.weights.values())):
            raise ValueError("matrix target weights must be finite/nonnegative and at least one positive")
        self.losses = {}
        self.physical_weights = {}
        self.config = {}
        for name in layout.targets:
            config = losses[name]
            if not isinstance(config, dict) or set(config) - {"name", "kwargs"}:
                raise ValueError(f"{name}: invalid matrix target loss config")
            metric_name = config.get("name")
            if metric_name not in ("elementwise_mse", "gauge_invariant_mse"):
                raise ValueError(f"{name}: unsupported matrix target loss {metric_name!r}")
            if metric_name == "gauge_invariant_mse" and name != "hamiltonian":
                raise ValueError("gauge_invariant_mse requires a hamiltonian target")
            kwargs = dict(config.get("kwargs") or {})
            physical = validate_physical_weights(**{k: kwargs.pop(k, 0.) for k in
                ("electron_weight", "mulliken_weight", "energy_weight")})
            if (physical[0] or physical[1]) and name != "density_matrix":
                raise ValueError(f"{name}: electron/Mulliken weights require density_matrix")
            if any(physical) and name == "overlap":
                raise ValueError("overlap target has no physical loss")
            if kwargs.pop("overlap_weight", 0.) != 0:
                raise ValueError("overlap_weight must be zero: overlap has its own target weight")
            if set(kwargs) - {"edge_weight", "log_verbose"}:
                raise ValueError(f"{name}: unsupported matrix target loss kwargs")
            edge_weight = float(kwargs.get("edge_weight", 1.))
            if not math.isfinite(edge_weight) or edge_weight < 0:
                raise ValueError(f"{name}: edge_weight must be finite and nonnegative")
            if metric_name == "gauge_invariant_mse":
                kwargs["overlap_weight"] = 0.
            self.losses[name] = getattr(metrics, metric_name)(**kwargs)
            self.physical_weights[name] = physical
            self.config[name] = {"name": metric_name, "kwargs": dict(config.get("kwargs") or {})}

    def __call__(self, *, nodes_pred, nodes_ref, edges_pred, edges_ref,
                 batch, basis_table, **kwargs):
        validate_layout_signature(batch, self.layout)
        if (nodes_pred.shape != nodes_ref.shape or edges_pred.shape != edges_ref.shape
                or nodes_pred.ndim != 2 or edges_pred.ndim != 2
                or nodes_pred.shape[1] != self.layout.n_components
                or edges_pred.shape[1] != self.layout.n_components):
            raise ValueError("multitarget prediction/reference shape does not match semantic layout")
        for field, reference in (("matrix_known_nodes", nodes_ref), ("matrix_known_edges", edges_ref)):
            mask = metrics._batch_field(batch, field)
            if not torch.is_tensor(mask) or mask.shape != reference.shape or mask.dtype != torch.bool:
                raise ValueError(f"{field}: observation mask must be boolean with reference shape")
        ns, es, count = metrics._per_element_structure_ids(nodes_ref, edges_ref, batch, basis_table)
        ns, es = ns.to(nodes_pred.device), es.to(edges_pred.device)
        stats, total = {}, nodes_pred.new_zeros(())
        for target, columns in self.layout.slices.items():
            weight = self.weights[target]
            if weight == 0:
                continue
            npred, epred = nodes_pred[:, columns], edges_pred[:, columns]
            nref, eref = nodes_ref[:, columns], edges_ref[:, columns]
            nknown = batch.matrix_known_nodes[:, columns]
            eknown = batch.matrix_known_edges[:, columns]
            metric = self.losses[target]
            edge_weight = metric.metric_kwargs.get("edge_weight", 1.)
            if (not torch.isfinite(nref[nknown]).all() or not torch.isfinite(eref[eknown]).all()
                    or not torch.isfinite(npred[nknown]).all() or not torch.isfinite(epred[eknown]).all()):
                raise ValueError(f"{target}: observed references/predictions must be finite")
            per_structure = nodes_pred.new_zeros(count).index_add(0, ns, nknown.sum(dim=1).to(nodes_pred))
            per_structure = per_structure.index_add(0, es, eknown.sum(dim=1).to(nodes_pred) * edge_weight)
            if (per_structure <= 0).any():
                structure = int((per_structure <= 0).nonzero()[0, 0])
                raise ValueError(f"{target}: no valid observations for structure {structure}")
            denominator = per_structure.sum()
            physical_weights = dict(zip(("electron_weight", "mulliken_weight", "energy_weight"),
                                        self.physical_weights[target]))
            if isinstance(metric, metrics.gauge_invariant_mse):
                if not nknown.all() or not eknown.all():
                    raise ValueError("hamiltonian gauge requires complete reference H/S")
                scol = self.layout.slices["overlap"]
                if not (torch.isfinite(nodes_ref[:, scol]).all() and torch.isfinite(edges_ref[:, scol]).all()):
                    raise ValueError("hamiltonian gauge requires finite reference overlap")
                rn = torch.cat((nref, nodes_ref[:, scol]), dim=1)
                re = torch.cat((eref, edges_ref[:, scol]), dim=1)
                value, extra = metric(nodes_pred=npred, nodes_ref=rn, edges_pred=epred,
                    edges_ref=re, batch=target_data_view(batch, self.layout, target),
                    basis_table=basis_table, **physical_weights)
                matrix_value = extra["gauge_h_rmse"].square()
            else:
                numerator = nodes_pred.new_zeros(())
                for component in range(npred.shape[1]):
                    kn, ke = nknown[:, component], eknown[:, component]
                    elements = kn.sum() + edge_weight * ke.sum()
                    component_loss, _ = metric(nodes_pred=npred[kn, component], nodes_ref=nref[kn, component],
                        edges_pred=epred[ke, component], edges_ref=eref[ke, component])
                    numerator = numerator + component_loss * elements
                value, extra = numerator / denominator, {}
                matrix_value = value
            if any(self.physical_weights[target]) and not isinstance(metric, metrics.gauge_invariant_mse):
                from graph2mat.core.data.physical import physical_loss_terms

                view = target_data_view(batch, self.layout, target)
                physical, physical_stats = physical_loss_terms(npred, epred, batch=view,
                    basis_table=basis_table, **physical_weights)
                value = value + physical
                extra.update(physical_stats)
            prefix = f"target/{target}/"
            stats[prefix + "matrix_sum"] = (matrix_value * denominator).detach()
            stats[prefix + "matrix_count"] = denominator.detach()
            stats[prefix + "loss"] = value.detach()
            stats.update({prefix + key: val.detach() if torch.is_tensor(val) else val for key, val in extra.items()})
            total = total + weight * value
        return total, stats
