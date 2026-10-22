"""Wrapping of raw models to use them in pytorch_lightning."""

import importlib
import importlib.metadata
import math
import platform
import subprocess
import warnings
import zipfile
from pathlib import Path
from typing import Any, Optional, Type, Union

import pytorch_lightning as pl
import torch
from e3nn import o3
from torchmetrics import Metric

from graph2mat import AtomicTableWithEdges, BasisTableWithEdges, __version__
from graph2mat.bindings.torch.load import sanitize_checkpoint
from graph2mat.bindings.torch.data.coefficient_targets import (
    COEFFICIENT_TARGETS_ATTR, CoefficientMatrixMSEProjector, CoefficientMSEUnsupported, coefficient_matrix_mse,
)
from graph2mat.bindings.torch.modules.graph2mat import layout_to
from graph2mat.core.modules.graph2mat import LAYOUT_ATTR

# from context import mace
from graph2mat.core.data import metrics as metrics_module
from graph2mat.core.data.metrics import OrbitalMatrixMetric, elementwise_mse
from graph2mat.core.data.node_feats import NodeFeature
from graph2mat.core.data.physical import validate_physical_weights
from graph2mat.bindings.torch.multitarget import MultitargetMatrixLoss

from ._helpers import glob, maybe_zip_path


def _training_provenance():
    """Record the installed stack and tracked source checkout, when available."""
    versions = {}
    for name in ("torch", "torch-geometric", "pytorch-lightning", "e3nn", "mace-torch", "sisl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None

    commit = dirty = None
    source = Path(__file__).resolve()
    git = ["git", "-C", str(source.parent)]
    try:
        # A wheel inside another repository must not inherit that repository's HEAD.
        subprocess.run(git + ["ls-files", "--error-unmatch", source.name],
                       check=True, capture_output=True, text=True, timeout=2)
        commit = subprocess.run(git + ["rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True, timeout=2).stdout.strip()
        dirty = bool(subprocess.run(git + ["status", "--porcelain", "--untracked-files=no"],
                                    check=True, capture_output=True, text=True,
                                    timeout=2).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit = dirty = None
    return {"python": platform.python_version(), "versions": versions,
            "git_commit": commit, "git_dirty": dirty}


def _epoch_reduce(stat_name: str) -> str:
    """Epoch aggregation of a per-batch stat: the maximum of maxima, else the mean."""
    return "max" if stat_name.endswith("_max") else "mean"


class _EpochRMSE(Metric):
    """Accumulate squared errors; take the root only after summing all batches."""

    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("squared_error", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, rmse, count):
        rmse = torch.as_tensor(rmse, device=self.device).detach().double()
        count = torch.as_tensor(count, device=self.device).double()
        self.squared_error += torch.where(count > 0, rmse.square() * count, 0.0)
        self.count += count

    def compute(self):
        return torch.where(self.count > 0, self.squared_error / self.count, 0.0).sqrt()


class _EpochErrorMoments(Metric):
    """Global absolute-error mean, sample standard deviation, maximum or sum."""

    full_state_update = False

    def __init__(self, statistic):
        super().__init__()
        self.statistic = statistic
        for name in ("total", "squared", "count"):
            self.add_state(name, torch.tensor(0., dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("maximum", torch.tensor(0., dtype=torch.float64), dist_reduce_fx="max")

    def update(self, errors):
        values = errors.detach().double().abs()
        self.total += values.sum()
        self.squared += values.square().sum()
        self.count += values.numel()
        if values.numel():
            self.maximum = torch.maximum(self.maximum, values.max())

    def compute(self):
        if self.statistic == "max":
            return self.maximum
        if self.statistic == "sum":
            return self.total
        mean = self.total / self.count.clamp_min(1)
        if self.statistic == "mean":
            return mean
        variance = (self.squared - self.total * mean) / (self.count - 1).clamp_min(1)
        return torch.where(self.count > 1, variance.clamp_min(0), 0.).sqrt()


class _EpochMatrixLoss(Metric):
    """Reduce matrix branches and equal-weight coefficient operations globally."""

    full_state_update = False

    def __init__(self, coefficient_keys=(), statistic=None, normalize_weights=False):
        super().__init__()
        self.add_state("totals", torch.zeros(2, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("counts", torch.zeros(2, dtype=torch.float64), dist_reduce_fx="sum")
        self.joint = False
        self.weights = (1., 1.)
        self.coefficient_keys = tuple(coefficient_keys)
        self.statistic = statistic
        self.normalize_weights = normalize_weights
        self.coefficient_weight = 0.
        for name in ("coefficient_totals", "coefficient_counts"):
            self.add_state(name, torch.zeros(len(coefficient_keys), dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, node_error, edge_error, power, joint, node_weight, edge_weight,
               coefficient_errors=None, coefficient_power=2, coefficient_weight=0., matrix_stats=None):
        self.joint, self.weights = joint, (node_weight, edge_weight)
        if matrix_stats is None:
            errors = (node_error.detach().double(), edge_error.detach().double())
            self.totals += torch.stack([error.abs().pow(power).sum() for error in errors])
            self.counts += self.counts.new_tensor([error.numel() for error in errors])
        else:
            self.totals += torch.stack([matrix_stats[f"{kind}_sse"].detach().to(self.totals) for kind in ("node", "edge")])
            self.counts += self.counts.new_tensor([matrix_stats[f"{kind}_count"] for kind in ("node", "edge")])
        self.coefficient_weight = coefficient_weight
        if coefficient_errors is not None:
            for i, key in enumerate(self.coefficient_keys):
                error = coefficient_errors.get(key)
                if error is not None:
                    self.coefficient_totals[i] += error.detach().double().abs().pow(coefficient_power).sum()
                    self.coefficient_counts[i] += error.numel()

    def compute(self):
        observed = self.coefficient_counts.count_nonzero()
        coefficient = (self.coefficient_totals / self.coefficient_counts.clamp_min(1)).sum() / observed.clamp_min(1)
        if self.statistic == "operations":
            return observed.double()
        if self.statistic == "coefficients":
            return coefficient
        weights = self.totals.new_tensor(self.weights)
        if self.statistic == "sum":
            label = (weights * self.totals).sum()
        elif self.joint:
            denominator = (weights * self.counts).sum()
            label = (weights * self.totals).sum() / torch.where(denominator != 0, denominator, 1.)
        else:
            label = (weights * self.totals / self.counts.clamp_min(1)).sum()
        if self.normalize_weights:
            denominator = (weights * (self.counts > 0)).sum()
            label = label / torch.where(denominator != 0, denominator, 1.)
        return label + self.coefficient_weight * coefficient


class _EpochWeightedLoss(Metric):
    """Combine physical terms only after reducing each term's own denominator."""

    full_state_update = False

    def __init__(self, terms, std_terms=0):
        super().__init__()
        self.add_state("totals", torch.zeros(terms, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("counts", torch.zeros(terms, dtype=torch.float64), dist_reduce_fx="sum")
        self.weights = (1.,) * terms
        self.std_terms = std_terms
        if std_terms:
            self.add_state("squared_totals", torch.zeros(std_terms, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, means, counts, weights, stds=None):
        means = torch.as_tensor(means, dtype=torch.float64, device=self.device).detach()
        counts = torch.as_tensor(counts, dtype=torch.float64, device=self.device)
        self.totals += torch.where(counts > 0, means * counts, 0.)
        self.counts += counts
        self.weights = weights
        if self.std_terms:
            stds = torch.as_tensor(stds, dtype=torch.float64, device=self.device).detach()
            self.squared_totals += (stds.square() * (counts[:self.std_terms] - 1).clamp_min(0)
                                    + means[:self.std_terms].square() * counts[:self.std_terms])

    def compute(self):
        denominator = torch.where(self.counts > 0, self.counts, 1.)
        means = self.totals / denominator
        value = (means * self.totals.new_tensor(self.weights)).sum()
        if self.std_terms:
            counts = self.counts[:self.std_terms]
            variance = (self.squared_totals - self.totals[:self.std_terms] * means[:self.std_terms]) / (counts - 1).clamp_min(1)
            value = value + torch.where(counts > 1, variance.clamp_min(0), 0.).sqrt().sum()
        return value


class LitBasisMatrixModel(pl.LightningModule):
    """Base class to wrap a matrix model to use it in pytorch_lightning."""

    basis_table: BasisTableWithEdges
    model: torch.nn.Module
    model_kwargs: dict

    def __init__(
        self,
        model_cls: Type[torch.nn.Module],
        root_dir: str = ".",
        basis_files: Union[str, None] = None,
        basis_table: Union[BasisTableWithEdges, None] = None,
        no_basis: Optional[dict] = None,
        loss: Type[OrbitalMatrixMetric] = elementwise_mse,
        loss_kwargs: Optional[dict[str, Any]] = None,
        initial_node_feats: str = "OneHotZ",
        training_stages: Optional[list[dict[str, Any]]] = None,
        coefficient_training: bool = False,
        selection_metric: Optional[str] = None,
        **kwargs,
    ):
        """``selection_metric`` (roadmap 1.15) names an epoch metric, e.g. the loss-independent
        ``val_matrix_rmse``, used as ``hp_metric`` and by the CLI's default best checkpoint
        instead of ``val_loss``. ``None`` keeps the loss-dependent selection."""
        super().__init__()

        self.save_hyperparameters()
        self.selection_metric = None if selection_metric == "val_loss" else selection_metric

        root_dir = maybe_zip_path(root_dir)

        if basis_table is None:
            if basis_files is None:
                self.basis_table = None
            else:
                self.basis_table = AtomicTableWithEdges.from_basis_glob(
                    glob(root_dir, basis_files), no_basis_atoms=no_basis
                )
        else:
            self.basis_table = basis_table

        self.initial_node_feats = [
            NodeFeature.registry[k] for k in initial_node_feats.split(" ")
        ]
        self.initial_node_feats_irreps = sum(
            [f.get_e3nn_irreps(self.basis_table) for f in self.initial_node_feats],
            o3.Irreps(),
        ).simplify()

        self.default_loss_fn = loss(**(loss_kwargs or {}))
        self.loss_fn = self.default_loss_fn
        self.training_stages_config = training_stages or []
        self._validated_training_stages = []
        self._active_training_stage_index = None
        self._active_training_stage_name = None
        self._epoch_rmse = torch.nn.ModuleDict()
        self._epoch_errors = torch.nn.ModuleDict()
        self._epoch_losses = torch.nn.ModuleDict()
        self.coefficient_training = coefficient_training
        self._coefficient_training_unavailable = None
        self._coefficient_fallback_reasons = set()

        self.model_cls = model_cls
        self.model = None  # Subclasses are responsible for initializing the model by calling init_model.

    def init_model(self, **kwargs):
        """Initializes the model, storing the arguments used."""
        self.model_kwargs = kwargs
        self.model = self.model_cls(**self.model_kwargs)
        self._validated_training_stages = self._validate_training_stages(
            self.training_stages_config
        )
        return self.model

    def forward(self, x):
        return self.model(x)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Also moves the batch's type layout (roadmap 6.3), which lives outside the PyG
        store and so is not moved by ``batch.to(device)``."""
        layout = getattr(batch, LAYOUT_ATTR, None)
        targets = getattr(batch, COEFFICIENT_TARGETS_ATTR, None)
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        if layout is not None:
            object.__setattr__(batch, LAYOUT_ATTR, layout_to(layout, device))
        if targets is not None:
            object.__setattr__(batch, COEFFICIENT_TARGETS_ATTR, layout_to(targets, device))
        return batch

    @staticmethod
    def _validate_pred_ref_shapes(out, batch, overlap_optional: bool = False):
        """``overlap_optional``: the loss uses the last label component (S) only as a label
        and accepts predictions with one component fewer (gauge_invariant_mse with
        n_output_components)."""
        pairs = (
            ("node_labels", "point_labels"),
            ("edge_labels", "edge_labels"),
        )

        for pred_key, ref_key in pairs:
            pred = out[pred_key]
            ref = batch[ref_key]
            n_pred = 1 if pred.ndim == 1 else pred.shape[1]
            if (overlap_optional and ref.ndim == 2 and pred.shape[0] == ref.shape[0]
                    and n_pred == ref.shape[1] - 1):
                continue

            if pred.ndim != ref.ndim:
                raise ValueError(
                    f"Shape mismatch for {pred_key}/{ref_key}: "
                    f"pred.ndim={pred.ndim}, ref.ndim={ref.ndim}, "
                    f"pred.shape={tuple(pred.shape)}, ref.shape={tuple(ref.shape)}. "
                    "This usually means n_matrix_components is misconfigured."
                )

            if pred.shape != ref.shape:
                raise ValueError(
                    f"Shape mismatch for {pred_key}/{ref_key}: "
                    f"pred.shape={tuple(pred.shape)}, ref.shape={tuple(ref.shape)}. "
                    "Ensure data.n_matrix_components equals model.n_matrix_components."
                )

    def _warn_coefficient_fallback(self, reason):
        if reason not in self._coefficient_fallback_reasons:
            warnings.warn(f"coefficient_training uses matrix fallback: {reason}", stacklevel=2)
            self._coefficient_fallback_reasons.add(reason)

    def _coefficient_training_reason(self, batch=None):
        from graph2mat.models.mace import MatrixMACE
        from .callbacks import MatrixWriter, SamplewiseMetricsLogger

        if any(self._physical_weights(self.loss_fn)):
            return "physical objectives require matrix predictions"
        if self._coefficient_training_unavailable is not None:
            return self._coefficient_training_unavailable
        if not isinstance(self.model, MatrixMACE):
            return "only MatrixMACE is certified"
        if type(self.loss_fn) is not metrics_module.elementwise_mse:
            return "the objective requires matrix predictions"
        if self.loss_fn.metric_kwargs.get("log_verbose", False):
            return "verbose matrix statistics are requested"
        edge_weight = self.loss_fn.metric_kwargs.get("edge_weight", 1.)
        if not math.isfinite(edge_weight) or edge_weight < 0:
            return "the certificate requires finite nonnegative edge_weight"
        if self.model.gradient_checkpointing or self.model.charge_readout is not None:
            return "gradient_checkpointing/long_range require matrix predictions"
        dtype = next(self.model.parameters()).dtype
        if dtype not in (torch.float32, torch.float64):
            return "only real float32/float64 training is certified"
        if torch.is_autocast_enabled(self.device.type) and not self.model.readout_fp32:
            return "autocast requires readout_fp32 for the certified objective"
        trainer = self._trainer
        for owner, parent in ((self, LitBasisMatrixModel),
                              (getattr(trainer, "datamodule", None), pl.LightningDataModule)):
            if owner is not None:
                for hook in ("on_before_batch_transfer", "on_after_batch_transfer", "transfer_batch_to_device"):
                    if getattr(type(owner), hook) is not getattr(parent, hook):
                        return f"custom {hook} can invalidate prepared targets"
        for hook in ("on_train_batch_start", "on_train_batch_end"):
            if getattr(type(self), hook) is not getattr(pl.LightningModule, hook):
                return f"custom {hook} requires the matrix contract"
        for hook in ("_log_loss", "_log_stats"):
            if getattr(type(self), hook) is not getattr(LitBasisMatrixModel, hook):
                return f"custom {hook} requires the matrix logging contract"
        for callback in getattr(trainer, "callbacks", ()):
            if type(callback) in (MatrixWriter, SamplewiseMetricsLogger):
                if "train" in callback.splits:
                    return "a training callback requires matrix predictions"
            elif not type(callback).__module__.startswith("pytorch_lightning.callbacks."):
                for hook in ("on_train_batch_start", "on_train_batch_end"):
                    if getattr(type(callback), hook) is not getattr(pl.Callback, hook):
                        return f"custom callback {hook} requires the matrix contract"
        if batch is not None:
            targets = getattr(batch, COEFFICIENT_TARGETS_ATTR, None)
            if not isinstance(targets, dict):
                return targets if isinstance(targets, str) else "CPU-prepared targets are unavailable"
        return None

    def _can_train_coefficients(self, batch):
        if not self.coefficient_training:
            return False
        reason = self._coefficient_training_reason(batch)
        if reason is not None:
            self._warn_coefficient_fallback(reason)
            return False
        return True

    def training_step(self, batch, batch_idx):
        datamodule = getattr(self._trainer, "datamodule", None)
        if (getattr(datamodule, "max_batch_cost", None) is not None
                and isinstance(self.loss_fn, metrics_module.pair_difference_mse)):
            raise ValueError("max_batch_cost is incompatible with pair_difference_mse: "
                             "partners must remain in the same batch.")
        coefficient_only = self._can_train_coefficients(batch)
        matrix_stats = None
        if coefficient_only:
            out = self.model(batch, coefficients_only=True)
            loss, matrix_stats = coefficient_matrix_mse(
                {"node": out["node_coefficients"], "edge": out["edge_coefficients"]},
                getattr(batch, COEFFICIENT_TARGETS_ATTR), self.loss_fn.metric_kwargs.get("edge_weight", 1.),
            )
            loss = loss.to(self.device)
            if not loss.requires_grad:
                loss = loss + next(p for p in self.model.parameters() if p.requires_grad).reshape(-1)[:0].sum()
            stats = {key: matrix_stats[key].to(self.device) for key in ("node_rmse", "edge_rmse")}
        else:
            out = self.model(batch)
            self._validate_pred_ref_shapes(
                out, batch, getattr(self.loss_fn, "overlap_prediction_optional", False)
            )

            loss, stats = self.loss_fn(
                nodes_pred=out["node_labels"],
                nodes_ref=batch["point_labels"],
                edges_pred=out["edge_labels"],
                edges_ref=batch["edge_labels"],
                batch=batch,
                basis_table=self.basis_table,
                out=out,
                model=self.model,
            )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # A rank can lack species/pairs; MACE energy heads are not in the loss.
            # DDP needs zero gradients for those parameters, not disconnected graphs.
            # ponytail: one zero dependency per tensor in DDP; evaluate empty blocks
            # instead if the extra zero-gradient materialization becomes costly.
            loss = loss + sum(parameter.reshape(-1)[:1].sum() * 0
                              for parameter in self.model.parameters() if parameter.requires_grad)

        logging_kwargs = {"matrix_stats": matrix_stats} if coefficient_only else {}
        self._log_loss("train", loss, out, batch, stats, **logging_kwargs)
        self._log_stats("train", stats, batch, out, **logging_kwargs)

        # Only what callbacks read: keeping every MACE output alive until the end of
        # the batch only costs memory.
        return {"loss": loss} if coefficient_only else {
            "loss": loss, "node_labels": out["node_labels"], "edge_labels": out["edge_labels"],
            **({"matrix_predictions": out["matrix_predictions"]} if "matrix_predictions" in out else {})}

    def _rmse_count(self, name, batch, stats=None):
        """Number of terms used by the standard matrix RMSE statistics."""
        if name in ("gauge_h_rmse", "overlap_rmse"):
            edge_weight = self.loss_fn.metric_kwargs.get("edge_weight", 1.0)
            count = (metrics_module._valid_label_rows(batch.point_labels).sum()
                     + edge_weight * metrics_module._valid_label_rows(batch.edge_labels).sum())
            return count * (batch.point_labels.shape[1] - 1) if name == "gauge_h_rmse" else count
        if name == "spectral_rmse" and stats is not None:
            return stats["spectral_elements"]
        if name == "pair_rmse" and stats is not None:
            return stats["n_pairs"]
        for kind, labels in (("node", batch.point_labels), ("edge", batch.edge_labels)):
            if name != f"{kind}_rmse" and not name.startswith(f"{kind}_rmse_spin"):
                continue
            if isinstance(self.loss_fn, (metrics_module.block_type_mse_per_structure,
                                         metrics_module.block_type_mse_per_domain)):
                node_ids, edge_ids, _ = metrics_module._per_element_structure_ids(
                    batch.point_labels, batch.edge_labels, batch, self.basis_table,
                )
                ids = (node_ids if kind == "node" else edge_ids).to(labels.device)
                ids = ids[metrics_module._valid_label_rows(labels)]
                return torch.bincount(ids, minlength=batch.num_graphs).count_nonzero()
            count = metrics_module._valid_label_rows(labels).sum()
            if name == f"{kind}_rmse" and labels.ndim == 2:
                count *= labels.shape[1]
            return count
        return None

    def _log_loss(self, split, loss, out, batch, stats=None, matrix_stats=None):
        if isinstance(self.loss_fn, MultitargetMatrixLoss):
            self._log_multitarget_loss(split, batch, stats)
            return
        policies = {
            metrics_module.block_type_mse: (2, False, 1., 1.),
            metrics_module.elementwise_mse: (2, True, 1., self.loss_fn.metric_kwargs.get("edge_weight", 1.)),
            metrics_module.node_mse: (2, False, 1., 0.),
            metrics_module.edge_mse: (2, False, 0., 1.),
            metrics_module.block_type_mae: (1, False, 1., 1.),
            metrics_module.elementwise_mae: (1, True, 1., 1.),
        } if hasattr(self.loss_fn, "metric_kwargs") else {}
        policy = policies.get(type(self.loss_fn))
        statistic, normalize_weights = None, False
        threshold = type(self.loss_fn) in (metrics_module.block_type_mse_threshold,
                                          metrics_module.block_type_mse_sigmoid_thresh,
                                          metrics_module.block_type_mae_sigmoid_thresh)
        domain = type(self.loss_fn) is metrics_module.block_type_mse_per_domain
        if threshold:
            power = 1 if type(self.loss_fn) is metrics_module.block_type_mae_sigmoid_thresh else 2
            policy, statistic = (power, False, 1., 1.), "sum"
        if domain:
            kwargs = self.loss_fn.metric_kwargs
            policy = (1, False, kwargs.get("small_domain_weight", .5), kwargs.get("large_domain_weight", .5))
            normalize_weights = True
        huber = type(self.loss_fn) in (metrics_module.block_type_huber, metrics_module.block_type_smooth_l1)
        if huber:
            policy = (1, False, 1., 1.)
        physical_huber = isinstance(self.loss_fn, (metrics_module.block_normalized_huber,
                                                   metrics_module.hamiltonian_composite_loss))
        if physical_huber:
            policy = (1, False, self.loss_fn.node_weight, self.loss_fn.edge_weight)
        coefficient = type(self.loss_fn) in (metrics_module.coefficient_space_mse, metrics_module.coefficient_space_mae)
        coefficient_keys, coefficient_errors = (), None
        coefficient_power = 1 if type(self.loss_fn) is metrics_module.coefficient_space_mae else 2
        coefficient_weight = getattr(self.loss_fn, "coefficient_mse_weight", 0.)
        if coefficient or coefficient_weight:
            coefficient_keys, coefficient_errors = self._coefficient_epoch_terms(out, batch)
        if coefficient:
            policy, coefficient_weight = (2, False, 0., 0.), 1.
        key = f"{split}_loss"
        value, attribute = loss, None
        relative_policies = {metrics_module.block_type_mape: (1, 0., 0),
                             metrics_module.block_type_mapemaemix: (1, 100., 0),
                             metrics_module.block_type_mapemsemix: (2, 10000., 0),
                             metrics_module.block_type_mapestdmix: (1, 0., 2)}
        if any(self._physical_weights(self.loss_fn)):
            value = self._physical_epoch_loss(key, batch, stats)
            attribute = f"_epoch_losses.{key}"
        elif type(self.loss_fn) is metrics_module.normalized_density_error:
            if (key not in self._epoch_losses or not isinstance(self._epoch_losses[key], _EpochWeightedLoss)
                    or len(self._epoch_losses[key].totals) != 1):
                self._epoch_losses[key] = _EpochWeightedLoss(1).to(self.device)
            value = self._epoch_losses[key]
            value([loss], [stats["n_structures"]], (1.,))
            attribute = f"_epoch_losses.{key}"
        elif type(self.loss_fn) in relative_policies:
            power, label_weight, std_terms = relative_policies[type(self.loss_fn)]
            if (key not in self._epoch_losses or not isinstance(self._epoch_losses[key], _EpochWeightedLoss)
                    or len(self._epoch_losses[key].totals) != 4 or self._epoch_losses[key].std_terms != std_terms):
                self._epoch_losses[key] = _EpochWeightedLoss(4, std_terms).to(self.device)
            value = self._epoch_losses[key]
            errors = tuple(error.detach().double() for error in metrics_module.get_predictions_error(
                out["node_labels"], batch.point_labels, out["edge_labels"], batch.edge_labels,
            ))
            relative = tuple(metrics_module._relative_error(error, ref) for error, ref in
                             zip(errors, (batch.point_labels, batch.edge_labels)))
            means = torch.stack([metrics_module._mean(error) for error in relative]
                                + [metrics_module._mean(error.abs().pow(power)) for error in errors])
            counts = means.new_tensor([error.numel() for error in (*relative, *errors)])
            stds = torch.stack([metrics_module._std(error) for error in relative]) if std_terms else None
            value(means, counts, (1., 1., label_weight, label_weight), stds)
            attribute = f"_epoch_losses.{key}"
        elif type(self.loss_fn) is metrics_module.pair_difference_mse:
            kwargs = self.loss_fn.metric_kwargs
            if (key not in self._epoch_losses or not isinstance(self._epoch_losses[key], _EpochWeightedLoss)
                    or len(self._epoch_losses[key].totals) != 2):
                self._epoch_losses[key] = _EpochWeightedLoss(2).to(self.device)
            value = self._epoch_losses[key]
            node_count = self._rmse_count("node_rmse", batch).double()
            edge_count = self._rmse_count("edge_rmse", batch).double() * kwargs.get("edge_weight", 1.)
            count = node_count + edge_count
            base = (stats["node_rmse"].double().square() * node_count + stats["edge_rmse"].double().square() * edge_count
                    ) / torch.where(count > 0, count, 1.)
            pair = torch.as_tensor(stats.get("pair_rmse", 0.), device=self.device).square()
            pairs = torch.as_tensor(stats.get("n_pairs", 0), device=self.device)
            value(torch.stack([base, pair]), torch.stack([count, pairs]), (1., kwargs.get("pair_weight", 1.)))
            attribute = f"_epoch_losses.{key}"
        elif type(self.loss_fn) is metrics_module.gauge_invariant_mse:
            kwargs = self.loss_fn.metric_kwargs
            if (key not in self._epoch_losses or not isinstance(self._epoch_losses[key], _EpochWeightedLoss)
                    or len(self._epoch_losses[key].totals) != 3):
                self._epoch_losses[key] = _EpochWeightedLoss(3).to(self.device)
            value = self._epoch_losses[key]
            means = torch.stack([stats["gauge_h_rmse"].square(), stats["overlap_rmse"].square(),
                                 torch.as_tensor(stats.get("spectral_rmse", 0.), device=self.device).square()])
            counts = torch.stack([self._rmse_count("gauge_h_rmse", batch),
                                  self._rmse_count("overlap_rmse", batch),
                                  torch.as_tensor(stats.get("spectral_elements", 0.), device=self.device)])
            value(means, counts, (1., kwargs.get("overlap_weight", 1.), kwargs.get("spectral_weight", 0.)))
            attribute = f"_epoch_losses.{key}"
        elif policy is not None:
            if matrix_stats is not None:
                errors = (loss.new_empty(0), loss.new_empty(0))
            elif domain:
                loss_s, _ = metrics_module._per_structure_losses(
                    out["node_labels"], batch.point_labels, out["edge_labels"], batch.edge_labels,
                    batch, self.basis_table, kwargs.get("node_weight", 1.), kwargs.get("edge_weight", 1.),
                )
                graph = metrics_module._node_graph(batch)
                atoms = torch.bincount(torch.as_tensor(graph, device=loss_s.device), minlength=len(loss_s))
                large = atoms >= int(kwargs["domain_threshold_atoms"])
                errors = (loss_s[~large], loss_s[large])
            elif physical_huber:
                errors, _ = self._huber_epoch_terms(out, batch)
            else:
                errors = metrics_module.get_predictions_error(
                    out["node_labels"], batch.point_labels, out["edge_labels"], batch.edge_labels,
                )
                if huber:
                    errors = tuple(metrics_module._smooth_l1(error, self.loss_fn.beta) for error in errors)
                if threshold:
                    cutoff = self.loss_fn.metric_kwargs.get("threshold", 1e-4)
                    if type(self.loss_fn) is metrics_module.block_type_mse_threshold:
                        errors = tuple(error[error.abs() > cutoff] for error in errors)
                    else:
                        factor = self.loss_fn.metric_kwargs.get("sigmoid_factor", 1e-5)
                        errors = tuple(metrics_module._sigmoid_threshold(error.abs(), cutoff, factor) for error in errors)
            if (key not in self._epoch_losses
                    or not isinstance(self._epoch_losses[key], _EpochMatrixLoss)
                    or (self._epoch_losses[key].coefficient_keys, self._epoch_losses[key].statistic,
                        self._epoch_losses[key].normalize_weights) != (coefficient_keys, statistic, normalize_weights)):
                self._epoch_losses[key] = _EpochMatrixLoss(coefficient_keys, statistic, normalize_weights).to(self.device)
            value = self._epoch_losses[key]
            value(*errors, *policy, coefficient_errors, coefficient_power, coefficient_weight, matrix_stats=matrix_stats)
            attribute = f"_epoch_losses.{key}"
        self.log(key, value, on_step=split == "train", on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch.num_graphs,
                 metric_attribute=attribute, sync_dist=attribute is None)
        if split == "val" and self.selection_metric is None:
            self.log("hp_metric", value, batch_size=batch.num_graphs,
                     metric_attribute=attribute, sync_dist=attribute is None)

    def _log_matrix_rmse(self, split, out, batch):
        """Loss-independent epoch error for model selection (roadmap 1.15).

        Global RMSE over every observed element of the predicted components, nodes and
        edges pooled with equal weight. Labels the model does not predict (S with an
        H-only gauge objective) are skipped; no gauge alignment is applied.
        """
        squared, count = 0., 0
        for pred, ref in ((out["node_labels"], batch.point_labels), (out["edge_labels"], batch.edge_labels)):
            if pred.ndim < ref.ndim or (pred.ndim == 2 and pred.shape[1] < ref.shape[1]):
                ref = ref[:, :1 if pred.ndim == 1 else pred.shape[1]].reshape(pred.shape)
            valid = torch.isfinite(ref)
            error = torch.where(valid, pred.detach().double() - ref.double(), 0.)
            squared = squared + error.square().sum()
            count = count + valid.sum()
        key = f"{split}_matrix_rmse"
        if key not in self._epoch_rmse:
            self._epoch_rmse[key] = _EpochRMSE().to(self.device)
        metric = self._epoch_rmse[key]
        metric(torch.sqrt(squared / torch.clamp(count, min=1)), count)
        self.log(key, metric, on_step=False, on_epoch=True, batch_size=batch.num_graphs,
                 metric_attribute=f"_epoch_rmse.{key}")

    def _log_selection_metric(self, batch):
        name = self.selection_metric
        stores = {"_epoch_losses": self._epoch_losses, "_epoch_rmse": self._epoch_rmse,
                  "_epoch_errors": self._epoch_errors}
        for key in (name, "multitarget_" + name.replace("/", "__")):
            for attribute, store in stores.items():
                if key in store and store[key].update_called:
                    self.log("hp_metric", store[key], batch_size=batch.num_graphs,
                             metric_attribute=f"{attribute}.{key}")
                    return
        available = sorted(key for store in stores.values() for key in store
                           if key.startswith("val") and store[key].update_called)
        raise ValueError(f"selection_metric {name!r} is not an epoch-aggregated metric in the current validation; "
                         f"available: {available}")

    def _physical_epoch_loss(self, key, batch, stats):
        """Matrix means use element counts; each physical mean uses structures."""
        kwargs = self.loss_fn.metric_kwargs
        if type(self.loss_fn) is metrics_module.gauge_invariant_mse:
            means = [stats["gauge_h_rmse"].double().square(), stats["overlap_rmse"].double().square(),
                torch.as_tensor(stats.get("spectral_rmse", 0.), device=self.device).double().square()]
            counts = [self._rmse_count("gauge_h_rmse", batch), self._rmse_count("overlap_rmse", batch),
                torch.as_tensor(stats.get("spectral_elements", 0.), device=self.device)]
            weights = [1., kwargs.get("overlap_weight", 1.), kwargs.get("spectral_weight", 0.)]
        else:
            node_count = self._rmse_count("node_rmse", batch).double()
            edge_count = self._rmse_count("edge_rmse", batch).double() * kwargs.get("edge_weight", 1.)
            count = node_count + edge_count
            base = (stats["node_rmse"].double().square()*node_count
                + stats["edge_rmse"].double().square()*edge_count) / torch.where(count > 0, count, 1.)
            means, counts, weights = [base], [count], [1.]
        for name, weight in zip(("electron", "mulliken", "energy"), self._physical_weights(self.loss_fn)):
            if weight:
                means.append(stats[f"physical_{name}_mse"].double())
                counts.append(torch.as_tensor(stats["physical_structures"], device=self.device))
                weights.append(weight)
        if (key not in self._epoch_losses or not isinstance(self._epoch_losses[key], _EpochWeightedLoss)
                or len(self._epoch_losses[key].totals) != len(means)):
            self._epoch_losses[key] = _EpochWeightedLoss(len(means)).to(self.device)
        value = self._epoch_losses[key]
        value(torch.stack(means), torch.stack(counts), tuple(weights))
        return value

    def _log_multitarget_loss(self, split, batch, stats):
        """Reduce each matrix/physical objective by its own global count."""
        def record(name, means, counts, weights):
            key = "multitarget_" + name.replace("/", "__")
            if key not in self._epoch_losses:
                self._epoch_losses[key] = _EpochWeightedLoss(len(means)).to(self.device)
            metric = self._epoch_losses[key]
            means = torch.stack([torch.as_tensor(value, device=self.device).double() for value in means])
            counts = torch.stack([torch.as_tensor(value, device=self.device).double() for value in counts])
            metric(means, counts, tuple(weights))
            self.log(name, metric, on_step=split == "train", on_epoch=True,
                     batch_size=batch.num_graphs, metric_attribute=f"_epoch_losses.{key}")
            return metric, key

        all_means, all_counts, all_weights = [], [], []
        for target in self.loss_fn.layout.targets:
            target_weight = self.loss_fn.weights[target]
            if target_weight == 0:
                continue
            prefix = f"target/{target}/"
            count = stats[prefix + "matrix_count"]
            means = [stats[prefix + "matrix_sum"] / count]
            counts, weights = [count], [1.]
            record(f"{split}/{prefix}matrix_mse", means, counts, weights)
            self.log(f"{split}/{prefix}matrix_count", count, on_step=False, on_epoch=True,
                     reduce_fx="sum", batch_size=1, sync_dist=True)
            for name, physical_weight in zip(("electron", "mulliken", "energy"),
                                              self.loss_fn.physical_weights[target]):
                if physical_weight:
                    value = stats[prefix + f"physical_{name}_mse"]
                    structures = stats[prefix + "physical_structures"]
                    record(f"{split}/{prefix}physical_{name}_mse", [value], [structures], [1.])
                    means.append(value)
                    counts.append(structures)
                    weights.append(physical_weight)
            record(f"{split}/{prefix}loss", means, counts, weights)
            all_means.extend(means)
            all_counts.extend(counts)
            all_weights.extend(target_weight * weight for weight in weights)
        metric, key = record(f"{split}_loss", all_means, all_counts, all_weights)
        if split == "val" and self.selection_metric is None:
            self.log("hp_metric", metric, batch_size=batch.num_graphs,
                     metric_attribute=f"_epoch_losses.{key}")

    def _huber_epoch_terms(self, out, batch):
        """Huber elements or physical-block means, with valid block types."""
        errors = metrics_module.get_predictions_error(
            out["node_labels"], batch.point_labels, out["edge_labels"], batch.edge_labels,
        )
        terms = tuple(metrics_module._smooth_l1(error, self.loss_fn.beta) for error in errors)
        per_block = (isinstance(self.loss_fn, metrics_module.block_normalized_huber)
                     or self.loss_fn.per_block_normalization)
        if not per_block:
            return terms, (None, None)
        pointers = metrics_module._node_edge_block_pointers(
            batch, self.basis_table, model=self.model, expected_edge_nlabels=len(errors[1]),
            nodes_ref=batch.point_labels, edges_ref=batch.edge_labels,
        )
        types = (metrics_module._as_numpy_int(batch.point_types),
                 abs(metrics_module._symmetric_edge_types_for_labels(
                     readout=getattr(self.model, "matrix_readouts", None), batch=batch,
                     basis_table=self.basis_table, expected_nlabels=len(batch.edge_labels),
                 )))
        blocks = [metrics_module._loss_blocks(values, ptr) for values, ptr in zip(terms, pointers)]
        return (
            tuple(torch.stack(values) if values else terms[i].new_empty(0)
                  for i, values in enumerate(blocks)),
            tuple(values[ptr[1:] > ptr[:-1]] for values, ptr in zip(types, pointers)),
        )

    def _coefficient_epoch_terms(self, out, batch):
        readout = metrics_module._coefficient_readout(self.model.matrix_readouts)
        keys = tuple(sorted(
            [f"node_node:{i}" for i, operation in enumerate(readout.self_interactions) if operation is not None]
            + [f"edge_edge:{key}" for key in readout.interactions]
        ))
        errors = metrics_module._coefficient_space_errors(
            batch.point_labels, batch.edge_labels, batch, self.basis_table, out, self.model,
        )
        return keys, errors

    def _log_stats(self, split, stats, batch, out, matrix_stats=None):
        if isinstance(self.loss_fn, MultitargetMatrixLoss):
            return  # Target metrics/counts are aggregated together in _log_multitarget_loss.
        errors = None
        physical_huber = isinstance(self.loss_fn, (metrics_module.block_normalized_huber,
                                                   metrics_module.hamiltonian_composite_loss))
        huber_terms = huber_types = None
        if physical_huber:
            huber_terms, huber_types = self._huber_epoch_terms(out, batch)
        coefficient = type(self.loss_fn) in (metrics_module.coefficient_space_mse, metrics_module.coefficient_space_mae)
        coefficient_keys, coefficient_errors = (), None
        if coefficient or getattr(self.loss_fn, "coefficient_mse_weight", 0.):
            coefficient_keys, coefficient_errors = self._coefficient_epoch_terms(out, batch)
            prefix = "coefficient_" if physical_huber else ""
            suffix = "mae" if type(self.loss_fn) is metrics_module.coefficient_space_mae else "rmse"
            stats = dict(stats)
            for key in coefficient_keys:
                stats.setdefault(f"{prefix}{key}_coeff_{suffix}", out["node_labels"].new_zeros(()))
            stats = dict(sorted(stats.items()))
        if isinstance(self.loss_fn, metrics_module.block_normalized_huber) and self.loss_fn.log_type_stats:
            # Every rank logs the same schema, even if it has no blocks of a type.
            stats = dict(stats)
            for kind, size in (("node", len(self.basis_table.basis)),
                               ("edge", len(self.basis_table.edge_block_size))):
                for type_id in range(size):
                    stats.setdefault(f"{kind}_type{type_id}_block_huber", out["node_labels"].new_zeros(()))
            stats = dict(sorted(stats.items()))
        for name, value in stats.items():
            huber_mean = (type(self.loss_fn) in (metrics_module.block_type_huber, metrics_module.block_type_smooth_l1)
                          and name in ("node_smooth_l1", "edge_smooth_l1"))
            count = (matrix_stats[f"{name.split('_')[0]}_count"] if matrix_stats is not None
                     else self._rmse_count(name, batch, stats))
            key = f"{split}_{name}"
            attribute = None
            if name == "physical_structures":
                if key not in self._epoch_errors:
                    self._epoch_errors[key] = _EpochErrorMoments("sum").to(self.device)
                value = self._epoch_errors[key]
                value(torch.as_tensor(stats[name], device=self.device))
                attribute = f"_epoch_errors.{key}"
            elif name == "physical_delta_n_rmse":
                if key not in self._epoch_rmse:
                    self._epoch_rmse[key] = _EpochRMSE().to(self.device)
                value = self._epoch_rmse[key]
                value(stats[name], stats["physical_structures"])
                attribute = f"_epoch_rmse.{key}"
            elif name.startswith("physical_"):
                if key not in self._epoch_losses:
                    self._epoch_losses[key] = _EpochWeightedLoss(1).to(self.device)
                value = self._epoch_losses[key]
                value([stats[name]], [stats["physical_structures"]], (1.,))
                attribute = f"_epoch_losses.{key}"
            elif coefficient_errors is not None and name.endswith(("_coeff_rmse", "_coeff_mae")):
                operation = name[len(prefix):].rsplit("_coeff_", 1)[0]
                terms = coefficient_errors.get(operation, out["node_labels"].new_empty(0))
                if name.endswith("_rmse"):
                    if key not in self._epoch_rmse:
                        self._epoch_rmse[key] = _EpochRMSE().to(self.device)
                    value = self._epoch_rmse[key]
                    value(metrics_module._mean(terms.detach().double().square()).sqrt(), terms.numel())
                    attribute = f"_epoch_rmse.{key}"
                else:
                    if key not in self._epoch_errors:
                        self._epoch_errors[key] = _EpochErrorMoments("mean").to(self.device)
                    value = self._epoch_errors[key]
                    value(terms)
                    attribute = f"_epoch_errors.{key}"
            elif coefficient_errors is not None and name in ("coefficient_mse", "coefficient_blocks", "coefficient_coefficient_blocks"):
                if key not in self._epoch_losses:
                    statistic = "operations" if name.endswith("_blocks") else "coefficients"
                    self._epoch_losses[key] = _EpochMatrixLoss(coefficient_keys, statistic).to(self.device)
                value = self._epoch_losses[key]
                empty = out["node_labels"].new_empty(0)
                value(empty, empty, 1, False, 0., 0., coefficient_errors)
                attribute = f"_epoch_losses.{key}"
            elif name == "gauge_shift_mean_abs":
                if key not in self._epoch_losses:
                    self._epoch_losses[key] = _EpochWeightedLoss(1).to(self.device)
                value = self._epoch_losses[key]
                value(stats[name].reshape(1), [batch.num_graphs], (1.,))
                attribute = f"_epoch_losses.{key}"
            elif (type(self.loss_fn) is metrics_module.normalized_density_error
                    and name in ("avg_per_config_error_percent", "avg_error_percent")):
                if key not in self._epoch_losses:
                    self._epoch_losses[key] = _EpochWeightedLoss(1).to(self.device)
                value = self._epoch_losses[key]
                denominator = stats["total_electrons"] if name == "avg_error_percent" else stats["n_structures"]
                value([stats[name]], [denominator], (1.,))
                attribute = f"_epoch_losses.{key}"
            elif name in ("n_structures", "n_large_structures", "spectral_elements", "n_pairs",
                          "total_electrons", "total_error", "per_config_total_error"):
                if key not in self._epoch_errors:
                    self._epoch_errors[key] = _EpochErrorMoments("sum").to(self.device)
                value = self._epoch_errors[key]
                value(torch.as_tensor(stats[name], dtype=torch.float64, device=self.device))
                attribute = f"_epoch_errors.{key}"
            elif name.endswith(("_above_threshold_mean", "_above_threshold_frac")):
                if errors is None:
                    errors = metrics_module.get_predictions_error(
                        out["node_labels"], batch.point_labels, out["edge_labels"], batch.edge_labels,
                    )
                terms = errors[0 if name.startswith("node_") else 1].abs()
                selected = terms > self.loss_fn.metric_kwargs.get("threshold", 1e-4)
                terms = selected.to(terms.dtype) if name.endswith("_frac") else terms[selected]
                if key not in self._epoch_errors:
                    self._epoch_errors[key] = _EpochErrorMoments("mean").to(self.device)
                value = self._epoch_errors[key]
                value(terms)
                attribute = f"_epoch_errors.{key}"
            elif count is not None:
                if key not in self._epoch_rmse:
                    self._epoch_rmse[key] = _EpochRMSE().to(self.device)
                metric = self._epoch_rmse[key]
                metric(value, count)
                value = metric
                attribute = f"_epoch_rmse.{key}"
            elif physical_huber and name == "label_huber":
                if key not in self._epoch_losses:
                    self._epoch_losses[key] = _EpochMatrixLoss().to(self.device)
                value = self._epoch_losses[key]
                value(*huber_terms, 1, False, self.loss_fn.node_weight, self.loss_fn.edge_weight)
                attribute = f"_epoch_losses.{key}"
            elif physical_huber and (name in ("node_smooth_l1", "edge_smooth_l1", "node_blocks", "edge_blocks")
                                     or name.endswith("_block_huber")):
                statistic = "sum" if name.endswith("_blocks") else "mean"
                if key not in self._epoch_errors:
                    self._epoch_errors[key] = _EpochErrorMoments(statistic).to(self.device)
                terms = huber_terms[0 if name.startswith("node_") else 1]
                if "_type" in name:
                    type_id = int(name.split("_type")[1].split("_")[0])
                    types = huber_types[0 if name.startswith("node_") else 1]
                    terms = terms[torch.as_tensor(types == type_id, device=terms.device)]
                elif statistic == "sum":
                    terms = stats[name]
                value = self._epoch_errors[key]
                value(terms)
                attribute = f"_epoch_errors.{key}"
            elif huber_mean or name in {f"{kind}_{stat}" for kind in ("node", "edge") for stat in ("mean", "std", "max")}:
                if errors is None:
                    errors = metrics_module.get_predictions_error(
                        out["node_labels"], batch.point_labels, out["edge_labels"], batch.edge_labels,
                    )
                if key not in self._epoch_errors:
                    statistic = "mean" if huber_mean else name.rsplit("_", 1)[1]
                    self._epoch_errors[key] = _EpochErrorMoments(statistic).to(self.device)
                value = self._epoch_errors[key]
                terms = errors[0 if name.startswith("node_") else 1]
                if huber_mean:
                    terms = metrics_module._smooth_l1(terms, self.loss_fn.beta)
                value(terms)
                attribute = f"_epoch_errors.{key}"
            self.log(key, value, on_step=split == "train", on_epoch=True,
                     batch_size=batch.num_graphs, reduce_fx=_epoch_reduce(name),
                     metric_attribute=attribute, sync_dist=attribute is None)

    def on_fit_start(self) -> None:
        self._check_stages_under_ddp()
        self._preflight_physical_objectives()
        self._apply_training_stage_for_epoch(getattr(self, "current_epoch", 0))
        self._fit_output_scaleshift()
        self._configure_coefficient_training()

    def _check_stages_under_ddp(self) -> None:
        """Stages that freeze parameters change which ones DDP sees used: DDP registers them
        before the stage freezes any, so without ``find_unused_parameters=True`` the first
        step fails with "parameters that were not used in producing the loss"."""
        trainer = getattr(self, "_trainer", None)
        strategy = getattr(trainer, "strategy", None)
        if strategy is None or getattr(trainer, "world_size", 1) <= 1:
            return
        freezes = any(stage.get("freeze") or stage.get("train") is not None
                      for stage in self._validated_training_stages)
        if freezes and not getattr(strategy, "_ddp_kwargs", {}).get("find_unused_parameters", False):
            raise ValueError("training stages that freeze or select parameters need "
                             "DDPStrategy(find_unused_parameters=True) with several processes.")

    @staticmethod
    def _physical_weights(loss_fn):
        kwargs = getattr(loss_fn, "metric_kwargs", {})
        return validate_physical_weights(**{name: kwargs.get(name, 0.)
            for name in ("electron_weight", "mulliken_weight", "energy_weight")})

    def _preflight_physical_objectives(self):
        """Check every future physical objective before forward/scale fitting/steps.

        Index map-style datasets in CPU, reusing existing caches without consuming
        sampler/worker RNG or streams. Lazy data pay one full validation pass.
        """
        from torch.utils.data import DataLoader, IterableDataset
        from .data import MatrixDataModule
        from graph2mat.bindings.torch.data.dataset import RotatingPoolData

        objectives = [("default", self.default_loss_fn)]
        for stage in self._validated_training_stages:
            loss_fn = (self._resolve_metric(stage["loss"])(**(stage.get("loss_kwargs") or {}))
                if "loss" in stage else self.default_loss_fn)
            objectives.append((stage["name"], loss_fn))
        objectives = [(name, fn) for name, fn in objectives
                      if isinstance(fn, MultitargetMatrixLoss) or any(self._physical_weights(fn))]
        if not objectives:
            return
        readouts = getattr(self.model, "matrix_readouts", self.model)
        readouts = list(readouts) if isinstance(readouts, torch.nn.ModuleList) else [readouts]
        components = [getattr(readout, "n_matrix_components",
            self.model_kwargs.get("n_matrix_components")) for readout in readouts]
        if (not components or any(not isinstance(n, int) or isinstance(n, bool) or n < 1
                for n in components) or len(set(components)) != 1):
            raise ValueError("physical preflight requires a declared, consistent model n_matrix_components")
        n_components = components[0]
        convention = None

        def loaders(value):
            if isinstance(value, DataLoader):
                yield value
            elif isinstance(value, dict):
                for child in value.values():
                    yield from loaders(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    yield from loaders(child)
            else:
                raise ValueError("physical preflight requires inspectable map-style DataLoaders")

        trainer = self.trainer
        # Lightning has not created worker iterators at on_fit_start. Its source
        # covers both explicit loaders and module hooks; fail closed if unavailable.
        loops = (("train", trainer.fit_loop), ("val", trainer.fit_loop.epoch_loop.val_loop))
        for split, loop in loops:
            source = getattr(loop, "_data_source", None)
            if source is None:
                raise ValueError("physical preflight cannot inspect this Lightning loader source")
            if split == "val" and (not trainer.limit_val_batches or not source.is_defined()):
                continue
            for loader in loaders(source.dataloader()):
                dataset = loader.dataset
                module = trainer.datamodule
                if isinstance(module, MatrixDataModule):
                    pool = getattr(module, split + "_dataset", None)
                    if isinstance(pool, RotatingPoolData):
                        dataset = pool.parent_data  # Validate future replacements, not only today's pool.
                if isinstance(dataset, IterableDataset) or not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
                    raise ValueError(f"physical preflight cannot inspect {split} streaming data without consuming them")
                for index in range(len(dataset)):
                    name = "all physical objectives"
                    try:
                        batch = loader.collate_fn([dataset[index]])
                        # Validate the actual declared output shape, not labels against themselves.
                        out = {pred: batch[ref].new_zeros((len(batch[ref]),)
                            if n_components == 1 else (len(batch[ref]), n_components))
                            for pred, ref in (("node_labels", "point_labels"), ("edge_labels", "edge_labels"))}
                        for name, loss_fn in objectives:
                            self._validate_pred_ref_shapes(out, batch,
                                getattr(loss_fn, "overlap_prediction_optional", False))
                            with torch.no_grad():
                                loss_fn(nodes_pred=out["node_labels"], nodes_ref=batch["point_labels"],
                                    edges_pred=out["edge_labels"], edges_ref=batch["edge_labels"],
                                    batch=batch, basis_table=self.basis_table)
                        if isinstance(self.default_loss_fn, MultitargetMatrixLoss):
                            current = tuple(batch.matrix_target_signature[0].tolist())
                        else:
                            current = (int(batch["physical_kind"].item()), int(batch["physical_spin"].item()))
                        if convention is not None and current != convention:
                            raise ValueError("physical target/spin must agree across train/validation samples")
                        convention = current
                    except (ValueError, TypeError, KeyError, AttributeError) as error:
                        raise ValueError(f"physical preflight {split} sample {index}, stage {name}: {error}") from error

    def _configure_coefficient_training(self):
        """Snapshot after normal restore/device setup, before train worker iterators."""
        from graph2mat.models.mace import MatrixMACE
        from .data import MatrixDataModule

        module = self.trainer.datamodule
        if isinstance(module, MatrixDataModule):
            module.set_coefficient_mse_projector(None)
        self._coefficient_training_unavailable = None
        if not self.coefficient_training:
            return
        if self.trainer.strategy.restore_checkpoint_after_setup:
            reason = "restore_checkpoint_after_setup cannot snapshot the restored Q before workers"
        elif not isinstance(self.model, MatrixMACE):
            reason = "only MatrixMACE is certified"
        elif not isinstance(module, MatrixDataModule):
            reason = "MatrixDataModule is required to prepare CPU targets"
        else:
            try:
                projector = CoefficientMatrixMSEProjector(self.model.matrix_readouts)
            except CoefficientMSEUnsupported as error:
                reason = str(error)
            else:
                if (projector.n_components != module.n_matrix_components
                        or projector.symmetric != module.data_processor.symmetric_matrix
                        or list(projector.basis_table.basis) != list(module.basis_table.basis)):
                    reason = "model/data physical basis, components or symmetry differ"
                else:
                    module.set_coefficient_mse_projector(projector)
                    return
        self._coefficient_training_unavailable = reason
        self._warn_coefficient_fallback(reason)

    def on_fit_end(self) -> None:
        from .data import MatrixDataModule

        if isinstance(self.trainer.datamodule, MatrixDataModule):
            self.trainer.datamodule.set_coefficient_mse_projector(None)

    def _fit_output_scaleshift(self) -> None:
        """Fits the readout output scale/shift from the training labels, once."""
        readout = getattr(self.model, "matrix_readouts", None)
        if not getattr(readout, "output_scaleshift", False):
            return
        if bool(readout.output_scaleshift_fitted):
            return  # e.g. resuming from a checkpoint
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            raise ValueError("output_scaleshift fits its scale/shift on the training data: fit with a "
                             "datamodule (Trainer.fit(model, datamodule=...)), not bare dataloaders.")
        loader = datamodule.train_dataloader()
        readout.fit_output_scaleshift(
            (batch.to(self.device) for batch in loader), self.basis_table
        )

    def on_train_epoch_start(self) -> None:
        self._apply_training_stage_for_epoch(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        out = self.model(batch)
        self._validate_pred_ref_shapes(
            out, batch, getattr(self.loss_fn, "overlap_prediction_optional", False)
        )

        loss, stats = self.loss_fn(
            nodes_pred=out["node_labels"],
            nodes_ref=batch["point_labels"],
            edges_pred=out["edge_labels"],
            edges_ref=batch["edge_labels"],
            batch=batch,
            basis_table=self.basis_table,
            log_verbose=True,
            out=out,
            model=self.model,
        )

        self._log_loss("val", loss, out, batch, stats)
        self._log_stats("val", stats, batch, out)
        if not isinstance(self.loss_fn, MultitargetMatrixLoss):
            # Packed multitarget components carry different units; they log per target.
            self._log_matrix_rmse("val", out, batch)
        if self.selection_metric is not None:
            self._log_selection_metric(batch)

        # Only what callbacks read: keeping every MACE output alive until the end of
        # the batch only costs memory.
        return {"loss": loss, "node_labels": out["node_labels"], "edge_labels": out["edge_labels"],
                **({"matrix_predictions": out["matrix_predictions"]} if "matrix_predictions" in out else {})}

    def test_step(self, batch, batch_idx):
        out = self.model(batch)
        self._validate_pred_ref_shapes(
            out, batch, getattr(self.loss_fn, "overlap_prediction_optional", False)
        )

        loss, stats = self.loss_fn(
            nodes_pred=out["node_labels"],
            nodes_ref=batch["point_labels"],
            edges_pred=out["edge_labels"],
            edges_ref=batch["edge_labels"],
            batch=batch,
            basis_table=self.basis_table,
            log_verbose=True,
            out=out,
            model=self.model,
        )

        self._log_loss("test", loss, out, batch, stats)
        self._log_stats("test", stats, batch, out)
        if not isinstance(self.loss_fn, MultitargetMatrixLoss):
            self._log_matrix_rmse("test", out, batch)

        return out

    def on_save_checkpoint(self, checkpoint) -> None:
        "Objects to include in checkpoint file"
        checkpoint["basis_table"] = self.basis_table
        checkpoint["version"] = __version__
        checkpoint["training_stages"] = self.training_stages_config
        checkpoint["active_training_stage"] = self._active_training_stage_name
        checkpoint["provenance"] = _training_provenance()
        layout = getattr(self, "matrix_target_layout", None)
        if layout is not None:
            checkpoint["matrix_target_descriptor"] = layout.to_dict()

    def on_load_checkpoint(self, checkpoint) -> None:
        "Objects to retrieve from checkpoint file"
        san_checkpoint = sanitize_checkpoint(checkpoint)
        checkpoint.update(san_checkpoint)

        active_stage = checkpoint.get("active_training_stage")
        if active_stage is not None:
            stages = [stage for stage in self._validated_training_stages if stage["name"] == active_stage]
            if len(stages) != 1:
                raise ValueError(f"checkpoint active training stage {active_stage!r} is missing or ambiguous")
            # Evaluation needs the saved loss; parameter/optimizer policies belong to fit,
            # after DDP registers parameters. Leave the active index unset for that hook.
            self._apply_stage_loss(stages[0])
            self._active_training_stage_name = active_stage
            self._active_training_stage_index = None

        descriptor = checkpoint.get("matrix_target_descriptor")
        layout = getattr(self, "matrix_target_layout", None)
        if descriptor is not None or layout is not None:
            from graph2mat.core.data.multitarget import MatrixTargetLayout

            if descriptor is None or MatrixTargetLayout.from_dict(descriptor) != layout:
                raise ValueError("checkpoint multitarget semantic descriptor/layout mismatch")
            data_params = checkpoint.get("datamodule_hyper_parameters", {})
            if data_params:
                data_layout = MatrixTargetLayout(tuple(data_params.get("matrix_targets") or ()),
                                                 data_params.get("matrix_target_spin"))
                if data_layout != layout:
                    raise ValueError("checkpoint data/model multitarget descriptor mismatch")

        try:
            self.basis_table = checkpoint["basis_table"]
        except KeyError:
            warnings.warn(
                "Failed to load basis_table from checkpoint: Key does not exist."
            )

        try:
            ckpt_version = checkpoint["version"]
        except KeyError:
            ckpt_version = None
            warnings.warn("Unable to determine version that created checkpoint file")
        if ckpt_version:
            if not (ckpt_version == __version__):
                warnings.warn(
                    "The checkpoint version %s does not match the current package version %s"
                    % (ckpt_version, __version__)
                )

    def _validate_training_stages(
        self,
        stages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not stages:
            return []
        if not isinstance(stages, list):
            raise TypeError("training_stages must be a list of dictionaries.")

        allowed_keys = {
            "name",
            "freeze",
            "train",
            "loss",
            "loss_kwargs",
            "max_epochs",
            "optim_lr",
        }
        validated = []
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                raise TypeError("Each training stage must be a dictionary.")
            unknown_keys = set(stage) - allowed_keys
            if unknown_keys:
                raise ValueError(
                    "Unknown training stage keys: "
                    f"{', '.join(sorted(unknown_keys))}."
                )

            max_epochs = int(stage.get("max_epochs", 0))
            if max_epochs <= 0:
                raise ValueError(
                    f"training stage {index} must define max_epochs > 0."
                )

            validated_stage = {
                **stage,
                "name": stage.get("name", f"stage_{index}"),
                "max_epochs": max_epochs,
                "freeze": self._normalize_stage_groups(stage.get("freeze", [])),
                "train": (
                    None
                    if "train" not in stage
                    else self._normalize_stage_groups(stage.get("train", []))
                ),
            }
            if "optim_lr" in stage:
                optim_lr = float(stage["optim_lr"])
                if optim_lr <= 0:
                    raise ValueError("training stage optim_lr must be positive.")
                validated_stage["optim_lr"] = optim_lr

            self._validate_stage_groups(validated_stage["freeze"])
            if validated_stage["train"] is not None:
                self._validate_stage_groups(validated_stage["train"])
            if "loss" in stage:
                self._resolve_metric(stage["loss"])
                if stage.get("loss_kwargs") is not None and not isinstance(
                    stage["loss_kwargs"], dict
                ):
                    raise TypeError("training stage loss_kwargs must be a dictionary.")
                self._resolve_metric(stage["loss"])(**(stage.get("loss_kwargs") or {}))
            elif stage.get("loss_kwargs"):
                weights = validate_physical_weights(**{name: stage["loss_kwargs"].get(name, 0.)
                    for name in ("electron_weight", "mulliken_weight", "energy_weight")})
                if any(weights):
                    raise ValueError("physical stage loss_kwargs require an explicit stage loss")

            validated.append(validated_stage)
        return validated

    @staticmethod
    def _normalize_stage_groups(groups) -> list[str]:
        if groups is None:
            return []
        if isinstance(groups, str):
            return [groups]
        if not isinstance(groups, (list, tuple)):
            raise TypeError("stage parameter groups must be a string or list.")
        return [str(group) for group in groups]

    def _validate_stage_groups(self, groups: list[str]) -> None:
        valid_groups = self._parameter_group_modules()
        unknown = [group for group in groups if group not in valid_groups]
        if unknown:
            raise ValueError(
                "Unknown training stage parameter groups: "
                f"{', '.join(sorted(unknown))}."
            )

    def _parameter_group_modules(self) -> dict[str, torch.nn.Module]:
        groups = {"model": self.model, "all": self.model}
        if hasattr(self.model, "mace"):
            groups["mace"] = self.model.mace
        if hasattr(self.model, "matrix_readouts"):
            groups["readout"] = self.model.matrix_readouts
            groups["readouts"] = self.model.matrix_readouts
            groups["matrix_readouts"] = self.model.matrix_readouts
        context = getattr(self.model, "hamiltonian_local_context", None)
        if isinstance(context, torch.nn.Module):
            groups["hamiltonian_context"] = context
            groups["context"] = context
        return groups

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)

    def _resolve_metric(self, metric):
        if isinstance(metric, type) and issubclass(metric, OrbitalMatrixMetric):
            return metric
        if isinstance(metric, OrbitalMatrixMetric):
            return metric.__class__
        if isinstance(metric, str):
            if "." not in metric:
                if not hasattr(metrics_module, metric):
                    raise ValueError(f"Unknown metric {metric!r}.")
                resolved = getattr(metrics_module, metric)
            else:
                path = metric
                if path.startswith("graph2mat.metrics."):
                    path = path.replace(
                        "graph2mat.metrics.",
                        "graph2mat.core.data.metrics.",
                        1,
                    )
                module_path, name = path.rsplit(".", 1)
                resolved = getattr(importlib.import_module(module_path), name)
            if not isinstance(resolved, type) or not issubclass(
                resolved, OrbitalMatrixMetric
            ):
                raise TypeError(f"Resolved metric {metric!r} is not a metric class.")
            return resolved
        raise TypeError("stage loss must be a metric class, instance, or string.")

    def _stage_for_epoch(self, epoch: int) -> Optional[tuple[int, dict[str, Any]]]:
        if not self._validated_training_stages:
            return None
        end_epoch = 0
        for index, stage in enumerate(self._validated_training_stages):
            end_epoch += stage["max_epochs"]
            if epoch < end_epoch:
                return index, stage
        return (
            len(self._validated_training_stages) - 1,
            self._validated_training_stages[-1],
        )

    def _apply_training_stage_for_epoch(self, epoch: int) -> None:
        stage_item = self._stage_for_epoch(epoch)
        if stage_item is None:
            return
        index, stage = stage_item
        if index == self._active_training_stage_index:
            self._apply_stage_optimizer(stage)
            return

        self._active_training_stage_index = index
        self._active_training_stage_name = stage["name"]
        self._apply_stage_parameter_policy(stage)
        self._apply_stage_loss(stage)
        self._apply_stage_optimizer(stage)

    def _apply_stage_parameter_policy(self, stage: dict[str, Any]) -> None:
        groups = self._parameter_group_modules()
        train_groups = stage["train"]
        if train_groups is None:
            self._set_requires_grad(self.model, True)
        else:
            self._set_requires_grad(self.model, False)
            for group in train_groups:
                self._set_requires_grad(groups[group], True)
        for group in stage["freeze"]:
            self._set_requires_grad(groups[group], False)

    def _apply_stage_loss(self, stage: dict[str, Any]) -> None:
        if "loss" not in stage:
            self.loss_fn = self.default_loss_fn
            return
        loss_cls = self._resolve_metric(stage["loss"])
        self.loss_fn = loss_cls(**(stage.get("loss_kwargs") or {}))

    def _apply_stage_optimizer(self, stage: dict[str, Any]) -> None:
        if "optim_lr" not in stage:
            return
        trainer = getattr(self, "trainer", None)
        optimizers = getattr(trainer, "optimizers", []) if trainer is not None else []
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = stage["optim_lr"]
