"""Wrapping of raw models to use them in pytorch_lightning."""

import importlib
import importlib.metadata
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
from graph2mat.bindings.torch.modules.graph2mat import layout_to
from graph2mat.core.modules.graph2mat import LAYOUT_ATTR

# from context import mace
from graph2mat.core.data import metrics as metrics_module
from graph2mat.core.data.metrics import OrbitalMatrixMetric, elementwise_mse
from graph2mat.core.data.node_feats import NodeFeature

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
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

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
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        if layout is not None:
            object.__setattr__(batch, LAYOUT_ATTR, layout_to(layout, device))
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

    def training_step(self, batch, batch_idx):
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

        n = batch.num_graphs
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
            batch_size=n,
        )

        self._log_stats("train", stats, batch)

        # Only what callbacks read: keeping every MACE output alive until the end of
        # the batch only costs memory.
        return {"loss": loss, "node_labels": out["node_labels"], "edge_labels": out["edge_labels"]}

    def _rmse_count(self, name, batch):
        """Number of terms used by the standard matrix RMSE statistics."""
        if name in ("gauge_h_rmse", "overlap_rmse"):
            edge_weight = self.loss_fn.metric_kwargs.get("edge_weight", 1.0)
            count = batch.point_labels.shape[0] + edge_weight * batch.edge_labels.shape[0]
            return count * (batch.point_labels.shape[1] - 1) if name == "gauge_h_rmse" else count
        for kind, labels in (("node", batch.point_labels), ("edge", batch.edge_labels)):
            if name != f"{kind}_rmse" and not name.startswith(f"{kind}_rmse_spin"):
                continue
            if isinstance(self.loss_fn, (metrics_module.block_type_mse_per_structure,
                                         metrics_module.block_type_mse_per_domain)):
                node_ids, edge_ids, _ = metrics_module._per_element_structure_ids(
                    batch.point_labels, batch.edge_labels, batch, self.basis_table,
                )
                ids = (node_ids if kind == "node" else edge_ids).to(labels.device)
                if kind == "edge":
                    valid = ~torch.isnan(labels)
                    if labels.ndim == 2:
                        valid = valid.all(dim=1)
                    ids = ids[valid]
                return torch.bincount(ids, minlength=batch.num_graphs).count_nonzero()
            count = labels.shape[0]
            if kind == "edge":
                invalid = torch.isnan(labels)
                if labels.ndim == 2:
                    invalid = invalid.any(dim=1)
                count = (~invalid).sum()
            if name == f"{kind}_rmse" and labels.ndim == 2:
                count *= labels.shape[1]
            return count
        return None

    def _log_stats(self, split, stats, batch):
        for name, value in stats.items():
            count = self._rmse_count(name, batch)
            key = f"{split}_{name}"
            if count is not None:
                if key not in self._epoch_rmse:
                    self._epoch_rmse[key] = _EpochRMSE().to(self.device)
                metric = self._epoch_rmse[key]
                metric(value, count)
                value = metric
            self.log(key, value, on_step=split == "train", on_epoch=True,
                     batch_size=batch.num_graphs, reduce_fx=_epoch_reduce(name),
                     metric_attribute=f"_epoch_rmse.{key}" if count is not None else None)

    def on_fit_start(self) -> None:
        self._apply_training_stage_for_epoch(getattr(self, "current_epoch", 0))
        self._fit_output_scaleshift()

    def _fit_output_scaleshift(self) -> None:
        """Fits the readout output scale/shift from the training labels, once."""
        readout = getattr(self.model, "matrix_readouts", None)
        if not getattr(readout, "output_scaleshift", False):
            return
        if bool(readout.output_scaleshift_fitted):
            return  # e.g. resuming from a checkpoint
        loader = self.trainer.datamodule.train_dataloader()
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

        # Epoch values are means weighted by structures (the last, smaller batch must not
        # weigh as much as a full one), except maxima.
        n = batch.num_graphs
        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=n)
        # save validation loss as the hyperparameter opt metric (used by tensorboard)
        self.log("hp_metric", loss, batch_size=n)

        self._log_stats("val", stats, batch)

        # Only what callbacks read: keeping every MACE output alive until the end of
        # the batch only costs memory.
        return {"loss": loss, "node_labels": out["node_labels"], "edge_labels": out["edge_labels"]}

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

        n = batch.num_graphs
        self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=n)

        self._log_stats("test", stats, batch)

        return out

    def on_save_checkpoint(self, checkpoint) -> None:
        "Objects to include in checkpoint file"
        checkpoint["basis_table"] = self.basis_table
        checkpoint["version"] = __version__
        checkpoint["training_stages"] = self.training_stages_config
        checkpoint["active_training_stage"] = self._active_training_stage_name
        checkpoint["provenance"] = _training_provenance()

    def on_load_checkpoint(self, checkpoint) -> None:
        "Objects to retrieve from checkpoint file"
        san_checkpoint = sanitize_checkpoint(checkpoint)
        checkpoint.update(san_checkpoint)

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
