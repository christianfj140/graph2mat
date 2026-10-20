"""Implements a custom CLI that slightly tweaks pytorch_lightning's default.

Most
"""

import copy
import os
from pytorch_lightning.cli import (
    LightningCLI,
    SaveConfigCallback,
    LightningArgumentParser,
    ArgsType,
)
import torch
from jsonargparse import Namespace
from jsonargparse._typehints import ActionTypeHint

from graph2mat.bindings.torch.load import sanitize_checkpoint
from graph2mat.tools.lightning.data import (
    infer_n_matrix_components_from_data_inputs,
    resolve_symmetric_matrix,
)


def _apply_selection_metric(config_ns: Namespace) -> None:
    """Point checkpoints monitoring ``val_loss`` at ``model.selection_metric`` (roadmap 1.15).

    Checkpoints that monitor anything else were chosen explicitly and are kept.
    """
    metric = getattr(getattr(config_ns, "model", None), "selection_metric", None)
    if metric is None or metric == "val_loss":
        return
    for callback in getattr(getattr(config_ns, "trainer", None), "callbacks", None) or []:
        init_args = getattr(callback, "init_args", None)
        if (str(getattr(callback, "class_path", "")).rsplit(".", 1)[-1] == "ModelCheckpoint"
                and getattr(init_args, "monitor", None) == "val_loss"):
            init_args.monitor = metric


def _autoconfigure_n_matrix_components(config_ns: Namespace, stage=None) -> None:
    """Infer matrix components from the dataset when config still uses the default."""
    data_cfg = getattr(config_ns, "data", None)
    model_cfg = getattr(config_ns, "model", None)
    if data_cfg is None or model_cfg is None:
        return

    spin_orbit = getattr(data_cfg, "matrix_component_policy", "h_only") == "spin_orbit"
    if getattr(model_cfg, "spin_orbit", False) and not spin_orbit:
        raise ValueError("model.spin_orbit=True needs data.matrix_component_policy='spin_orbit'.")
    if spin_orbit or hasattr(model_cfg, "spin_orbit"):
        model_cfg.spin_orbit = spin_orbit
    options = dict(
        root_dir=getattr(data_cfg, "root_dir", "."),
        basis_files=getattr(data_cfg, "basis_files", None),
        no_basis=getattr(data_cfg, "no_basis", None),
        basis_table=getattr(data_cfg, "basis_table", None),
        out_matrix=getattr(data_cfg, "out_matrix", None),
        matrix_component_policy=getattr(data_cfg, "matrix_component_policy", "h_only"),
        symmetric_matrix=resolve_symmetric_matrix(
            getattr(data_cfg, "symmetric_matrix", None),
            getattr(data_cfg, "out_matrix", None),
        ),
        sub_point_matrix=getattr(data_cfg, "sub_point_matrix", True),
        initial_node_feats=getattr(data_cfg, "initial_node_feats", "OneHotZ"),
        train_runs=getattr(data_cfg, "train_runs", None),
        val_runs=getattr(data_cfg, "val_runs", None),
        test_runs=getattr(data_cfg, "test_runs", None),
        runs_json=getattr(data_cfg, "runs_json", None),
        stage=stage,
    )
    targets = getattr(data_cfg, "matrix_targets", None)
    if targets is not None:
        from graph2mat.core.data.multitarget import MatrixTargetLayout

        if getattr(data_cfg, "sub_point_matrix", True):
            raise ValueError("multitarget requires sub_point_matrix=False; no baseline")
        spin = getattr(data_cfg, "matrix_target_spin", None)
        if spin is None:
            if stage == "predict":
                raise ValueError("matrix_target_spin must come from config/checkpoint for unlabeled inference")
            options.update(out_matrix="hamiltonian", matrix_component_policy="h_only")
            spin = infer_n_matrix_components_from_data_inputs(**options)
            if spin is None:
                raise ValueError("cannot infer matrix_target_spin from labeled references")
        layout = MatrixTargetLayout(tuple(targets), spin)
        model_targets = getattr(model_cfg, "matrix_targets", None)
        model_spin = getattr(model_cfg, "matrix_target_spin", None)
        if model_targets is not None and MatrixTargetLayout(tuple(model_targets), model_spin or spin) != layout:
            raise ValueError("data/model multitarget semantic descriptors disagree")
        if model_spin is not None and model_spin != spin:
            raise ValueError("data/model multitarget spin disagrees")
        for config in (data_cfg, model_cfg):
            if getattr(config, "n_matrix_components", 1) not in (1, layout.n_components):
                raise ValueError("n_matrix_components disagrees with multitarget descriptor")
            config.matrix_targets = list(layout.targets)
            config.matrix_target_spin = spin
            config.n_matrix_components = layout.n_components
        if getattr(model_cfg, "n_output_components", None) not in (None, layout.n_components):
            raise ValueError("n_output_components cannot omit multitarget components")
        return
    if getattr(model_cfg, "matrix_targets", None) is not None:
        raise ValueError("model matrix_targets require matching data matrix_targets")
    if spin_orbit:
        # [h0, gx, gy, gz] is fixed by the policy: no labels needed (also for predict).
        for config in (data_cfg, model_cfg):
            if getattr(config, "n_matrix_components", 1) not in (1, 4):
                raise ValueError("spin_orbit stores 4 components; n_matrix_components must be 1 or 4")
            config.n_matrix_components = 4
        return
    configured = getattr(data_cfg, "n_matrix_components", None)
    if configured != 1:
        return
    inferred = infer_n_matrix_components_from_data_inputs(**options)
    if inferred is None or inferred == configured:
        return

    data_cfg.n_matrix_components = inferred
    model_cfg.n_matrix_components = inferred


class OrbitalMatrixCLI(LightningCLI):
    """Custom pytorch_lightning CLI optimized for matrix learning.

    There are some defaults that change.

    However, the most relevant change is that when loading a checkpoint
    with the ``ckpt_path`` key, all the options stored in the checkpoint
    file will be used as defaults. This change was made so that you can
    just load a checkpoint file and use it without needing to provide
    all the settings that were used to generate that checkpoint (which
    is the way raw pytorch_lightning works).
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.link_arguments("data.root_dir", "model.root_dir")
        parser.link_arguments("data.basis_files", "model.basis_files")
        parser.link_arguments("data.basis_table", "model.basis_table")
        parser.link_arguments("data.no_basis", "model.no_basis")
        parser.link_arguments(
            ("data.symmetric_matrix", "data.out_matrix"),
            "model.symmetric_matrix",
            compute_fn=resolve_symmetric_matrix,
        )
        parser.link_arguments(
            "data.n_matrix_components", "model.n_matrix_components"
        )
        parser.link_arguments("data.matrix_targets", "model.matrix_targets")
        parser.link_arguments("data.matrix_target_spin", "model.matrix_target_spin")
        parser.link_arguments("data.edge_cutoff_padding", "model.edge_cutoff_padding")
        parser.link_arguments("data.initial_node_feats", "model.initial_node_feats")

        defaults = {}
        # Set logger defaults based on environment variables
        if os.environ.get("PL_LOGGER_DEFAULTS_ENABLE") is not None:
            init_args = {
                "save_dir": os.environ.get("PL_LOGGER_SAVE_DIR", "."),
                "name": os.environ.get("PL_LOGGER_NAME", "lightning_logs"),
                "version": os.environ.get("PL_LOGGER_VERSION"),
            }

            default_loggers = [
                {"class_path": "TensorBoardLogger", "init_args": init_args},
                {"class_path": "CSVLogger", "init_args": init_args},
            ]
            defaults["trainer.logger"] = default_loggers

        # saves last checkpoints based on "step" metric
        # as well as "val_loss" metric
        init_args_last = {
            "monitor": "step",
            "mode": "max",
            "filename": "last-{step:02d}",
            "save_last": True,
            "auto_insert_metric_name": False,
        }
        init_args_best = {
            "monitor": "val_loss",
            "mode": "min",
            "filename": "best-{step:02d}",
            "auto_insert_metric_name": False,
        }
        default_callbacks = [
            {
                "class_path": "ModelCheckpoint",
                "init_args": init_args_last,
            },
            {
                "class_path": "ModelCheckpoint",
                "init_args": init_args_best,
            },
        ]
        defaults["trainer.callbacks"] = default_callbacks
        parser.add_argument("--multiprocessing_sharing_strategy", default="", type=str)

        parser.set_defaults(defaults)

    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        """Parses command line arguments and stores it in ``self.config``."""

        super().parse_arguments(parser, args)

        # Pytorch_lightning has this strange behavior that it can not resume
        # simply from a checkpoint file because it doesn't use the hyperparameters
        # of the model to instantiate the model. Instead, it needs you to pass
        # exactly the same config that you used when you created the model. This
        # is really weird (at least for our case). So what we do here is: if the user
        # has provided a checkpoint file, use all the parameters of the checkpoint file
        # as defaults.

        # Check if ckpt_path is given in subcommand
        subcommand = self.config.subcommand
        # Get namespace for current subcommand
        config_ns = getattr(self.config, subcommand)
        # Get the path of the checkpoint
        ckpt_path = getattr(config_ns, "ckpt_path", None)

        # If there is a checkpoint path, we need to reparse the arguments, using the
        # checkpoint parameters as defaults.
        if ckpt_path is not None:
            defaults = self._config_from_ckpt(ckpt_path)

            subcommand_parser = self._parser(subcommand)

            # Arguments are defined as "k.subkey" in lightning. E.g. model.num_neighbours
            # So we need to convert the dict to those keys. Another problem is that some
            # arguments are linked and therefore "data.x" might not exist because it is linked
            # to "model.x". That's why we need to set the defaults one by one inside a try/except
            # block (I found no way to check if the argument is defined in the parser).
            with ActionTypeHint.allow_default_instance_context():
                for k in defaults:
                    for subkey in defaults[k]:
                        try:
                            subcommand_parser.set_defaults(
                                {f"{k}.{subkey}": defaults[k][subkey]}
                            )
                        except KeyError:
                            pass

            # We have set all the right defaults now! So we can reparse the arguments.
            super().parse_arguments(parser, args)

    def _parse_ckpt_path(self) -> None:
        # Newer Lightning reparses checkpoints with weights_only=True here.
        # Graph2Mat already loaded the trusted Python hyperparameters above,
        # preserving both data/model defaults and explicit CLI overrides.
        # Do not reread or overwrite that config, or change Torch's global policy.
        pass

    def before_instantiate_classes(self) -> None:
        # This is executed after config/argparser has been instanced
        # but before data and model has been instantiated.
        import torch.multiprocessing

        config_ns = getattr(self.config, self.config.subcommand)
        _autoconfigure_n_matrix_components(config_ns, stage=self.config.subcommand)
        _apply_selection_metric(config_ns)
        if config_ns.multiprocessing_sharing_strategy:
            assert (
                config_ns.multiprocessing_sharing_strategy
                in torch.multiprocessing.get_all_sharing_strategies()
            )
            torch.multiprocessing.set_sharing_strategy(
                config_ns.multiprocessing_sharing_strategy
            )

    def _config_from_ckpt(self, ckpt_path: str):
        # Load parameters from the checkpoint file.
        # We only need to load hyperparameters, not weights. Therefore, we don't care whether things were in
        # the GPU. The torch tensors might be located on GPU (or any other device), which is possibly not
        # available when we load the checkpoint. Map those tensors to CPU so that there are no loading errors.
        # Lightning checkpoints contain Python objects; only load trusted files.
        checkpoint = torch.load(
            ckpt_path, map_location=torch.device("cpu"), weights_only=False
        )
        checkpoint = sanitize_checkpoint(checkpoint)

        config = {}

        if os.environ.get("E3MAT_FROMCKPT_DATAPROC", "").lower() not in [
            "off",
            "false",
            "f",
            "no",
            "0",
        ]:
            # Extract the keys of the data module that control how to process the data.
            # And set them explicitly in the config.
            config["data"] = checkpoint["datamodule_hyper_parameters"]

        if os.environ.get("E3MAT_FROMCKPT_MODEL", "").lower() not in [
            "off",
            "false",
            "f",
            "no",
            "0",
        ]:
            # Extract the parameters that where used to instantiate the model in the first
            # place and set them in the config.
            config["model"] = checkpoint["hyper_parameters"]

        if os.environ.get("E3MAT_FROMCKPT_BASISTABLE", "").lower() not in [
            "off",
            "false",
            "f",
            "no",
            "0",
        ]:
            basis_table = checkpoint.get("basis_table")
            if basis_table:
                config.setdefault("model", {})["basis_table"] = basis_table
                config.setdefault("data", {})["basis_table"] = basis_table

        return config


class SaveConfigSkipBasisTableCallback(SaveConfigCallback):
    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
    ) -> None:
        # Make a shallow copy of config and overwrite basis_table argument
        # to not save basis_table to yaml which makes no sense
        config = copy.copy(config)
        if hasattr(config.data, "basis_table"):
            config.data.basis_table = None
            config.model.basis_table = None
        super().__init__(
            parser=parser,
            config=config,
            config_filename=config_filename,
            overwrite=overwrite,
            multifile=multifile,
        )
