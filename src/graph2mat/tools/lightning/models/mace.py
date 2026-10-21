import contextlib
import warnings
from typing import Literal, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from mace.modules import MACE, InteractionBlock, RealAgnosticResidualInteractionBlock
from mace.modules.blocks import RadialEmbeddingBlock
from mace.tools.compile import disable_e3nn_codegen

from graph2mat import BasisTableWithEdges
from graph2mat.bindings.e3nn import (
    E3nnEdgeMessageBlock,
    E3nnLinearEdgeBlock,
    E3nnSimpleEdgeBlock,
    E3nnSimpleNodeBlock,
    E3nnSpinOrbitFullMatrixBlock,
    E3nnSpinOrbitMatrixBlock,
)
from graph2mat.bindings.e3nn.modules.hamiltonian_context import (
    HamiltonianLocalContext,
)

from graph2mat.core.data.metrics import (
    OrbitalMatrixMetric,
    block_type_mae,
    block_type_mse,
    elementwise_mae,
    elementwise_mse,
)
from graph2mat.models.mace import MatrixMACE, drop_last_mace_interaction
from graph2mat.tools.lightning import LitBasisMatrixModel


def _cueq_config():
    from mace.modules.wrapper_ops import CUET_AVAILABLE, CuEquivarianceConfig

    if not CUET_AVAILABLE:
        raise ImportError("cueq=True needs cuequivariance-torch.")
    return CuEquivarianceConfig(enabled=True, layout="mul_ir", group="O3_e3nn", optimize_all=True)


class LitMACEMatrixModel(LitBasisMatrixModel):
    """Wrapper to use MACEMatrix in pytorch_lightning.

    ``edge_block_readout`` selects the edge operation; the optional
    ``edge_block_readout_kwargs`` forwards its constructor arguments (e.g.
    ``{"rank": 4}`` for ``E3nnLowRankEdgeBlock`` or the experimental
    ``E3nnSlaterKosterEdgeBlock``). The default stays unchanged.
    """

    model: MACE
    _magnetic_neighbours_restored = False  # set by on_load_checkpoint: fit must not derive over it

    def __init__(
        self,
        root_dir: str = ".",
        basis_files: Union[str, None] = None,
        basis_table: Union[BasisTableWithEdges, None] = None,
        no_basis: Optional[dict] = None,
        # r_max: float=3.0,
        num_bessel: int = 10,
        num_polynomial_cutoff: int = 3,
        max_ell: int = 3,
        interaction_cls: Type[InteractionBlock] = RealAgnosticResidualInteractionBlock,
        interaction_cls_first: Type[
            InteractionBlock
        ] = RealAgnosticResidualInteractionBlock,
        num_interactions: int = 2,
        keep_last_layer_irreps: bool = False,
        mp_cutoff: Optional[float] = None,
        # num_elements: int=2,
        hidden_irreps: Union[o3.Irreps, str] = "20x0e+20x1o+20x2e",
        edge_hidden_irreps: Union[o3.Irreps, str] = "4x0e+4x1o+4x2e",
        edge_message_mid_irreps: Union[o3.Irreps, str, int, None] = None,
        edge_update_layers: int = 0,
        # Roadmap 7.3: readout radial features that vanish at each pair's R_i + R_j.
        pair_envelope: bool = False,
        # Roadmap 8.4: learned charges ("charges"/True) or charge equilibration ("qeq") + Ewald
        # potential, (V_i + V_j) / 2 * S_ij added to H.
        long_range: Union[bool, str] = False,
        long_range_smearing: float = 2.0,
        edge_update_node_irreps: Union[o3.Irreps, str, int, None] = None,
        avg_num_neighbors: float = 1.0,
        # atomic_numbers: List[int],
        correlation: int = 1,
        # unique_atoms: Sequence[sisl.Atom],
        symmetric_matrix: bool = False,
        basis_grouping: Literal["point_type", "basis_shape", "max"] = "point_type",
        preprocessing_nodes: Optional[Type[torch.nn.Module]] = None,
        preprocessing_edges: Optional[Type[torch.nn.Module]] = E3nnEdgeMessageBlock,
        preprocessing_edges_reuse_nodes: bool = False,
        node_block_readout: Type[torch.nn.Module] = E3nnSimpleNodeBlock,
        # Tensor product + linear term (roadmap 1.7): the linear term of the edge blocks carries
        # the whole gain (w90 12.98 -> 4.52 meV; node-only 13.67). Checkpoints store their own.
        edge_block_readout: Type[torch.nn.Module] = E3nnLinearEdgeBlock,
        readout_per_interaction: bool = False,
        readout_interaction_aggregation: Literal["sum", "mean"] = "sum",
        return_coefficients: bool = False,
        coefficient_training: bool = False,
        hamiltonian_local_context: bool = False,
        hamiltonian_local_context_kwargs: Optional[dict] = None,
        readout_fp32: bool = False,
        # MACE node features only (no energy readouts, no position gradients): same
        # matrices and gradients, ~10% fewer kernels and less memory (roadmap 2.2).
        lean_mace: bool = True,
        gradient_checkpointing: bool = False,
        simplify_node_irreps: bool = False,
        cueq: bool = False,
        torch_compile: bool = False,
        # torch.compile mode (roadmap 6.8): None (default), "reduce-overhead" (CUDA graphs,
        # for batches of constant shape) or "max-autotune-no-cudagraphs".
        torch_compile_mode: Optional[str] = None,
        output_scaleshift: bool = False,
        n_matrix_components: int = 1,
        # Components the readout predicts, when fewer than the labels carry: 1 with (H, S)
        # labels and gauge_invariant_mse predicts H only (S is only the loss's label).
        n_output_components: Optional[int] = None,
        optim_wdecay: float = 5e-7,
        optim_amsgrad: bool = True,
        optim_lr: float = 1e-3,
        loss: Type[OrbitalMatrixMetric] = elementwise_mse,
        loss_kwargs: Optional[dict] = None,
        training_stages: Optional[list[dict]] = None,
        initial_node_feats: str = "OneHotZ",
        version: str = "new",
        edge_block_readout_kwargs: Optional[dict] = None,
        matrix_targets: Optional[list[str]] = None,
        matrix_target_spin: Optional[int] = None,
        matrix_target_weights: Optional[dict] = None,
        matrix_target_losses: Optional[dict] = None,
        selection_metric: Optional[str] = None,
        # Roadmap 8.1: TRS spin-orbit readout [h0, gx, gy, gz] (E3nnSpinOrbitMatrixBlock).
        # Needs data with matrix_component_policy="spin_orbit"; uncertified options fail.
        spin_orbit: bool = False,
        # Roadmap 8.1 (magnetism): nominal magnetic moments of the atoms as model input, with exact
        # time reversal. "vector" (+ spin_orbit=True): the 8 channels of policy spin_orbit_full;
        # "collinear": the 2 channels of policy spin_h_only. Needs data with the same moments.
        magnetic_moments: Optional[str] = None,
        # Roadmap 8.2: must equal the data's edge_cutoff_padding (Å); the CLI links them.
        # Without mp_cutoff, MACE also passes messages up to 2(maxR + p) (cost grows ~r^3);
        # set mp_cutoff = 2 * maxR to keep the old message-passing range.
        edge_cutoff_padding: float = 0.0,
    ):
        if not (np.isfinite(edge_cutoff_padding) and edge_cutoff_padding >= 0):
            raise ValueError(
                f"edge_cutoff_padding must be a finite float >= 0 (Å), got {edge_cutoff_padding!r}."
            )
        if magnetic_moments not in (None, "collinear", "vector"):
            raise ValueError(f"magnetic_moments must be None, 'collinear' or 'vector', got {magnetic_moments!r}.")
        if magnetic_moments == "vector" and not spin_orbit:
            raise ValueError("magnetic_moments='vector' needs spin_orbit=True (data policy spin_orbit_full).")
        if magnetic_moments == "collinear" and spin_orbit:
            raise ValueError("magnetic_moments='collinear' is the spin_h_only model: spin_orbit must be False.")
        if spin_orbit or magnetic_moments:
            channels = {"vector": 8, "collinear": 2}.get(magnetic_moments, 4)
            rejected = {
                # Without it on-site blocks use "ij" and each edge direction is predicted alone,
                # so H is not Hermitian (measured |Hk - Hk^H| = 0.54 on an untrained model).
                "symmetric_matrix=False (Hermiticity needs symmetric_matrix=True)": spin_orbit and not symmetric_matrix,
                "matrix_targets": matrix_targets is not None,
                "coefficient_training": coefficient_training,
                "return_coefficients": return_coefficients,
                "output_scaleshift": output_scaleshift,
                "long_range": long_range,
                "hamiltonian_local_context": hamiltonian_local_context,
                # The magnetic layer reads the single concatenated node feature vector.
                "readout_per_interaction": readout_per_interaction and magnetic_moments is not None,
                # Not exercised with the double pass (m, -m) of the moment models.
                "torch_compile": torch_compile and magnetic_moments is not None,
                "cueq": cueq and magnetic_moments is not None,
                "gradient_checkpointing": gradient_checkpointing and magnetic_moments is not None,
                "basis_grouping != 'point_type'": basis_grouping != "point_type",
                "n_output_components": n_output_components is not None,
                f"n_matrix_components not in (1, {channels})": n_matrix_components not in (1, channels),
                # ponytail: exact classes; SK, low-rank and other readouts assume h0-only blocks.
                "node_block_readout": node_block_readout is not E3nnSimpleNodeBlock,
                "edge_block_readout": edge_block_readout not in (E3nnSimpleEdgeBlock, E3nnLinearEdgeBlock),
            }
            bad = [name for name, value in rejected.items() if value]
            if bad:
                raise ValueError(f"{'magnetic_moments' if magnetic_moments else 'spin_orbit'} does not support: {', '.join(bad)}.")
            n_matrix_components = channels
        if spin_orbit:
            # The node TensorSquare of h0 + g has irreps up to l=5: with e3nn's TorchScript fx
            # codegen, the second no_grad forward (Lightning's sanity check) did not finish in
            # 40 min (Bi2Se3, 2026-10-20); without it the same step takes 0.06 s.
            # ponytail: process-wide e3nn default, set before any module is built; scope it per
            # module if non-SOC models in the same process need TorchScript.
            from e3nn import set_optimization_defaults

            set_optimization_defaults(jit_script_fx=False)
        if matrix_targets is not None:
            from graph2mat.core.data.multitarget import MatrixTargetLayout

            layout = MatrixTargetLayout(tuple(matrix_targets), matrix_target_spin)
            if n_matrix_components not in (1, layout.n_components):
                raise ValueError("n_matrix_components disagrees with multitarget layout")
            if n_output_components not in (None, layout.n_components):
                raise ValueError("n_output_components cannot omit multitarget components")
            n_matrix_components = layout.n_components
            matrix_targets = list(layout.targets)
            if long_range or hamiltonian_local_context or coefficient_training:
                raise ValueError("multitarget matrix_targets reject long_range, hamiltonian_local_context and coefficient_training")
            if any(isinstance(stage, dict) and ("loss" in stage or stage.get("loss_kwargs"))
                   for stage in (training_stages or [])):
                raise ValueError("multitarget matrix_targets reject single-target training stage losses")
            if loss is not elementwise_mse or loss_kwargs:
                raise ValueError("multitarget requires per-target matrix_target_losses, not global loss/loss_kwargs")
        elif matrix_target_weights is not None or matrix_target_losses is not None:
            raise ValueError("matrix target weights/losses require matrix_targets")
        if version != "new":
            # The old OrbitalMatrixMACE path is gone; the argument stays for saved checkpoints.
            raise ValueError(f"LitMACEMatrixModel only supports version='new', got {version!r}.")
        model_cls = MatrixMACE

        if output_scaleshift and readout_per_interaction:
            # Each per-interaction readout would be fitted to the full target and then summed.
            raise ValueError("output_scaleshift is not supported with readout_per_interaction.")

        if initial_node_feats.split() != ["OneHotZ"]:
            # The MACE node embedding is built for one-hot species attributes only.
            raise ValueError(
                "LitMACEMatrixModel only supports initial_node_feats='OneHotZ', "
                f"got {initial_node_feats!r}."
            )

        super().__init__(
            root_dir=root_dir,
            basis_files=basis_files,
            basis_table=basis_table,
            no_basis=no_basis,
            loss=loss,
            loss_kwargs=loss_kwargs,
            training_stages=training_stages,
            coefficient_training=coefficient_training,
            initial_node_feats=initial_node_feats,
            model_cls=model_cls,
            selection_metric=selection_metric,
        )
        self.save_hyperparameters()
        if spin_orbit or magnetic_moments:
            self._check_spin_orbit_losses()
        self.matrix_target_layout = layout if matrix_targets is not None else None
        if self.matrix_target_layout is not None:
            from graph2mat.bindings.torch.multitarget import MultitargetMatrixLoss

            self.default_loss_fn = MultitargetMatrixLoss(layout, weights=matrix_target_weights,
                                                        losses=matrix_target_losses)
            self.loss_fn = self.default_loss_fn

        if isinstance(hidden_irreps, str):
            hidden_irreps = o3.Irreps(hidden_irreps)
        if isinstance(edge_hidden_irreps, str):
            edge_hidden_irreps = o3.Irreps(edge_hidden_irreps)

        # Matrix edges reach (R_i + p) + (R_j + p) (MatrixDataProcessor.edge_cutoff_padding).
        matrix_r_max = (self.basis_table.maxR() + edge_cutoff_padding) * 2
        readout_radial_embedding = None
        if mp_cutoff is not None:
            if not 0 < mp_cutoff < matrix_r_max:
                raise ValueError(
                    f"mp_cutoff must be in (0, {matrix_r_max:.3f}) (the matrix graph cutoff)."
                )
            # MACE passes messages up to mp_cutoff; the matrix edges keep the full range.
            readout_radial_embedding = RadialEmbeddingBlock(
                r_max=matrix_r_max,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
            )

        # torch_compile: e3nn modules without TorchScript code generation (dynamo cannot
        # trace it); they are compiled when fitting starts (on_fit_start).
        with disable_e3nn_codegen() if torch_compile else contextlib.nullcontext():
            mace = MACE(
                r_max=matrix_r_max if mp_cutoff is None else mp_cutoff,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
                max_ell=max_ell,
                interaction_cls=interaction_cls,
                interaction_cls_first=interaction_cls_first,
                # MACE's last interaction only outputs scalars: build one more and drop it.
                num_interactions=num_interactions + int(keep_last_layer_irreps),
                num_elements=len(self.basis_table.basis),
                hidden_irreps=hidden_irreps,
                MLP_irreps=o3.Irreps("1x0e"),
                atomic_energies=torch.zeros(len(self.basis_table.basis)),
                avg_num_neighbors=avg_num_neighbors,
                atomic_numbers=torch.arange(len(self.basis_table.basis)),
                correlation=correlation,
                gate=None,
                # cuEquivariance kernels for the MACE backbone (roadmap 6.9). mul_ir layout
                # and e3nn conventions, so that the e3nn readout reads the same features.
                **({"cueq_config": _cueq_config()} if cueq else {}),
            )
            if keep_last_layer_irreps:
                mace = drop_last_mace_interaction(mace)

            self.init_model(
                mace=mace,
                **({"matrix_targets": matrix_targets, "matrix_target_spin": matrix_target_spin}
                   if matrix_targets is not None else {}),
                readout_per_interaction=readout_per_interaction,
                readout_interaction_aggregation=readout_interaction_aggregation,
                return_coefficients=return_coefficients,
                hamiltonian_local_context=hamiltonian_local_context,
                **({"pair_envelope_radii": np.broadcast_to(np.asarray(self.basis_table.R, dtype=float) + edge_cutoff_padding, (len(self.basis_table.basis),)).tolist(),
                    "pair_envelope_p": num_polynomial_cutoff} if pair_envelope else {}),
                **({"long_range": long_range, "long_range_smearing": long_range_smearing} if long_range else {}),
                hamiltonian_local_context_kwargs=hamiltonian_local_context_kwargs,
                readout_fp32=readout_fp32,
                lean_mace=lean_mace,
                gradient_checkpointing=gradient_checkpointing,
                simplify_node_irreps=simplify_node_irreps,
                mp_cutoff=mp_cutoff,
                readout_radial_embedding=readout_radial_embedding,
                unique_basis=self.basis_table.basis,
                **({"output_scaleshift": True} if output_scaleshift else {}),
                edge_hidden_irreps=edge_hidden_irreps,
                edge_message_mid_irreps=edge_message_mid_irreps,
                # Roadmap 7.2: residual edge updates with the receiver (E3nnEdgeMessageBlock).
                **({"edge_update_layers": edge_update_layers, "edge_update_node_irreps": edge_update_node_irreps}
                   if edge_update_layers else {}),
                symmetric=symmetric_matrix,
                preprocessing_nodes=preprocessing_nodes,
                preprocessing_edges=preprocessing_edges,
                preprocessing_edges_reuse_nodes=preprocessing_edges_reuse_nodes,
                node_operation=node_block_readout,
                edge_operation=edge_block_readout,
                **({"edge_operation_kwargs": dict(edge_block_readout_kwargs)}
                   if edge_block_readout_kwargs is not None else {}),
                basis_grouping=basis_grouping,
                n_matrix_components=n_output_components or n_matrix_components,
                **({"matrix_block_cls": E3nnSpinOrbitFullMatrixBlock if magnetic_moments else E3nnSpinOrbitMatrixBlock}
                   if spin_orbit else {}),
                **({"magnetic_moments": magnetic_moments} if magnetic_moments else {}),
            )

    def _check_spin_orbit_losses(self) -> None:
        # Other losses read channels as H/S (gauge) or project h0-only coefficients.
        name = "magnetic_moments" if self.hparams.get("magnetic_moments") else "spin_orbit"
        allowed = (elementwise_mse, elementwise_mae, block_type_mse, block_type_mae)
        losses = [self.default_loss_fn] + [
            self._resolve_metric(stage["loss"])(**(stage.get("loss_kwargs") or {}))
            for stage in self.training_stages_config if isinstance(stage, dict) and "loss" in stage
        ]
        for loss_fn in losses:
            if type(loss_fn) not in allowed or any(self._physical_weights(loss_fn)):
                raise ValueError(
                    f"{name} only supports {[m.__name__ for m in allowed]} without physical "
                    f"weights, got {type(loss_fn).__name__}."
                )

    def _check_data_agreement(self) -> None:
        # Data and model must agree: a plain model would fit [h0, gx, gy, gz] as four
        # unrelated scalars. Without a datamodule only the spin-orbit and magnetic models
        # fail (unverifiable).
        spin_orbit = self.hparams.get("spin_orbit", False)  # older checkpoints lack it
        moments = self.hparams.get("magnetic_moments")
        module = getattr(self.trainer, "datamodule", None)
        # Edges beyond the readout's radial range would get zero features (roadmap 8.2).
        padding = self.hparams.get("edge_cutoff_padding", 0.0)
        data_padding = getattr(module, "edge_cutoff_padding", 0.0)
        if module is not None and padding != data_padding:
            raise ValueError(
                f"model edge_cutoff_padding={padding} disagrees with data edge_cutoff_padding={data_padding}."
            )
        # The species order comes from separate basis listings (model: root_dir, data: maybe a
        # temporary copy): types must match one to one, or a species would get another's weights.
        model_table, data_table = getattr(self, "basis_table", None), getattr(module, "basis_table", None)
        if model_table is not None and data_table is not None and list(model_table.types) != list(data_table.types):
            raise ValueError(f"model basis types {list(model_table.types)} differ from data basis types "
                             f"{list(data_table.types)}.")
        policy = getattr(module, "matrix_component_policy", None)
        # long_range reads S from the labels' last column: only h_and_overlap has it there.
        if self.hparams.get("long_range") and module is not None and policy != "h_and_overlap":
            raise ValueError(f"long_range needs matrix_component_policy='h_and_overlap' data, not {policy!r}.")
        required = {(True, None): "spin_orbit", (True, "vector"): "spin_orbit_full",
                    (False, "collinear"): "spin_h_only"}.get((spin_orbit, moments))
        # A plain model only excludes the spin-orbit policies, whose channels it would misread.
        agrees = policy == required if required else policy not in ("spin_orbit", "spin_orbit_full")
        if not agrees and (module is not None or required):
            raise ValueError(
                f"model spin_orbit={spin_orbit}, magnetic_moments={moments!r} disagrees with data "
                f"matrix_component_policy={policy!r}."
            )
        data_moments = getattr(module, "magnetic_moments", None)
        if module is not None and data_moments != moments:
            raise ValueError(
                f"model magnetic_moments={moments!r} disagrees with data magnetic_moments={data_moments!r}."
            )
        if policy in ("spin_orbit", "spin_orbit_full") or data_moments:
            bad = [name for name in ("spectral_window", "physical_labels", "matrix_targets")
                   if getattr(module, name, None)]
            if bad:
                raise ValueError(
                    f"{'magnetic_moments' if data_moments else 'spin_orbit'} data do not support: {', '.join(bad)}."
                )

    def _normalise_magnetic_neighbours(self) -> None:
        """Sets ``avg_num_neighbors`` to the mean number of readout-graph neighbours of the training data.

        The magnetic layer divides its neighbour sum by MACE's ``avg_num_neighbors``, whose default 1.0
        would leave it ~100 times too large for a dense metal (Fe bcc). Only done while the option has
        its default: a stored, restored or explicit value is kept. The mean comes from up to 32 training
        graphs spread over the dataset and is stored in the hyperparameters (so checkpoints rebuild the
        same model; ``on_load_checkpoint`` restores it into a model built with the default).
        """
        if (self.hparams.get("magnetic_moments") is None or self.hparams.avg_num_neighbors != 1.0
                or self._magnetic_neighbours_restored):
            return
        dataset = getattr(getattr(self.trainer, "datamodule", None), "train_dataset", None)
        dataset = getattr(dataset, "parent_data", dataset)  # rotating pool -> its dataset
        if dataset is None or len(dataset) == 0:
            warnings.warn("No training dataset to derive avg_num_neighbors from: the magnetic layer keeps 1.0.")
            return
        graphs = [dataset[int(i)] for i in np.unique(np.linspace(0, len(dataset) - 1, min(len(dataset), 32)).astype(int))]
        mean = sum(g["edge_index"].shape[1] for g in graphs) / sum(len(g["positions"]) for g in graphs)
        if mean <= 0:
            warnings.warn("The training graphs have no edges: the magnetic layer keeps avg_num_neighbors=1.0.")
            return
        self._set_magnetic_neighbours(mean)

    def _set_magnetic_neighbours(self, value: float) -> None:
        """Sets ``avg_num_neighbors`` everywhere the model uses it: hparams, magnetic layer, MACE interactions."""
        self.hparams.avg_num_neighbors = value
        self.model.magnetic_layer.avg_num_neighbors = value
        for interaction in self.model.mace.interactions:
            interaction.avg_num_neighbors = value

    def on_load_checkpoint(self, checkpoint) -> None:
        # The derived avg_num_neighbors is not in the state dict: a checkpoint loaded into a model built with
        # the default (``ckpt_path``, a config saved before fit) must restore it, or predictions use 1.0.
        super().on_load_checkpoint(checkpoint)
        stored = checkpoint.get("hyper_parameters", {}).get("avg_num_neighbors")
        if self.hparams.get("magnetic_moments") is not None and stored is not None:
            self._set_magnetic_neighbours(stored)
            self._magnetic_neighbours_restored = True  # a later on_fit_start must not derive over it

    def setup(self, stage: str) -> None:
        # validate/test/predict never reach on_fit_start. The datamodule is already set up here.
        if getattr(self._trainer, "datamodule", None) is not None:
            self._check_data_agreement()
        super().setup(stage)

    def on_fit_start(self) -> None:
        self._check_data_agreement()  # fit without a datamodule too
        self._normalise_magnetic_neighbours()
        super().on_fit_start()
        if self.hparams.torch_compile:
            # 6x6 step: 138 -> 70 ms and -31% peak memory; ~90 s of compilation first.
            self.model.compile_submodules(mode=self.hparams.torch_compile_mode)

    def _parameter_group_modules(self) -> dict[str, torch.nn.Module]:
        # The magnetic layer is part of the readout path: stages that train or freeze the readout include it.
        groups = super()._parameter_group_modules()
        layer = getattr(self.model, "magnetic_layer", None)
        if layer is not None:
            for name in ("readout", "readouts", "matrix_readouts"):
                groups[name] = torch.nn.ModuleList([groups[name], layer])
        return groups

    def configure_optimizers(self):
        model = self.model

        # Some parameters of the optimizer saved in the module.
        weight_decay = self.hparams.optim_wdecay
        amsgrad = self.hparams.optim_amsgrad

        # Optimizer setup used by the developers of MACE:
        decay_interactions = {}
        no_decay_interactions = {}
        for name, param in model.mace.interactions.named_parameters():
            if "linear.weight" in name or "skip_tp_full.weight" in name:
                decay_interactions[name] = param
            else:
                no_decay_interactions[name] = param

        param_options = dict(
            params=[
                {
                    "name": "embedding",
                    "params": list(model.mace.node_embedding.parameters()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "interactions_decay",
                    "params": list(decay_interactions.values()),
                    "weight_decay": weight_decay,
                },
                {
                    "name": "interactions_no_decay",
                    "params": list(no_decay_interactions.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "products",
                    "params": list(model.mace.products.parameters()),
                    "weight_decay": weight_decay,
                },
                {
                    "name": "readouts",
                    "params": list(model.matrix_readouts.parameters())
                    + (list(model.charge_readout.parameters()) if getattr(model, "charge_readout", None) else [])
                    + (list(model.magnetic_layer.parameters()) if getattr(model, "magnetic_layer", None) else []),
                    "weight_decay": 0.0,
                },
            ],
            lr=self.hparams.optim_lr,
            amsgrad=amsgrad,
        )
        if isinstance(model.hamiltonian_local_context, HamiltonianLocalContext):
            context_params = list(model.hamiltonian_local_context.parameters())
            if context_params:
                param_options["params"].append(
                    {
                        "name": "hamiltonian_local_context",
                        "params": context_params,
                        "weight_decay": 0.0,
                    }
                )

        # A parameter outside every group would silently never be trained (e.g. a module
        # added by a newer MACE). MACE's own energy readouts are the only unused ones.
        covered = {id(p) for group in param_options["params"] for p in group["params"]}
        covered |= {id(p) for p in model.mace.readouts.parameters()}
        missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in covered]
        if missing:
            raise ValueError(f"Parameters in no optimizer group (never trained): {missing}")

        return torch.optim.Adam(**param_options)
