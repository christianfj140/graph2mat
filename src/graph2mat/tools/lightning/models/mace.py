import contextlib
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
)
from graph2mat.bindings.e3nn.modules.hamiltonian_context import (
    HamiltonianLocalContext,
)

from graph2mat.core.data.metrics import OrbitalMatrixMetric, elementwise_mse
from graph2mat.models.mace import MatrixMACE, drop_last_mace_interaction
from graph2mat.tools.lightning import LitBasisMatrixModel


def _cueq_config():
    from mace.modules.wrapper_ops import CUET_AVAILABLE, CuEquivarianceConfig

    if not CUET_AVAILABLE:
        raise ImportError("cueq=True needs cuequivariance-torch.")
    return CuEquivarianceConfig(enabled=True, layout="mul_ir", group="O3_e3nn", optimize_all=True)


class LitMACEMatrixModel(LitBasisMatrixModel):
    """Wrapper to use MACEMatrix in pytorch_lightning."""

    model: MACE

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
    ):
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
        )
        self.save_hyperparameters()

        if isinstance(hidden_irreps, str):
            hidden_irreps = o3.Irreps(hidden_irreps)
        if isinstance(edge_hidden_irreps, str):
            edge_hidden_irreps = o3.Irreps(edge_hidden_irreps)

        matrix_r_max = self.basis_table.maxR() * 2
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
                readout_per_interaction=readout_per_interaction,
                readout_interaction_aggregation=readout_interaction_aggregation,
                return_coefficients=return_coefficients,
                hamiltonian_local_context=hamiltonian_local_context,
                **({"pair_envelope_radii": np.broadcast_to(np.asarray(self.basis_table.R, dtype=float), (len(self.basis_table.basis),)).tolist(),
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
                basis_grouping=basis_grouping,
                n_matrix_components=n_output_components or n_matrix_components,
            )

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.hparams.torch_compile:
            # 6x6 step: 138 -> 70 ms and -31% peak memory; ~90 s of compilation first.
            self.model.compile_submodules(mode=self.hparams.torch_compile_mode)

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
                    + (list(model.charge_readout.parameters()) if getattr(model, "charge_readout", None) else []),
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
