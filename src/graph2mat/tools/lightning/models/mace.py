from typing import Literal, Optional, Type, Union

import torch
from e3nn import o3
from mace.modules import MACE, InteractionBlock, RealAgnosticResidualInteractionBlock
from mace.modules.blocks import RadialEmbeddingBlock

from graph2mat import BasisTableWithEdges
from graph2mat.bindings.e3nn import (
    E3nnEdgeMessageBlock,
    E3nnGraph2Mat,
    E3nnSimpleEdgeBlock,
    E3nnSimpleNodeBlock,
)
from graph2mat.bindings.e3nn.modules.hamiltonian_context import (
    HamiltonianLocalContext,
)

# from context import mace
from graph2mat.core.data.metrics import OrbitalMatrixMetric, elementwise_mse
from graph2mat.models.mace import MatrixMACE, drop_last_mace_interaction
from graph2mat.tools.lightning import LitBasisMatrixModel

# from graph2mat.models._mace.models import OrbitalMatrixMACE


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
        edge_block_readout: Type[torch.nn.Module] = E3nnSimpleEdgeBlock,
        readout_per_interaction: bool = False,
        readout_interaction_aggregation: Literal["sum", "mean"] = "sum",
        return_coefficients: bool = False,
        hamiltonian_local_context: bool = False,
        hamiltonian_local_context_kwargs: Optional[dict] = None,
        readout_fp32: bool = False,
        output_scaleshift: bool = False,
        n_matrix_components: int = 1,
        optim_wdecay: float = 5e-7,
        optim_amsgrad: bool = True,
        optim_lr: float = 1e-3,
        loss: Type[OrbitalMatrixMetric] = elementwise_mse,
        loss_kwargs: Optional[dict] = None,
        training_stages: Optional[list[dict]] = None,
        initial_node_feats: str = "OneHotZ",
        version: str = "new",
    ):
        model_cls = MatrixMACE  # if version == "new" else OrbitalMatrixMACE

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

        if version == "new":
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
            )
            if keep_last_layer_irreps:
                mace = drop_last_mace_interaction(mace)

            self.init_model(
                mace=mace,
                readout_per_interaction=readout_per_interaction,
                readout_interaction_aggregation=readout_interaction_aggregation,
                return_coefficients=return_coefficients,
                hamiltonian_local_context=hamiltonian_local_context,
                hamiltonian_local_context_kwargs=hamiltonian_local_context_kwargs,
                readout_fp32=readout_fp32,
                mp_cutoff=mp_cutoff,
                readout_radial_embedding=readout_radial_embedding,
                unique_basis=self.basis_table.basis,
                **({"output_scaleshift": True} if output_scaleshift else {}),
                edge_hidden_irreps=edge_hidden_irreps,
                edge_message_mid_irreps=edge_message_mid_irreps,
                symmetric=symmetric_matrix,
                preprocessing_nodes=preprocessing_nodes,
                preprocessing_edges=preprocessing_edges,
                preprocessing_edges_reuse_nodes=preprocessing_edges_reuse_nodes,
                node_operation=node_block_readout,
                edge_operation=edge_block_readout,
                basis_grouping=basis_grouping,
                n_matrix_components=n_matrix_components,
            )

        else:
            self.init_model(
                r_max=self.basis_table.maxR() * 2,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
                max_ell=max_ell,
                interaction_cls=interaction_cls,
                interaction_cls_first=interaction_cls_first,
                num_interactions=num_interactions,
                num_elements=len(self.basis_table.basis),
                hidden_irreps=hidden_irreps,
                edge_hidden_irreps=edge_hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                correlation=correlation,
                unique_basis=self.basis_table.basis,
                matrix_readout=E3nnGraph2Mat,
                return_coefficients=return_coefficients,
                symmetric_matrix=symmetric_matrix,
                node_block_readout=node_block_readout,
                edge_block_readout=edge_block_readout,
                only_last_readout=False,
                node_attr_irreps=self.initial_node_feats_irreps,
            )

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
                    "params": model.mace.node_embedding.parameters(),
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
                    "params": model.mace.products.parameters(),
                    "weight_decay": weight_decay,
                },
                {
                    "name": "readouts",
                    "params": model.matrix_readouts.parameters(),
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

        return torch.optim.Adam(**param_options)
