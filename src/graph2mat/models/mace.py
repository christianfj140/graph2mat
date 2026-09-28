from copy import copy
from typing import Literal, Optional, Union

import torch
from e3nn import o3
from torch.utils.checkpoint import checkpoint
from mace.modules import MACE
from mace.modules.utils import get_edge_vectors_and_lengths

from graph2mat import Graph2Mat
from graph2mat.bindings.e3nn import E3nnGraph2Mat
from graph2mat.bindings.e3nn.modules.hamiltonian_context import (
    HamiltonianLocalContext,
)
from graph2mat.bindings.torch.data import TorchBasisMatrixData



def drop_last_mace_interaction(mace: MACE) -> MACE:
    """Removes the last interaction of a MACE model (and its product and readout).

    MACE builds its last interaction with scalar outputs only, which is right for
    energies but leaves a matrix readout without the l>0 features of the deepest
    layer. Building MACE with one extra interaction and dropping it keeps the full
    hidden irreps in every remaining layer, for any MACE version (including those
    without ``keep_last_layer_irreps``).
    """
    n = len(mace.interactions) - 1
    if n < 1:
        raise ValueError("MACE needs at least two interactions to drop the last one.")
    mace.interactions = mace.interactions[:n]
    mace.products = mace.products[:n]
    mace.readouts = mace.readouts[:n]
    mace.num_interactions.fill_(n)
    return mace

def mace_node_feats(mace: MACE, data, gradient_checkpointing: bool = False) -> torch.Tensor:
    """Node features of all MACE interactions, as ``MACE.forward(data)["node_feats"]``.

    ``MACE.forward`` always makes the positions require grad (for forces), so the
    backward pass also differentiates the edge vectors, spherical harmonics and radial
    features of every edge; it also computes the energy readouts. Neither is needed
    for a matrix readout. With ``gradient_checkpointing`` each interaction is
    recomputed in the backward pass instead of keeping its activations.
    """
    if hasattr(mace, "joint_embedding") or hasattr(mace, "pair_repulsion"):
        raise ValueError("mace_node_feats does not support MACE joint embeddings or pair repulsion.")
    node_attrs, edge_index = data["node_attrs"], data["edge_index"]
    vectors, lengths = get_edge_vectors_and_lengths(
        positions=data["positions"], edge_index=edge_index, shifts=data["shifts"]
    )
    edge_attrs = mace.spherical_harmonics(vectors)
    edge_feats, cutoff = mace.radial_embedding(lengths, node_attrs, edge_index, mace.atomic_numbers)

    node_feats = mace.node_embedding(node_attrs)
    all_feats = []
    for i, (interaction, product) in enumerate(zip(mace.interactions, mace.products)):

        def layer(node_feats, edge_attrs, edge_feats, cutoff, interaction=interaction,
                  product=product, first_layer=i == 0):
            node_feats, sc = interaction(
                node_attrs=node_attrs, node_feats=node_feats, edge_attrs=edge_attrs,
                edge_feats=edge_feats, edge_index=edge_index, cutoff=cutoff,
                first_layer=first_layer,
            )
            return product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)

        args = (node_feats, edge_attrs, edge_feats, cutoff)
        if gradient_checkpointing and torch.is_grad_enabled():
            # Reentrant: e3nn's TorchScript modules do not recompute the exact same
            # saved tensors, which the non-reentrant checkpoint requires. Every tensor
            # is an explicit argument so that gradients reach whatever produced it.
            node_feats = checkpoint(layer, *args, use_reentrant=True)
        else:
            node_feats = layer(*args)
        all_feats.append(node_feats)
    return torch.cat(all_feats, dim=-1)


def simplified_irreps_index(irreps: o3.Irreps) -> tuple[o3.Irreps, torch.Tensor]:
    """Irreps with equal irreps merged, and the feature index that reorders to them.

    Concatenated per-interaction features (e.g. ``20x0e+20x1o+20x0e+20x1o``) make every
    tensor product and linear layer work segment pair by segment pair. Grouping the
    segments of each irrep gives the same function class with far fewer instructions.
    """
    irreps = o3.Irreps(irreps)
    starts = [s.start for s in irreps.slices()]
    order = sorted(range(len(irreps)), key=lambda i: (irreps[i].ir, i))
    index = torch.cat([torch.arange(starts[i], starts[i] + irreps[i].dim) for i in order])
    return o3.Irreps([irreps[i] for i in order]).simplify(), index


class MatrixMACE(torch.nn.Module):
    """Model that wraps a MACE model to produce a matrix output.

    Parameters
    ----------
    mace :
        MACE model to wrap.
    readout_per_interaction :
        If ``True``, a separate readout is applied to the features of each
        message passing interaction.
        If ``False``, the features of all interactions are concatenated
        and passed to a single readout.
    readout_interaction_aggregation :
        Aggregation used when ``readout_per_interaction=True``. ``"sum"``
        matches the documented per-interaction readout semantics. ``"mean"``
        is available as an explicit legacy-compatible option.
    graph2mat_cls :
        Class of the graph2mat model to use for the readouts.
    return_coefficients :
        If ``True``, include operation-wise irreducible coefficients in the
        output dictionary.
    hamiltonian_local_context :
        If ``True``, append deterministic scalar local Hamiltonian context
        channels to node readout features and edge radial features. This is
        experimental and opt-in.
    mp_cutoff :
        If given, MACE only passes messages through edges shorter than this (the
        matrix edges are still all read out). The number of message passing edges
        scales as mp_cutoff^3 (^2 in 2D), and so does most of MACE's memory. MACE
        must have been built with ``r_max = mp_cutoff``.
    readout_radial_embedding :
        Radial embedding for the matrix edges; needed with ``mp_cutoff``.
    readout_fp32 :
        If ``True``, the Graph2Mat readout runs in fp32 even when the model runs
        under mixed precision (e.g. ``bf16-mixed``). MACE keeps the reduced precision.
    lean_mace :
        If ``True``, MACE only computes node features (see ``mace_node_feats``):
        no energy readouts and no gradients w.r.t. positions. Same output, less work.
        The output then has no MACE energy keys.
    gradient_checkpointing :
        If ``True``, MACE interactions and the matrix readout are recomputed in the
        backward pass instead of storing their activations (less VRAM, more time).
        Implies ``lean_mace``.
    simplify_node_irreps :
        If ``True``, the concatenated node features of all interactions are reordered so
        that equal irreps are contiguous (see ``simplified_irreps_index``). Same function
        class, fewer tensor-product instructions; only with a single readout.
    hamiltonian_local_context_kwargs :
        Keyword arguments passed to ``HamiltonianLocalContext`` when
        ``hamiltonian_local_context=True``.
    **kwargs :
        Additional keyword arguments to pass to ``graph2mat_cls`` for
        initialization.
    """

    def __init__(
        self,
        mace: MACE,
        readout_per_interaction: bool = False,
        readout_interaction_aggregation: Literal["sum", "mean"] = "sum",
        graph2mat_cls: type[Graph2Mat] = E3nnGraph2Mat,
        return_coefficients: bool = False,
        hamiltonian_local_context: Union[bool, HamiltonianLocalContext] = False,
        hamiltonian_local_context_kwargs: Optional[dict] = None,
        readout_fp32: bool = False,
        mp_cutoff: Optional[float] = None,
        readout_radial_embedding: Optional[torch.nn.Module] = None,
        lean_mace: bool = False,
        gradient_checkpointing: bool = False,
        simplify_node_irreps: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.mace = mace
        self.lean_mace = lean_mace or gradient_checkpointing
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing and (return_coefficients or hamiltonian_local_context):
            raise ValueError(
                "gradient_checkpointing only supports the node/edge labels outputs "
                "(no return_coefficients or hamiltonian_local_context)."
            )

        self.readout_per_interaction = readout_per_interaction
        if readout_interaction_aggregation not in {"sum", "mean"}:
            raise ValueError(
                "readout_interaction_aggregation must be 'sum' or 'mean', got "
                f"{readout_interaction_aggregation!r}."
            )
        self.readout_interaction_aggregation = readout_interaction_aggregation
        self.return_coefficients = return_coefficients
        self.readout_fp32 = readout_fp32
        self.mp_cutoff = mp_cutoff
        # Radial features of the matrix edges. With mp_cutoff, MACE's own radial
        # embedding only reaches mp_cutoff, so the readout needs one up to the full cutoff.
        self.readout_radial_embedding = readout_radial_embedding
        self.hamiltonian_local_context = self._init_hamiltonian_local_context(
            hamiltonian_local_context,
            hamiltonian_local_context_kwargs or {},
        )

        edge_hidden_irreps = kwargs.pop("edge_hidden_irreps", None)
        edge_message_mid_irreps = kwargs.pop("edge_message_mid_irreps", None)
        edge_feats_irreps = self._with_edge_context_irreps(
            self.mace.interactions[0].edge_feats_irreps
        )

        if self.readout_per_interaction:
            self.mace_inter_irreps = [
                o3.Irreps(inter.hidden_irreps) for inter in self.mace.interactions
            ]

            self.matrix_readouts = torch.nn.ModuleList(
                [
                    graph2mat_cls(
                        irreps=dict(
                            node_attrs_irreps=inter.node_attrs_irreps,
                            node_feats_irreps=self._with_node_context_irreps(
                                o3.Irreps(inter.hidden_irreps)
                            ),
                            edge_attrs_irreps=inter.edge_attrs_irreps,
                            edge_feats_irreps=self._with_edge_context_irreps(
                                inter.edge_feats_irreps
                            ),
                            edge_hidden_irreps=edge_hidden_irreps,
                            edge_message_mid_irreps=edge_message_mid_irreps,
                        ),
                        **kwargs,
                    )
                    for inter in self.mace.interactions
                ]
            )
        else:
            self.mace_inter_irreps = sum(
                [inter.hidden_irreps for inter in self.mace.interactions], o3.Irreps()
            )
            if simplify_node_irreps:
                self.mace_inter_irreps, index = simplified_irreps_index(self.mace_inter_irreps)
                self.register_buffer("node_feats_index", index, persistent=False)
            else:
                self.node_feats_index = None

            self.matrix_readouts = graph2mat_cls(
                irreps=dict(
                    node_attrs_irreps=self.mace.interactions[0].node_attrs_irreps,
                    node_feats_irreps=self._with_node_context_irreps(
                        self.mace_inter_irreps
                    ),
                    edge_attrs_irreps=self.mace.interactions[0].edge_attrs_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    edge_hidden_irreps=edge_hidden_irreps,
                    edge_message_mid_irreps=edge_message_mid_irreps,
                ),
                **kwargs,
            )

    def _init_hamiltonian_local_context(
        self,
        hamiltonian_local_context: Union[bool, HamiltonianLocalContext],
        kwargs: dict,
    ) -> Optional[HamiltonianLocalContext]:
        if hamiltonian_local_context is False or hamiltonian_local_context is None:
            if kwargs:
                raise ValueError(
                    "hamiltonian_local_context_kwargs were provided, but "
                    "hamiltonian_local_context is disabled."
                )
            return None

        if isinstance(hamiltonian_local_context, HamiltonianLocalContext):
            if kwargs:
                raise ValueError(
                    "hamiltonian_local_context_kwargs cannot be used when passing "
                    "an initialized HamiltonianLocalContext module."
                )
            return hamiltonian_local_context

        if hamiltonian_local_context is not True:
            raise TypeError(
                "hamiltonian_local_context must be a bool or "
                "HamiltonianLocalContext instance."
            )

        context_kwargs = {**kwargs}
        if "num_types" not in context_kwargs:
            if not hasattr(self.mace, "atomic_numbers"):
                raise ValueError(
                    "hamiltonian_local_context=True requires an explicit "
                    "hamiltonian_local_context_kwargs['num_types'] when the "
                    "MACE model does not expose atomic_numbers."
                )
            context_kwargs["num_types"] = len(self.mace.atomic_numbers)
        r_max = getattr(self.mace, "r_max", None)
        if "r_max" not in context_kwargs:
            if r_max is None:
                raise ValueError(
                    "hamiltonian_local_context=True requires an explicit "
                    "hamiltonian_local_context_kwargs['r_max'] when the MACE "
                    "model does not expose r_max."
                )
            context_kwargs["r_max"] = r_max

        return HamiltonianLocalContext(**context_kwargs)

    def _with_node_context_irreps(self, irreps: o3.Irreps) -> o3.Irreps:
        if self.hamiltonian_local_context is None:
            return o3.Irreps(irreps)
        return (
            o3.Irreps(irreps) + self.hamiltonian_local_context.node_context_irreps
        ).simplify()

    def _with_edge_context_irreps(self, irreps: o3.Irreps) -> o3.Irreps:
        if self.hamiltonian_local_context is None:
            return o3.Irreps(irreps)
        return (
            o3.Irreps(irreps) + self.hamiltonian_local_context.edge_context_irreps
        ).simplify()

    def _aggregate_interaction_tensors(
        self, values: list[torch.Tensor]
    ) -> torch.Tensor:
        if not values:
            raise ValueError("No per-interaction readout outputs were produced.")

        stacked = torch.stack(values, dim=0)
        if self.readout_interaction_aggregation == "sum":
            return stacked.sum(dim=0)
        if self.readout_interaction_aggregation == "mean":
            return stacked.mean(dim=0)

        raise RuntimeError(
            "Invalid readout_interaction_aggregation reached forward pass: "
            f"{self.readout_interaction_aggregation!r}."
        )

    def _aggregate_interaction_coefficients(
        self, coefficients: list[dict[str, torch.Tensor]], *, kind: str
    ) -> dict[str, torch.Tensor]:
        if not coefficients:
            return {}

        expected_keys = set(coefficients[0])
        for interaction_index, item in enumerate(coefficients[1:], start=1):
            if set(item) != expected_keys:
                raise ValueError(
                    f"Per-interaction {kind} coefficient keys do not match. "
                    f"interaction 0 keys={sorted(expected_keys)!r}, "
                    f"interaction {interaction_index} keys={sorted(item)!r}."
                )

        return {
            key: self._aggregate_interaction_tensors(
                [interaction_coefficients[key] for interaction_coefficients in coefficients]
            )
            for key in sorted(expected_keys)
        }

    def forward(
        self, data: TorchBasisMatrixData, compute_force: bool = False, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        data :
            Input data.
        compute_force :
            Passed directly to the ``compute_force`` argument of the MACE model.
        **kwargs :
            Additional keyword arguments to pass to the MACE
            model for the forward pass.

        Returns
        -------
        output :
            The output of the MACE model, with the additional keys "node_labels"
            and "edge_labels" containing the output of ``Graph2Mat``.
        """
        mp_data = self._message_passing_data(data)
        if self.lean_mace and not compute_force:
            mace_out = {"node_feats": mace_node_feats(self.mace, mp_data, self.gradient_checkpointing)}
        else:
            mace_out = self.mace(mp_data, compute_force=compute_force, **kwargs)
        if not self.readout_fp32:
            return self._checkpointed_readout(data, mace_out)

        # MACE may run in reduced precision (bf16/fp16 autocast), but matrix elements
        # need ~1e-4 relative precision (bf16 rounds a 10 eV element in steps of ~60 meV).
        node_feats = mace_out["node_feats"]
        with torch.autocast(device_type=node_feats.device.type, enabled=False):
            return self._checkpointed_readout(data, {**mace_out, "node_feats": node_feats.float()})

    def compile_submodules(self) -> None:
        """torch.compile, in place, the MACE layers and every readout operation.

        In place (``Module.compile``) keeps the state_dict keys, so checkpoints are the
        same as without compiling. The model should have been built with e3nn's
        TorchScript code generation disabled (``mace.tools.compile.disable_e3nn_codegen``),
        which dynamo cannot trace. Shapes change between batches, hence dynamic=True.

        Note: torch 2.11 runs ``nvcc --version`` on every compilation and only tolerates
        a missing nvcc; a PATH entry that cannot be searched (permission denied) makes
        the compilation fail with ``PermissionError: 'nvcc'``. Triton's bundled binaries
        (``triton/backends/nvidia/bin``) must also be executable.
        """
        modules = [*self.mace.interactions, *self.mace.products]
        readouts = self.matrix_readouts
        for readout in readouts if isinstance(readouts, torch.nn.ModuleList) else [readouts]:
            if readout.preprocessing_edges is not None:
                modules.append(readout.preprocessing_edges)
            blocks = [*readout.self_interactions, *readout.interactions.values()]
            modules += [block.operation for block in blocks if block is not None]
        for module in modules:
            module.compile(dynamic=True)

    def _checkpointed_readout(self, data: TorchBasisMatrixData, mace_out: dict) -> dict:
        if not (self.gradient_checkpointing and torch.is_grad_enabled()):
            return self._readout(data, mace_out)

        # Reentrant (see mace_node_feats): only tensor arguments and outputs.
        def labels(node_feats):
            out = self._readout(data, {**mace_out, "node_feats": node_feats})
            return out["node_labels"], out["edge_labels"]

        node_labels, edge_labels = checkpoint(labels, mace_out["node_feats"], use_reentrant=True)
        return {**mace_out, "node_labels": node_labels, "edge_labels": edge_labels}

    def _message_passing_data(self, data: TorchBasisMatrixData):
        """``data`` restricted to the edges shorter than mp_cutoff (all of them if None)."""
        if self.mp_cutoff is None:
            return data
        _, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        short = lengths.reshape(-1) < self.mp_cutoff
        mp_data = copy(data)
        mp_data["edge_index"] = data["edge_index"][:, short]
        mp_data["shifts"] = data["shifts"][short]
        return mp_data

    def _readout(self, data: TorchBasisMatrixData, mace_out: dict) -> dict:
        """Applies the Graph2Mat readout(s) to the MACE output."""
        # Compute edge feats and edge attrs from the modules in the mace model
        # (we can't access them from the model because they are not stored/outputted,
        # but they are very cheap to recompute)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.mace.spherical_harmonics(vectors)
        radial_embedding = self.readout_radial_embedding
        if radial_embedding is None:
            radial_embedding = self.mace.radial_embedding
        edge_feats = radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.mace.atomic_numbers
        )

        if isinstance(edge_feats, tuple):
            # From MACE 0.3.14, the radial embedding returns a tuple, with the second
            # element being the cutoff.
            edge_feats = edge_feats[0]

        node_feats = mace_out["node_feats"]
        if getattr(self, "node_feats_index", None) is not None:
            node_feats = node_feats[:, self.node_feats_index]
        hamiltonian_context = None
        if self.hamiltonian_local_context is not None:
            hamiltonian_context = self.hamiltonian_local_context(
                data=data,
                edge_vectors=vectors,
                edge_lengths=lengths,
            )
            node_feats = torch.cat([node_feats, hamiltonian_context.node], dim=-1)
            edge_feats = torch.cat([edge_feats, hamiltonian_context.edge], dim=-1)

        data_for_readout = copy(data)

        data_for_readout["edge_attrs"] = edge_attrs
        data_for_readout["edge_feats"] = edge_feats

        # data._edge_attrs_keys = (*data._edge_attrs_keys, "edge_attrs", "edge_feats")

        # Apply the readouts.
        if not self.readout_per_interaction:
            # Readout from the whole set of features
            readout_output = self.matrix_readouts(
                data=data_for_readout,
                node_feats=node_feats,
                return_coefficients=self.return_coefficients,
            )
            if self.return_coefficients:
                node_labels, edge_labels, coefficients = readout_output
            else:
                node_labels, edge_labels = readout_output
        else:
            # Go interaction by interaction and grab the features that each one produced
            # Apply the readout to each interaction and then sum them all.
            used = 0
            node_labels_list = []
            edge_labels_list = []
            node_coefficients_list = []
            edge_coefficients_list = []
            coefficient_metadata = {}
            for i, readout in enumerate(self.matrix_readouts):
                inter_dim = self.mace_inter_irreps[i].dim
                inter_node_feats = mace_out["node_feats"][:, used : used + inter_dim]
                used += inter_dim
                if hamiltonian_context is not None:
                    inter_node_feats = torch.cat(
                        [inter_node_feats, hamiltonian_context.node],
                        dim=-1,
                    )

                readout_output = readout(
                    data=data_for_readout,
                    node_feats=inter_node_feats,
                    return_coefficients=self.return_coefficients,
                )
                if self.return_coefficients:
                    if len(readout_output) != 3:
                        raise ValueError(
                            "return_coefficients=True with "
                            "readout_per_interaction=True requires every readout "
                            "to return (node_labels, edge_labels, coefficients)."
                        )
                    node_labels, edge_labels, readout_coefficients = readout_output
                    node_coefficients_list.append(readout_coefficients["node"])
                    edge_coefficients_list.append(readout_coefficients["edge"])
                    coefficient_metadata = readout_coefficients.get(
                        "metadata", coefficient_metadata
                    )
                else:
                    node_labels, edge_labels = readout_output

                node_labels_list.append(node_labels)
                edge_labels_list.append(edge_labels)

            node_labels = self._aggregate_interaction_tensors(node_labels_list)
            edge_labels = self._aggregate_interaction_tensors(edge_labels_list)
            if self.return_coefficients:
                coefficients = {
                    "node": self._aggregate_interaction_coefficients(
                        node_coefficients_list, kind="node"
                    ),
                    "edge": self._aggregate_interaction_coefficients(
                        edge_coefficients_list, kind="edge"
                    ),
                    "metadata": coefficient_metadata,
                }

        output = {**mace_out, "node_labels": node_labels, "edge_labels": edge_labels}
        if self.return_coefficients:
            output["node_coefficients"] = coefficients["node"]
            output["edge_coefficients"] = coefficients["edge"]
            output["coefficient_metadata"] = coefficients.get("metadata", {})
        if hamiltonian_context is not None:
            output["hamiltonian_node_context"] = hamiltonian_context.node
            output["hamiltonian_edge_context"] = hamiltonian_context.edge

        return output
