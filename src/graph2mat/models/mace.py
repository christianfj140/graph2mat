from copy import copy
from typing import Literal, Optional, Sequence, Union

import math

import torch
from e3nn import o3
from torch.utils.checkpoint import checkpoint
from mace.modules import MACE
from mace.modules.utils import get_edge_vectors_and_lengths

from graph2mat import Graph2Mat
from graph2mat.bindings.e3nn import E3nnGraph2Mat
from graph2mat.bindings.e3nn.modules.edge_operations import E3nnLowRankEdgeBlock
from graph2mat.bindings.e3nn.modules.hamiltonian_context import (
    HamiltonianLocalContext,
)
from graph2mat.bindings.e3nn.modules.magnetic import MagneticMomentLayer
from graph2mat.bindings.e3nn.modules.spin_orbit import E3nnSpinOrbitFullMatrixBlock, E3nnSpinOrbitMatrixBlock
from graph2mat.bindings.torch.data import TorchBasisMatrixData
from graph2mat.core.data.spin_orbit import SOC_FULL_TIME_REVERSAL_PARITY
from graph2mat.models.long_range import ewald_potentials, equilibrate_charges, label_element_atoms, neutralize



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

def _reentrant_checkpoint(function, *args):
    """``function(*args)`` through a reentrant checkpoint when some argument needs gradients.

    A reentrant checkpoint only differentiates its output if an input requires gradients:
    with frozen inputs (training stages that freeze MACE or its embedding) the parameters
    inside ``function`` would get none. Those calls run without the checkpoint.
    """
    if torch.is_grad_enabled() and any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
        return checkpoint(function, *args, use_reentrant=True)
    return function(*args)


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
        if gradient_checkpointing:
            # Reentrant: e3nn's TorchScript modules do not recompute the exact same
            # saved tensors, which the non-reentrant checkpoint requires. Every tensor
            # is an explicit argument so that gradients reach whatever produced it.
            node_feats = _reentrant_checkpoint(layer, *args)
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


def pair_envelope(r: torch.Tensor, r_cut: torch.Tensor, p: int) -> torch.Tensor:
    """MACE's polynomial cutoff with a per-edge range: 1 at r = 0, 0 (with p-1 zero
    derivatives) at r >= r_cut."""
    x = r / r_cut
    f = (1.0 - (p + 1) * (p + 2) / 2 * x**p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2))
    return f * (x < 1.0)


#: Hardness (eV per e^2) of every atom before training in long_range="qeq"; softplus keeps it > 0.
QEQ_HARDNESS = 10.0
QEQ_SOFTPLUS_SHIFT = math.log(math.expm1(QEQ_HARDNESS))


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
        experimental and opt-in. Not supported with ``E3nnLowRankEdgeBlock``:
        its radial gates require every edge feature to vanish at the cutoff.
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
    long_range :
        If ``True`` or ``"charges"`` (roadmap 8.4), every atom gets a learned charge (neutral
        per structure), the Ewald potential V of those charges is computed
        (``models.long_range``) and the output gets ``(V_i + V_j) / 2 * S_ij``, with S the
        labels' last component (so S must also be loaded for predictions). Starts at zero
        (no charges). The charges are electronic (an extra electron is +1), so that V is the
        potential energy of an electron and ``+V S`` its shift. ``"qeq"``: the charges come
        from charge equilibration instead, with a learned electronegativity and hardness per
        atom and the structure's total charge (``data.total_charge``, SIESTA's NetCharge, 0 if
        absent; positive when electrons are removed, so the electronic charges add up to
        ``-total_charge``), so they respond to the whole structure.
    long_range_smearing :
        Width (Angstrom) of the Gaussian charges: the potential is that of erf(r/s)/r.
    magnetic_moments :
        ``None`` (default), ``"collinear"`` or ``"vector"`` (roadmap 8.1, magnetism): per-atom
        magnetic moments (``data.magnetic_moments``) become an input. MACE still only sees the
        geometry; a ``MagneticMomentLayer`` (linear in m, over the readout graph) adds features
        to the node features of the readout. The readout runs with m and with -m and the two
        outputs are combined so that time reversal is exact: ``"vector"`` (axial moments, with
        ``matrix_block_cls=E3nnSpinOrbitFullMatrixBlock``) gives the even channels
        ``(F(m) + F(-m)) / 2`` and the odd ones ``(F(m) - F(-m)) / 2``
        (``SOC_FULL_TIME_REVERSAL_PARITY``); ``"collinear"`` (signed scalars,
        ``n_matrix_components=2``: up, down) gives ``up = (u(m) + d(-m)) / 2`` and
        ``down = (d(m) + u(-m)) / 2``. Only with a single readout and none of
        return_coefficients, hamiltonian_local_context, long_range or matrix_targets.
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
        pair_envelope_radii: Optional[Sequence[float]] = None,
        pair_envelope_p: int = 6,
        long_range: Union[bool, str] = False,
        long_range_smearing: float = 2.0,
        matrix_targets: Optional[list[str]] = None,
        matrix_target_spin: Optional[int] = None,
        magnetic_moments: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.magnetic_moments = magnetic_moments
        if magnetic_moments is not None:
            if magnetic_moments not in ("collinear", "vector"):
                raise ValueError(
                    f"magnetic_moments must be None, 'collinear' or 'vector', got {magnetic_moments!r}."
                )
            bad = [name for name, value in (
                ("readout_per_interaction", readout_per_interaction),
                ("return_coefficients", return_coefficients),
                ("hamiltonian_local_context", hamiltonian_local_context),
                ("long_range", long_range),
                ("matrix_targets", matrix_targets is not None),
            ) if value]
            if bad:
                raise ValueError(f"magnetic_moments does not support: {', '.join(bad)}.")
            if magnetic_moments == "vector" and not issubclass(
                kwargs.get("matrix_block_cls", object), E3nnSpinOrbitFullMatrixBlock
            ):
                raise ValueError(
                    "magnetic_moments='vector' needs matrix_block_cls=E3nnSpinOrbitFullMatrixBlock "
                    "(policy spin_orbit_full)."
                )
            if magnetic_moments == "collinear" and kwargs.get("n_matrix_components", 1) != 2:
                raise ValueError(
                    "magnetic_moments='collinear' needs n_matrix_components=2 (policy spin_h_only: up, down)."
                )

        self.matrix_target_layout = None
        if matrix_targets is not None:
            from graph2mat.core.data.multitarget import MatrixTargetLayout

            self.matrix_target_layout = MatrixTargetLayout(tuple(matrix_targets), matrix_target_spin)
            if long_range or hamiltonian_local_context:
                raise ValueError("multitarget matrix_targets do not support long_range or hamiltonian_local_context")
            if kwargs.get("n_matrix_components", 1) not in (1, self.matrix_target_layout.n_components):
                raise ValueError("n_matrix_components disagrees with multitarget semantic layout")
            kwargs["n_matrix_components"] = self.matrix_target_layout.n_components

        self.mace = mace
        # Roadmap 7.3: the readout radial features also vanish at each pair's own range
        # R_i + R_j (orbital radii per point type), not only at the global 2 max(R).
        if pair_envelope_radii is None:
            self.pair_envelope_radii = None
        else:
            self.register_buffer("pair_envelope_radii", torch.as_tensor(pair_envelope_radii, dtype=torch.get_default_dtype()))
        self.pair_envelope_p = pair_envelope_p
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
        edge_update = {k: kwargs.pop(k) for k in ("edge_update_layers", "edge_update_node_irreps") if k in kwargs}
        edge_feats_irreps = self._with_edge_context_irreps(
            self.mace.interactions[0].edge_feats_irreps
        )
        self.magnetic_layer = None

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
                            **edge_update,
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

            node_feats_irreps = self._with_node_context_irreps(self.mace_inter_irreps)
            if magnetic_moments is not None:
                self.magnetic_layer = MagneticMomentLayer(
                    magnetic_moments,
                    node_feats_irreps=node_feats_irreps,
                    edge_attrs_irreps=self.mace.interactions[0].edge_attrs_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    avg_num_neighbors=self.mace.interactions[0].avg_num_neighbors,
                )
                node_feats_irreps = node_feats_irreps + self.magnetic_layer.irreps_out

            self.matrix_readouts = graph2mat_cls(
                irreps=dict(
                    node_attrs_irreps=self.mace.interactions[0].node_attrs_irreps,
                    node_feats_irreps=node_feats_irreps,
                    edge_attrs_irreps=self.mace.interactions[0].edge_attrs_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    edge_hidden_irreps=edge_hidden_irreps,
                    edge_message_mid_irreps=edge_message_mid_irreps,
                    **edge_update,
                ),
                **kwargs,
            )

        if self.hamiltonian_local_context is not None and any(
            isinstance(module, E3nnLowRankEdgeBlock) for module in self.matrix_readouts.modules()
        ):
            raise ValueError(
                "E3nnLowRankEdgeBlock does not support hamiltonian_local_context: "
                "context edge features do not vanish at the radial cutoff."
            )

        self.long_range_smearing = long_range_smearing
        self.long_range_mode = {True: "charges", False: None}.get(long_range, long_range)
        if self.long_range_mode not in (None, "charges", "qeq"):
            raise ValueError(f"long_range must be a bool, 'charges' or 'qeq', not {long_range!r}")
        if self.long_range_mode:
            if self.readout_per_interaction or gradient_checkpointing:
                raise ValueError("long_range needs a single readout and no gradient_checkpointing.")
            if issubclass(kwargs.get("matrix_block_cls", object), (E3nnSpinOrbitMatrixBlock, E3nnSpinOrbitFullMatrixBlock)):
                raise ValueError("long_range needs h_and_overlap labels (S last), not spin-orbit channels.")
            # charges: q; qeq: electronegativity and hardness (around QEQ_HARDNESS eV per e^2).
            n_out = 1 if self.long_range_mode == "charges" else 2
            self.charge_readout = o3.Linear(self._with_node_context_irreps(self.mace_inter_irreps), o3.Irreps(f"{n_out}x0e"))
            torch.nn.init.zeros_(self.charge_readout.weight)  # no charges at the start (neutral structures)
        else:
            self.charge_readout = None

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
        self, data: TorchBasisMatrixData, compute_force: bool = False,
        coefficients_only: bool = False, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        data :
            Input data.
        compute_force :
            Passed directly to the ``compute_force`` argument of the MACE model.
        coefficients_only :
            Return only the matrix readout's coefficient dictionaries, not its
            labels. Default outputs/checkpoints are unchanged. Reentrant
            gradient checkpointing and long-range labels require matrix mode.
        **kwargs :
            Additional keyword arguments to pass to the MACE
            model for the forward pass.

        Returns
        -------
        output :
            The output of the MACE model, with the additional keys "node_labels"
            and "edge_labels" containing the output of ``Graph2Mat``.
        """
        if self.matrix_target_layout is not None:
            from graph2mat.bindings.torch.multitarget import validate_layout_signature

            if coefficients_only:
                raise ValueError("multitarget matrix_targets do not support coefficients_only")
            validate_layout_signature(data, self.matrix_target_layout)
        if coefficients_only and (self.gradient_checkpointing or self.charge_readout is not None):
            raise ValueError("coefficients_only is unsupported with gradient_checkpointing or long_range; use matrix mode.")
        if coefficients_only and self.magnetic_layer is not None:
            raise ValueError("coefficients_only is unsupported with magnetic_moments; use matrix mode.")
        mp_data = self._message_passing_data(data)
        if self.lean_mace and not compute_force:
            mace_out = {"node_feats": mace_node_feats(self.mace, mp_data, self.gradient_checkpointing)}
        else:
            mace_out = self.mace(mp_data, compute_force=compute_force, **kwargs)
        if not self.readout_fp32:
            output = self._checkpointed_readout(data, mace_out, coefficients_only)
        else:
            # Matrix elements need more precision than bf16's ~60 meV at 10 eV.
            node_feats = mace_out["node_feats"]
            with torch.autocast(device_type=node_feats.device.type, enabled=False):
                output = self._checkpointed_readout(data, {**mace_out, "node_feats": node_feats.float()}, coefficients_only)
        if self.matrix_target_layout is not None:
            output["matrix_predictions"] = {target: {
                "node_labels": output["node_labels"][:, columns],
                "edge_labels": output["edge_labels"][:, columns],
            } for target, columns in self.matrix_target_layout.slices.items()}
        return output

    def compile_submodules(self, mode: Optional[str] = None) -> None:
        """torch.compile, in place, the MACE layers and every readout operation.

        ``mode`` is torch.compile's (roadmap 6.8): None or "max-autotune-no-cudagraphs"
        (autotunes the kernels). "reduce-overhead" (CUDA graphs) is rejected: in the 6.8
        measurements it crashed with a segmentation fault (torch 2.11).

        In place (``Module.compile``) keeps the state_dict keys, so checkpoints are the
        same as without compiling. The model should have been built with e3nn's
        TorchScript code generation disabled (``mace.tools.compile.disable_e3nn_codegen``),
        which dynamo cannot trace. Shapes change between batches, hence dynamic=True.

        Note: torch 2.11 runs ``nvcc --version`` on every compilation and only tolerates
        a missing nvcc; a PATH entry that cannot be searched (permission denied) makes
        the compilation fail with ``PermissionError: 'nvcc'``. Triton's bundled binaries
        (``triton/backends/nvidia/bin``) must also be executable.
        """
        if mode == "reduce-overhead":
            raise ValueError('torch_compile mode "reduce-overhead" is not supported: it crashed with a '
                             "segmentation fault in the roadmap 6.8 measurements (torch 2.11).")
        modules = [*self.mace.interactions, *self.mace.products]
        readouts = self.matrix_readouts
        for readout in readouts if isinstance(readouts, torch.nn.ModuleList) else [readouts]:
            if readout.preprocessing_edges is not None:
                modules.append(readout.preprocessing_edges)
            blocks = [*readout.self_interactions, *readout.interactions.values()]
            modules += [block.operation for block in blocks if block is not None]
        for module in modules:
            module.compile(dynamic=True, mode=mode)

    def _checkpointed_readout(self, data: TorchBasisMatrixData, mace_out: dict, coefficients_only: bool = False) -> dict:
        if not (self.gradient_checkpointing and torch.is_grad_enabled()):
            return self._readout(data, mace_out, coefficients_only)

        # Reentrant (see mace_node_feats): only tensor arguments and outputs.
        def labels(node_feats):
            out = self._readout(data, {**mace_out, "node_feats": node_feats})
            return out["node_labels"], out["edge_labels"]

        node_labels, edge_labels = _reentrant_checkpoint(labels, mace_out["node_feats"])
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

    def _readout(self, data: TorchBasisMatrixData, mace_out: dict, coefficients_only: bool = False) -> dict:
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
        envelope = None
        if self.pair_envelope_radii is not None:
            types = data["point_types"]
            sender, receiver = data["edge_index"]
            pair_range = self.pair_envelope_radii[types[sender]] + self.pair_envelope_radii[types[receiver]]
            envelope = pair_envelope(lengths, pair_range[:, None], self.pair_envelope_p)
            edge_feats = edge_feats * envelope

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
            edge_context = hamiltonian_context.edge
            if envelope is not None:
                # Scalar channels like the radial features: they must vanish at the pair range too.
                edge_context = edge_context * envelope
            edge_feats = torch.cat([edge_feats, edge_context], dim=-1)

        data_for_readout = copy(data)

        data_for_readout["edge_attrs"] = edge_attrs
        data_for_readout["edge_feats"] = edge_feats
        data_for_readout["edge_vectors"] = vectors

        # data._edge_attrs_keys = (*data._edge_attrs_keys, "edge_attrs", "edge_feats")

        if self.magnetic_layer is not None:
            node_labels, edge_labels = self._time_reversal_readout(
                data, data_for_readout, node_feats, edge_attrs, edge_feats
            )
            return {**mace_out, "node_labels": node_labels, "edge_labels": edge_labels}

        # Apply the readouts.
        need_coefficients = self.return_coefficients or coefficients_only
        coefficient_kwargs = {"coefficients_only": True} if coefficients_only else {}
        if not self.readout_per_interaction:
            # Readout from the whole set of features
            readout_output = self.matrix_readouts(
                data=data_for_readout,
                node_feats=node_feats,
                return_coefficients=need_coefficients,
                **coefficient_kwargs,
            )
            if coefficients_only:
                coefficients = readout_output
            elif self.return_coefficients:
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
                    return_coefficients=need_coefficients,
                    **coefficient_kwargs,
                )
                if need_coefficients:
                    if not coefficients_only and len(readout_output) != 3:
                        raise ValueError(
                            "return_coefficients=True with "
                            "readout_per_interaction=True requires every readout "
                            "to return (node_labels, edge_labels, coefficients)."
                        )
                    if coefficients_only:
                        readout_coefficients = readout_output
                    else:
                        node_labels, edge_labels, readout_coefficients = readout_output
                    node_coefficients_list.append(readout_coefficients["node"])
                    edge_coefficients_list.append(readout_coefficients["edge"])
                    coefficient_metadata = readout_coefficients.get(
                        "metadata", coefficient_metadata
                    )
                else:
                    node_labels, edge_labels = readout_output

                if not coefficients_only:
                    node_labels_list.append(node_labels)
                    edge_labels_list.append(edge_labels)

            if not coefficients_only:
                node_labels = self._aggregate_interaction_tensors(node_labels_list)
                edge_labels = self._aggregate_interaction_tensors(edge_labels_list)
            if need_coefficients:
                coefficients = {
                    "node": self._aggregate_interaction_coefficients(
                        node_coefficients_list, kind="node"
                    ),
                    "edge": self._aggregate_interaction_coefficients(
                        edge_coefficients_list, kind="edge"
                    ),
                    "metadata": coefficient_metadata,
                }

        output = {**mace_out}
        if not coefficients_only:
            output.update(node_labels=node_labels, edge_labels=edge_labels)
        if need_coefficients:
            output["node_coefficients"] = coefficients["node"]
            output["edge_coefficients"] = coefficients["edge"]
            output["coefficient_metadata"] = coefficients.get("metadata", {})
        if hamiltonian_context is not None:
            output["hamiltonian_node_context"] = hamiltonian_context.node
            output["hamiltonian_edge_context"] = hamiltonian_context.edge
        if self.charge_readout is not None:
            output = self._long_range(data, node_feats, output)

        return output

    def _time_reversal_readout(self, data, data_for_readout, node_feats, edge_attrs, edge_feats):
        """Labels from the readout with m and with -m (same MACE features), combined so that
        time reversal is exact (see ``magnetic_moments``)."""
        moments = data.get("magnetic_moments")
        if moments is None:
            raise ValueError(
                f"MatrixMACE(magnetic_moments={self.magnetic_moments!r}) needs data with magnetic_moments "
                "(MatrixDataProcessor(magnetic_moments=...))."
            )
        magnetic = self.magnetic_layer(node_feats, moments, data["edge_index"], edge_attrs, edge_feats)
        # The layer is linear in m: -magnetic are the features of -m.
        (node_plus, edge_plus), (node_minus, edge_minus) = (
            self.matrix_readouts(data=data_for_readout, node_feats=torch.cat([node_feats, sign * magnetic], dim=-1))
            for sign in (1, -1)
        )
        if self.magnetic_moments == "vector":
            parity = node_plus.new_tensor(SOC_FULL_TIME_REVERSAL_PARITY)
            return (node_plus + parity * node_minus) / 2, (edge_plus + parity * edge_minus) / 2
        return (node_plus + node_minus.flip(-1)) / 2, (edge_plus + edge_minus.flip(-1)) / 2

    @torch.compiler.disable
    def _long_range(self, data: TorchBasisMatrixData, node_feats: torch.Tensor, output: dict) -> dict:
        """Adds (V_i + V_j) / 2 * S_ij to the H component of the output (roadmap 8.4)."""
        point_labels, edge_labels = data.get("point_labels"), data.get("edge_labels")
        if point_labels is None or point_labels.ndim != 2 or point_labels.shape[1] < 2:
            raise ValueError("long_range needs the overlap as the last label component "
                             "(matrix_component_policy='h_and_overlap'; for predictions, "
                             "MatrixDataModule(predict_labels=True)).")
        batch = data.get("batch")
        if batch is None:
            batch = torch.zeros(len(node_feats), dtype=torch.long, device=node_feats.device)
        n_structures = int(data["cell"].shape[0] // 3)
        raw = self.charge_readout(node_feats)
        if self.long_range_mode == "qeq":
            total = data.get("total_charge")
            # NetCharge > 0 means electrons removed: the electronic charges add up to -NetCharge.
            total = raw.new_zeros(n_structures) if total is None else -total.reshape(-1).to(raw.dtype)
            hardness = torch.nn.functional.softplus(raw[:, 1] + QEQ_SOFTPLUS_SHIFT)
            charges = equilibrate_charges(raw[:, 0], hardness, data["positions"], data["cell"], batch, total,
                                          self.long_range_smearing)
        else:
            charges = neutralize(raw[:, 0], batch, n_structures)
        potential = ewald_potentials(charges, data["positions"], data["cell"], batch, self.long_range_smearing)
        node_labels, edge_out = output["node_labels"], output["edge_labels"]
        node_atoms, edge_i, edge_j = (
            a.to(node_feats.device) for a in label_element_atoms(
                data["point_types"].cpu(), data["edge_types"].cpu(), data["edge_index"].cpu(),
                len(edge_out), self.matrix_readouts.basis_table))
        d_nodes = potential[node_atoms] * point_labels[:, -1]
        d_edges = 0.5 * (potential[edge_i] + potential[edge_j]) * edge_labels[:, -1]
        # Labels are [H..., S] (h_and_overlap, one H column per spin): every H column of the
        # output gets the shift; a predicted S column (the labels' last one) does not.
        n_h = point_labels.shape[1] - 1

        def add(out, d):
            if out.ndim == 1:
                return out + d
            mask = (torch.arange(out.shape[1], device=out.device) < n_h).to(out.dtype)
            return out + d[:, None] * mask
        return {**output, "node_labels": add(node_labels, d_nodes), "edge_labels": add(edge_out, d_edges),
                "node_charges": charges, "node_potential": potential}
