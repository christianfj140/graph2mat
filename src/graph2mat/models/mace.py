from copy import copy
from typing import Literal, Optional, Union

import torch
from e3nn import o3
from mace.modules import MACE
from mace.modules.utils import get_edge_vectors_and_lengths

from graph2mat import Graph2Mat
from graph2mat.bindings.e3nn import E3nnGraph2Mat
from graph2mat.bindings.e3nn.modules.hamiltonian_context import (
    HamiltonianLocalContext,
)
from graph2mat.bindings.torch.data import TorchBasisMatrixData


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
        **kwargs,
    ):
        super().__init__()

        self.mace = mace

        self.readout_per_interaction = readout_per_interaction
        if readout_interaction_aggregation not in {"sum", "mean"}:
            raise ValueError(
                "readout_interaction_aggregation must be 'sum' or 'mean', got "
                f"{readout_interaction_aggregation!r}."
            )
        self.readout_interaction_aggregation = readout_interaction_aggregation
        self.return_coefficients = return_coefficients
        self.hamiltonian_local_context = self._init_hamiltonian_local_context(
            hamiltonian_local_context,
            hamiltonian_local_context_kwargs or {},
        )

        edge_hidden_irreps = kwargs.pop("edge_hidden_irreps", None)
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

            self.matrix_readouts = graph2mat_cls(
                irreps=dict(
                    node_attrs_irreps=self.mace.interactions[0].node_attrs_irreps,
                    node_feats_irreps=self._with_node_context_irreps(
                        self.mace_inter_irreps
                    ),
                    edge_attrs_irreps=self.mace.interactions[0].edge_attrs_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    edge_hidden_irreps=edge_hidden_irreps,
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
        mace_out = self.mace(data, compute_force=compute_force, **kwargs)

        # Compute edge feats and edge attrs from the modules in the mace model
        # (we can't access them from the model because they are not stored/outputted,
        # but they are very cheap to recompute)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.mace.spherical_harmonics(vectors)
        edge_feats = self.mace.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.mace.atomic_numbers
        )

        if isinstance(edge_feats, tuple):
            # From MACE 0.3.14, the radial embedding returns a tuple, with the second
            # element being the cutoff.
            edge_feats = edge_feats[0]

        node_feats = mace_out["node_feats"]
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
