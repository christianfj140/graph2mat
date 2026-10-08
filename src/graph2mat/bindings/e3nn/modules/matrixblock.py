import inspect
import warnings
from typing import Dict, Sequence, Type

import torch
from e3nn import o3

from graph2mat import PointBasis
from graph2mat.bindings.e3nn.irreps_tools import ReducedTensorProducts
from graph2mat.bindings.torch import TorchMatrixBlock

__all__ = ["E3nnIrrepsMatrixBlock"]


def unreachable_irreps(irreps_in: Sequence[o3.Irreps], irreps_out: o3.Irreps) -> o3.Irreps:
    """Output irreps that no linear or bilinear path from the inputs can produce.

    Readout operations are linear/bilinear in their inputs, so an output irrep is
    only reachable if it is an input irrep or appears in the product of two of them.
    Unreachable irreps are silently predicted as zero.
    """
    inputs = {ir for irreps in irreps_in for _, ir in o3.Irreps(irreps)}
    reachable = set(inputs)
    for ir1 in inputs:
        for ir2 in inputs:
            reachable.update(ir1 * ir2)
    return o3.Irreps([(mul, ir) for mul, ir in o3.Irreps(irreps_out) if ir not in reachable])


class E3nnIrrepsMatrixBlock(TorchMatrixBlock):
    """Computes a matrix block by computing its irreps first."""

    def __init__(
        self,
        i_basis: PointBasis,
        j_basis: PointBasis,
        symmetry: str,
        operation_cls: Type,
        symm_transpose: bool = False,
        preprocessor=None,
        irreps: Dict[str, o3.Irreps] = {},
        n_matrix_components: int = 1,
        output_scaleshift: bool = False,
        **operation_kwargs,
    ):
        torch.nn.Module.__init__(self)

        i_irreps = i_basis.e3nn_irreps
        j_irreps = j_basis.e3nn_irreps

        self.i_irreps = i_irreps
        self.j_irreps = j_irreps

        self.n_matrix_components = n_matrix_components
        self.symm_transpose = symm_transpose
        self.setup_reduced_tp(i_irreps=i_irreps, j_irreps=j_irreps, symmetry=symmetry)

        operation_kwargs = {
            **self.get_init_kwargs(irreps, operation_cls),
            **operation_kwargs,
        }

        self.operation = operation_cls(**operation_kwargs)

        # Output = shift + scale * operation output, per irrep instance (fitted from data
        # with set_output_scaleshift). Buffers only exist when enabled, so checkpoints of
        # models without it keep loading.
        self.output_scaleshift = output_scaleshift
        if output_scaleshift:
            self.register_buffer("output_scale", torch.ones(self._irreps_out.dim))
            self.register_buffer("output_shift", torch.zeros(self._irreps_out.dim))

        irreps_in = [
            ir
            for ir in (irreps.get("node_feats_irreps"), irreps.get("edge_messages_irreps"))
            if ir is not None
        ]
        missing = unreachable_irreps(irreps_in, self._irreps_out) if irreps_in else None
        if missing is not None and missing.dim > 0:
            warnings.warn(
                f"Block ({i_irreps}) x ({j_irreps}): output irreps {missing.simplify()} "
                f"({missing.dim}/{self._irreps_out.dim} coefficients) cannot be produced "
                f"from the readout inputs {[str(ir) for ir in irreps_in]} and will always "
                "be zero. Increase the lmax of hidden_irreps / edge_hidden_irreps "
                "(with MACE, num_interactions=1 leaves only scalar node features)."
            )

    def get_summary(self):
        return f"{str(self.operation.__class__.__name__)}: ({self.i_irreps}) x ({self.j_irreps}) -> {self._irreps_out}"

    def setup_reduced_tp(self, i_irreps: o3.Irreps, j_irreps: o3.Irreps, symmetry: str):
        # Store the shape of the block.
        if self.n_matrix_components == 1:
            self.block_shape = (i_irreps.dim, j_irreps.dim)
        else:
            self.block_shape = (i_irreps.dim, j_irreps.dim, self.n_matrix_components)
        # And number of elements in the block.
        self.block_size = i_irreps.dim * j_irreps.dim * self.n_matrix_components

        # Understand the irreps out that we need in order to create the block.
        # The block is a i_irreps.dim X j_irreps.dim matrix, with possible symmetries that can
        # reduce the number of degrees of freedom. We indicate this to the ReducedTensorProducts,
        # which we only use as a helper.
        reduced_tp = ReducedTensorProducts(symmetry, i=i_irreps, j=j_irreps)
        self._single_irreps_out = reduced_tp.irreps_out
        self._irreps_out = self._single_irreps_out * self.n_matrix_components

        # We also store the change of basis, a matrix that will bring us from the irreps_out
        # to the actual matrix block that we want to calculate.
        self.register_buffer("change_of_basis", reduced_tp.change_of_basis)
        if self.symm_transpose:
            self.register_buffer("_coefficient_transpose_index", getattr(reduced_tp, "transpose_index", None), persistent=False)
            self.register_buffer("_coefficient_transpose_sign", getattr(reduced_tp, "transpose_sign", None), persistent=False)

    def _compute_coefficients(self, *args, **kwargs):
        coefficients = self.operation(*args, **kwargs)
        if self.output_scaleshift:
            coefficients = coefficients * self.output_scale + self.output_shift
        return coefficients

    def _irrep_instances(self):
        """Per coefficient: index of its irrep instance and whether it is a 0e scalar."""
        ids, scalar = [], []
        for mul, ir in self._irreps_out:
            for _ in range(mul):
                ids.extend([len(set(ids))] * ir.dim)
                scalar.extend([ir.l == 0 and ir.p == 1] * ir.dim)
        return torch.tensor(ids), torch.tensor(scalar)

    @torch.no_grad()
    def set_output_scaleshift(
        self, coefficients: torch.Tensor, shift_scalars: bool, rel_floor: float = 1e-2
    ):
        """Fits the output scale/shift from target coefficients of shape (n_blocks, dim).

        The shift is the mean of the 0e coefficients (only if ``shift_scalars``; a
        shift in any other irrep would break equivariance). The scale is the RMS of
        each irrep instance, shared by its 2l+1 components so that equivariance is
        kept, floored at ``rel_floor`` times the largest one so that almost-empty
        channels are not amplified.
        """
        ids, scalar = self._irrep_instances()
        ids, scalar = ids.to(coefficients.device), scalar.to(coefficients.device)
        shift = coefficients.mean(0) * scalar if shift_scalars else torch.zeros_like(coefficients[0])
        squares = ((coefficients - shift) ** 2).mean(0)
        n_instances = int(ids.max()) + 1
        rms = (
            torch.zeros(n_instances, dtype=squares.dtype, device=squares.device).index_add_(0, ids, squares)
            / torch.bincount(ids, minlength=n_instances)
        ).sqrt()
        floor = max(float(rms.max()) * rel_floor, 1e-12)
        self.output_scale.copy_(rms.clamp_min(floor)[ids])
        self.output_shift.copy_(shift)

    def coefficients_to_block(self, irreducible_out: torch.Tensor) -> torch.Tensor:
        if self.n_matrix_components > 1:
            irreducible_out = irreducible_out.reshape(
                irreducible_out.shape[0],
                self.n_matrix_components,
                self._single_irreps_out.dim,
            )
            return self.numpy.einsum(
                "nci,ixy->nxyc", irreducible_out, self.change_of_basis
            )

        # And convert it to the actual block of the matrix, using the change of basis
        # matrix stored on initialization.
        # n = number of nodes, i = dim of irreps, x = rows in block, y = cols in block
        return self.numpy.einsum("ni,ixy->nxy", irreducible_out, self.change_of_basis)

    def block_to_coefficients(self, block: torch.Tensor) -> torch.Tensor:
        """Project a matrix block back to its irreducible coefficients."""
        if block.ndim == 2:
            block = block.unsqueeze(0)

        if self.n_matrix_components > 1:
            coefficients = self.numpy.einsum(
                "ixy,nxyc->nci", self.change_of_basis.to(block), block
            )
            return coefficients.reshape(coefficients.shape[0], -1)

        return self.numpy.einsum("ixy,nxy->ni", self.change_of_basis.to(block), block)

    def _compute_block(self, *args, **kwargs):
        # Get the irreducible output
        irreducible_out = self._compute_coefficients(*args, **kwargs)
        return self.coefficients_to_block(irreducible_out)

    def forward_coefficients(self, *args, **kwargs) -> torch.Tensor:
        if self.symm_transpose == False:
            return self._compute_coefficients(*args, **kwargs)

        index = getattr(self, "_coefficient_transpose_index", None)
        if index is None:
            # shortcut: uncertified RTP maps reconstruct; certify a map before coefficient-only training.
            return self.block_to_coefficients(self.forward(*args, **kwargs))

        def transpose(coefficients):
            shaped = coefficients.reshape(coefficients.shape[0], self.n_matrix_components, self._single_irreps_out.dim)
            return (shaped[:, :, index] * self._coefficient_transpose_sign).reshape(
                coefficients.shape[0], self._irreps_out.dim
            )

        values = [*args, *kwargs.values()]
        if values and all(isinstance(value, tuple) and len(value) == 2 for value in values):
            def both(pair):
                return (torch.cat([pair[0], pair[1]]), torch.cat([pair[1], pair[0]]))

            n = values[0][0].shape[0]
            coefficients = self._compute_coefficients(
                *map(both, args), **{key: both(value) for key, value in kwargs.items()}
            )
            return (coefficients[:n] + transpose(coefficients[n:])) / 2

        forward = self._compute_coefficients(*args, **kwargs)
        back_args = [(value[1], value[0]) if isinstance(value, tuple) and len(value) == 2 else value for value in args]
        back_kwargs = {key: (value[1], value[0]) if isinstance(value, tuple) and len(value) == 2 else value
                       for key, value in kwargs.items()}
        return (forward + transpose(self._compute_coefficients(*back_args, **back_kwargs))) / 2

    def forward_with_coefficients(self, *args, **kwargs):
        """(block, coefficients) evaluating the operation once, not twice as
        calling ``forward`` and ``forward_coefficients`` separately would."""
        if self.symm_transpose == False:
            coefficients = self._compute_coefficients(*args, **kwargs)
            return self.coefficients_to_block(coefficients), coefficients

        block = self.forward(*args, **kwargs)
        return block, self.block_to_coefficients(block)

    def get_init_kwargs(self, irreps: Dict[str, o3.Irreps], operation_cls) -> dict:
        kwargs = {}
        op_sig = inspect.signature(operation_cls)

        irreps = {**irreps}

        irreps["irreps_in"] = [
            irrep
            for irrep in [
                irreps["node_feats_irreps"],
                irreps.get("edge_messages_irreps"),
            ]
            if irrep is not None
        ]
        irreps["irreps_out"] = self._irreps_out

        for k in op_sig.parameters:
            if k in irreps:
                kwargs[k] = irreps[k]

        return kwargs
