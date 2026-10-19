"""Equivariant readout block for time-reversal-symmetric spin-orbit matrices."""
from typing import Dict, Type

import torch
from e3nn import o3

from graph2mat import PointBasis
from graph2mat.core.data.basis import get_change_of_basis
from graph2mat.core.data.spin_orbit import SOC_TRANSPOSE_SIGN
from .matrixblock import E3nnIrrepsMatrixBlock

__all__ = ["E3nnSpinOrbitMatrixBlock"]


class E3nnSpinOrbitMatrixBlock(E3nnIrrepsMatrixBlock):
    """Block with the 4 TRS spin-orbit channels ``[h0, gx, gy, gz]`` on its last axis.

    ``h0`` is built as in `E3nnIrrepsMatrixBlock`. ``g`` comes from the reduced tensor
    product ``i x j x 1e`` (an axial vector), antisymmetric in ``(i, j)`` for blocks with
    ``"ij=ji"`` symmetry, and is returned in Cartesian axes (x, y, z) as the labels store it.
    The operation outputs the h0 irreps followed by the g irreps (not simplified).
    """

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
        fuse_homonuclear: bool = False,
        **operation_kwargs,
    ):
        if n_matrix_components != 4:
            raise ValueError(f"Spin-orbit blocks need n_matrix_components=4, got {n_matrix_components}.")
        if output_scaleshift:
            raise ValueError("output_scaleshift is not supported with spin-orbit blocks.")
        if fuse_homonuclear:
            raise ValueError("fuse_homonuclear is not supported with spin-orbit blocks.")
        super().__init__(
            i_basis=i_basis, j_basis=j_basis, symmetry=symmetry, operation_cls=operation_cls,
            symm_transpose=symm_transpose, preprocessor=preprocessor, irreps=irreps,
            n_matrix_components=n_matrix_components, **operation_kwargs,
        )

    def setup_reduced_tp(self, i_irreps: o3.Irreps, j_irreps: o3.Irreps, symmetry: str):
        # h0: block shape, `change_of_basis` and `_single_irreps_out` as in the parent.
        super().setup_reduced_tp(i_irreps=i_irreps, j_irreps=j_irreps, symmetry=symmetry)
        antisymmetric = symmetry == "ij=ji"
        if antisymmetric and i_irreps.dim == 1:
            # A 1x1 antisymmetric block has no freedom (e3nn's RTP fails on an empty space).
            g_irreps, q = o3.Irreps(""), torch.zeros(0, 1, 1, 3)
        else:
            rtp = o3.ReducedTensorProducts(
                "ijk=-jik" if antisymmetric else "ijk", i=i_irreps, j=j_irreps, k="1e"
            )
            g_irreps, q = rtp.irreps_out, rtp.change_of_basis
        # k lives in the data (basis convention) frame, where D(1e)(R) = R; labels are Cartesian.
        _, cob_inv = get_change_of_basis("cartesian", self.i_basis.basis_convention)
        q = torch.einsum("dxyk,ak->dxya", q, torch.as_tensor(cob_inv, dtype=q.dtype))
        self.register_buffer("change_of_basis_g", q)
        self._g_irreps_out = g_irreps
        self._irreps_out = self._single_irreps_out + g_irreps

    def coefficients_to_block(self, irreducible_out: torch.Tensor) -> torch.Tensor:
        n = self._single_irreps_out.dim
        h0 = torch.einsum("ni,ixy->nxy", irreducible_out[:, :n], self.change_of_basis)
        g = torch.einsum("ni,ixya->nxya", irreducible_out[:, n:], self.change_of_basis_g)
        return torch.cat([h0[..., None], g], dim=-1)

    def _transpose_block(self, block):
        # g changes sign under transposition: H_ji = H_ij^dagger with H = h0 + i g.sigma.
        return block.swapaxes(-3, -2) * block.new_tensor(SOC_TRANSPOSE_SIGN)

    def _no_coefficients(self, *args, **kwargs):
        raise NotImplementedError("coefficient outputs are not supported with spin-orbit blocks")

    forward_coefficients = forward_with_coefficients = block_to_coefficients = _no_coefficients
