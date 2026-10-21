"""Equivariant readout blocks for spin-orbit matrices (with and without time-reversal symmetry)."""
from typing import Dict, Type

import torch
from e3nn import o3

from graph2mat import PointBasis
from graph2mat.core.data.basis import get_change_of_basis
from graph2mat.core.data.spin_orbit import SOC_FULL_TRANSPOSE_SIGN, SOC_TRANSPOSE_SIGN
from .matrixblock import E3nnIrrepsMatrixBlock

__all__ = ["E3nnSpinOrbitMatrixBlock", "E3nnSpinOrbitFullMatrixBlock"]


def _axial_reduced_tp(formula: str, i_irreps: o3.Irreps, j_irreps: o3.Irreps, basis_convention: str):
    """Irreps and change of basis ``(d, i, j, 3)`` of ``i x j x 1e`` under ``formula``, with the
    axial vector index in Cartesian axes (x, y, z) as the labels store it."""
    if formula == "ijk=-jik" and i_irreps.dim == 1:
        # A 1x1 antisymmetric block has no freedom (e3nn's RTP fails on an empty space).
        irreps, q = o3.Irreps(""), torch.zeros(0, 1, 1, 3)
    else:
        rtp = o3.ReducedTensorProducts(formula, i=i_irreps, j=j_irreps, k="1e")
        irreps, q = rtp.irreps_out, rtp.change_of_basis
    # k lives in the data (basis convention) frame, where D(1e)(R) = R; labels are Cartesian.
    _, cob_inv = get_change_of_basis("cartesian", basis_convention)
    return irreps, torch.einsum("dxyk,ak->dxya", q, torch.as_tensor(cob_inv, dtype=q.dtype))


class E3nnSpinOrbitMatrixBlock(E3nnIrrepsMatrixBlock):
    """Block with the 4 TRS spin-orbit channels ``[h0, gx, gy, gz]`` on its last axis.

    ``h0`` is built as in `E3nnIrrepsMatrixBlock`. ``g`` comes from the reduced tensor
    product ``i x j x 1e`` (an axial vector), antisymmetric in ``(i, j)`` for blocks with
    ``"ij=ji"`` symmetry, and is returned in Cartesian axes (x, y, z) as the labels store it.
    The operation outputs the h0 irreps followed by the g irreps (not simplified).
    """

    #: Sign of each channel under transposition; its length is the number of channels.
    transpose_sign = SOC_TRANSPOSE_SIGN

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
        if n_matrix_components != len(self.transpose_sign):
            raise ValueError(
                f"{type(self).__name__} needs n_matrix_components={len(self.transpose_sign)}, "
                f"got {n_matrix_components}."
            )
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
        g_irreps, q = _axial_reduced_tp(
            "ijk=-jik" if symmetry == "ij=ji" else "ijk", i_irreps, j_irreps, self.i_basis.basis_convention
        )
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
        return block.swapaxes(-3, -2) * block.new_tensor(self.transpose_sign)

    def _no_coefficients(self, *args, **kwargs):
        raise NotImplementedError("coefficient outputs are not supported with spin-orbit blocks")

    forward_coefficients = forward_with_coefficients = block_to_coefficients = _no_coefficients


class E3nnSpinOrbitFullMatrixBlock(E3nnSpinOrbitMatrixBlock):
    """Block with the 8 spinor channels ``[a0, bx, by, bz, b0, ax, ay, az]`` on its last axis.

    The spinor block is ``sum_mu sigma_mu (x) (a^mu + i b^mu)`` (policy ``spin_orbit_full``), with
    no time-reversal symmetry imposed. ``a0`` and ``b`` are built as ``h0`` and ``g`` of
    `E3nnSpinOrbitMatrixBlock`. ``b0`` uses the ``h0`` basis but changes sign under
    transposition (antisymmetric for ``"ij=ji"`` blocks); ``a`` is the axial vector
    ``i x j x 1e`` like ``g`` but keeps its sign (symmetric for ``"ij=ji"`` blocks), in Cartesian
    axes. The operation outputs the ``a0``, ``b``, ``b0`` and ``a`` irreps, in that order.
    """

    transpose_sign = SOC_FULL_TRANSPOSE_SIGN

    def setup_reduced_tp(self, i_irreps: o3.Irreps, j_irreps: o3.Irreps, symmetry: str):
        super().setup_reduced_tp(i_irreps=i_irreps, j_irreps=j_irreps, symmetry=symmetry)
        hermitian = symmetry == "ij=ji"
        if not hermitian:
            # No (anti)symmetry within the block: b0 and a share the bases of a0 and b.
            b0_irreps, b0_q = self._single_irreps_out, self.change_of_basis
            a_irreps, a_q = self._g_irreps_out, self.change_of_basis_g
        else:
            if i_irreps.dim == 1:
                b0_irreps, b0_q = o3.Irreps(""), torch.zeros(0, 1, 1)
            else:
                rtp = o3.ReducedTensorProducts("ij=-ji", i=i_irreps, j=j_irreps)
                b0_irreps, b0_q = rtp.irreps_out, rtp.change_of_basis
            a_irreps, a_q = _axial_reduced_tp("ijk=jik", i_irreps, j_irreps, self.i_basis.basis_convention)
        self.register_buffer("change_of_basis_b0", b0_q)
        self.register_buffer("change_of_basis_a", a_q)
        self._b0_irreps_out, self._a_irreps_out = b0_irreps, a_irreps
        self._irreps_out = self._irreps_out + b0_irreps + a_irreps

    def coefficients_to_block(self, irreducible_out: torch.Tensor) -> torch.Tensor:
        n = self._single_irreps_out.dim + self._g_irreps_out.dim
        m = n + self._b0_irreps_out.dim
        even = super().coefficients_to_block(irreducible_out[:, :n])
        b0 = torch.einsum("ni,ixy->nxy", irreducible_out[:, n:m], self.change_of_basis_b0)
        a = torch.einsum("ni,ixya->nxya", irreducible_out[:, m:], self.change_of_basis_a)
        return torch.cat([even, b0[..., None], a], dim=-1)
