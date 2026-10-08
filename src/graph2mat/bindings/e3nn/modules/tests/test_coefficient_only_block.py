"""Symmetric coefficients are computed without allocating orbital matrix blocks."""

import pytest
import torch
from e3nn import o3

from graph2mat import PointBasis
from graph2mat.bindings.e3nn import E3nnDiagnosticDenseEdgeBlock, E3nnDiagnosticDenseNodeBlock
from graph2mat.bindings.e3nn.modules.matrixblock import E3nnIrrepsMatrixBlock


FEATURES = o3.Irreps("2x0e+1x1o+1x2e+1x3o")


def symmetric_block(basis, dtype, n_components, paired, symmetry="ij", scaleshift=True):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        point = PointBasis("A", R=2, basis=basis)
        return E3nnIrrepsMatrixBlock(
            point, point, symmetry=symmetry, symm_transpose=True,
            operation_cls=E3nnDiagnosticDenseEdgeBlock if paired else E3nnDiagnosticDenseNodeBlock,
            irreps={"node_feats_irreps": FEATURES}, n_matrix_components=n_components,
            output_scaleshift=scaleshift,
        )
    finally:
        torch.set_default_dtype(previous)


def no_dense_blocks(*args, **kwargs):
    raise AssertionError("Coefficient-only execution reconstructed a dense block.")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("basis", ["1x0e", "1x1o", "2x0e+1x1o", "1x0e+1x1o+1x2e+1x3o", "2x0e+2x1o+2x2e+2x3o"])
@pytest.mark.parametrize("n_components", [1, 2])
@pytest.mark.parametrize("paired", [True, False])
@pytest.mark.parametrize("symmetry", ["ij", "ij=ji"])
def test_coefficient_only_matches_dense_matrices_and_gradients(
    dtype, basis, n_components, paired, symmetry, monkeypatch,
):
    torch.manual_seed(37)
    block = symmetric_block(basis, dtype, n_components, paired, symmetry)
    k = block._irreps_out.dim
    block.set_output_scaleshift(torch.arange(1, 10*k+1, dtype=dtype).reshape(10, k) / (10*k), True)
    features = torch.randn(2, 3, FEATURES.dim, dtype=dtype, requires_grad=True)
    args = {"node_feats": (features[0], features[1]) if paired else features[0]}
    expected = block(**args)
    expected_gradient, = torch.autograd.grad(expected.square().sum(), features, retain_graph=True)

    monkeypatch.setattr(block, "coefficients_to_block", no_dense_blocks)
    coefficients = block.forward_coefficients(**args)
    single_dim = block._single_irreps_out.dim
    if n_components > 1:
        actual = torch.einsum("ncz,zij->nijc", coefficients.reshape(3, n_components, single_dim),
                              block.change_of_basis)
    else:
        actual = torch.einsum("nz,zij->nij", coefficients, block.change_of_basis)
    tolerance = 5e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    actual_gradient, = torch.autograd.grad(actual.square().sum(), features)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance)
    extra_buffers = [value for key, value in block.named_buffers(recurse=False)
                     if key not in ("change_of_basis", "output_scale", "output_shift")]
    assert all(value.ndim == 1 for value in extra_buffers)
    assert sum(value.numel() for value in extra_buffers) <= 2 * single_dim


@pytest.mark.parametrize("n_components", [1, 2])
def test_empty_coefficient_only_batches_keep_component_dimensions(n_components, monkeypatch):
    block = symmetric_block("1x0e+1x1o", torch.float64, n_components, True)
    empty = torch.empty(0, FEATURES.dim, dtype=torch.float64)
    monkeypatch.setattr(block, "coefficients_to_block", no_dense_blocks)
    actual = block.forward_coefficients(node_feats=(empty, empty))
    assert actual.shape == (0, block._irreps_out.dim)


def test_cast_and_old_state_dict_keep_matrix_predictions():
    torch.manual_seed(51)
    block = symmetric_block("2x0e+1x1o", torch.float32, 2, True).double()
    args = {"node_feats": (torch.randn(3, FEATURES.dim, dtype=torch.float64),
                           torch.randn(3, FEATURES.dim, dtype=torch.float64))}
    # Simulate a checkpoint made before derived transpose buffers existed.
    old_state = {key: value for key, value in block.state_dict().items() if "transpose" not in key}
    clone = symmetric_block("2x0e+1x1o", torch.float32, 2, True).double()
    clone.load_state_dict(old_state, strict=True)
    for model in (block, clone):
        coefficients = model.forward_coefficients(**args)
        actual = torch.einsum("ncz,zij->nijc", coefficients.reshape(3, 2, -1), model.change_of_basis)
        torch.testing.assert_close(actual, block(**args), atol=1e-12, rtol=1e-12)


def test_nonstandard_antisymmetric_formula_preserves_fallback():
    block = symmetric_block("1x1o", torch.float64, 1, False, "ij=-ji", scaleshift=False)
    args = {"node_feats": torch.randn(3, FEATURES.dim, dtype=torch.float64)}
    expected = torch.einsum("zij,nij->nz", block.change_of_basis, block(**args))
    torch.testing.assert_close(block.forward_coefficients(**args), expected, atol=1e-12, rtol=1e-12)


class GlobalEdgeReadout(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.linear = torch.nn.Linear(irreps_in[0].dim, irreps_out.dim)

    def forward(self, node_feats, offset):
        return self.linear(node_feats[0] + 2 * node_feats[1]) + offset


@pytest.mark.parametrize("positional", [False, True])
def test_global_arguments_keep_two_direction_evaluations(positional, monkeypatch):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        point = PointBasis("A", R=2, basis="1x0e+1x1o")
        block = E3nnIrrepsMatrixBlock(
            point, point, symmetry="ij", symm_transpose=True, n_matrix_components=2,
            operation_cls=GlobalEdgeReadout, irreps={"node_feats_irreps": FEATURES},
        )
    finally:
        torch.set_default_dtype(previous)
    pair = (torch.randn(3, FEATURES.dim, dtype=torch.float64),
            torch.randn(3, FEATURES.dim, dtype=torch.float64))
    args, kwargs = ((pair, 0.3), {}) if positional else ((), {"node_feats": pair, "offset": 0.3})
    expected = block(*args, **kwargs)
    monkeypatch.setattr(block, "coefficients_to_block", no_dense_blocks)
    sizes = []
    hook = block.operation.register_forward_hook(lambda module, args, result: sizes.append(len(result)))
    try:
        coefficients = block.forward_coefficients(*args, **kwargs)
    finally:
        hook.remove()
    actual = torch.einsum("ncz,zij->nijc", coefficients.reshape(3, 2, -1), block.change_of_basis)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert sizes == [3, 3]
