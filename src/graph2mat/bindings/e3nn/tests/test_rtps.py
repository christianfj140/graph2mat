import pytest
import torch
from e3nn import o3

from graph2mat.bindings.e3nn import ReducedTensorProducts


@pytest.fixture(
    params=[
        "1x0e",
        "1x1o",
        "2x0e + 1x2o",
        "2x0e + 2x2o",
        "1x0e + 1x1o + 1x2e",
        "1x0e + 2x3o",
        "2x2e",
        "1x0e + 1x1o + 2x2e",
        "2x0e + 2x1o + 2x2e + 2x3o",
    ]
)
def i_irreps(request):
    return request.param


@pytest.fixture(
    params=[
        "1x0e",
        "1x1o",
        "2x0e + 1x2o",
        "2x0e + 2x2o",
        "1x0e + 1x1o + 1x2e",
        "1x0e + 2x3o",
        "2x2e",
        "1x0e + 1x1o + 2x2e",
        "2x0e + 2x1o + 2x2e + 2x3o",
    ]
)
def j_irreps(request):
    return request.param


@pytest.fixture(params=["ij", "ij=ji"])
def formula(request):
    return request.param


def test_same_irreps(i_irreps, formula):
    fast_rtp = ReducedTensorProducts(formula, i=i_irreps, j=i_irreps)
    slow_rtp = o3.ReducedTensorProducts(formula, i=i_irreps, j=i_irreps)
    assert fast_rtp.irreps_out == slow_rtp.irreps_out
    assert torch.allclose(fast_rtp.change_of_basis, slow_rtp.change_of_basis)


def test_different_irreps(i_irreps, j_irreps):
    fast_rtp = ReducedTensorProducts("ij", i=i_irreps, j=j_irreps)
    slow_rtp = o3.ReducedTensorProducts("ij", i=i_irreps, j=j_irreps)
    assert fast_rtp.irreps_out == slow_rtp.irreps_out
    assert torch.allclose(fast_rtp.change_of_basis, slow_rtp.change_of_basis)


@pytest.mark.parametrize("formula,i,j", [("ij", "1o", "2e"), ("ij=ji", "2e", "2e")])
def test_cached_rtps_preserve_precision_after_default_dtype_changes(formula, i, j):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        warm = ReducedTensorProducts(formula, i=i, j=j)
        assert warm.change_of_basis.dtype == torch.float32

        torch.set_default_dtype(torch.float64)
        actual = ReducedTensorProducts(formula, i=i, j=j)
        expected = o3.ReducedTensorProducts(formula, i=i, j=j)
        assert actual.irreps_out == expected.irreps_out
        torch.testing.assert_close(actual.change_of_basis, expected.change_of_basis,
                                   rtol=1e-14, atol=1e-14)

        torch.set_default_dtype(torch.float32)
        actual = ReducedTensorProducts(formula, i=i, j=j)
        expected = o3.ReducedTensorProducts(formula, i=i, j=j)
        torch.testing.assert_close(actual.change_of_basis, expected.change_of_basis,
                                   rtol=1e-6, atol=1e-7)
    finally:
        torch.set_default_dtype(previous)
