"""Low-rank edge contracts: scalar oracle, O(3), radial zero and block symmetry."""

import math

import pytest
import torch
from e3nn import o3

from graph2mat import PointBasis
import graph2mat.bindings.e3nn as binding
from graph2mat.bindings.e3nn.modules.matrixblock import E3nnIrrepsMatrixBlock


def operation(irreps_in="2x0e+2x1o", irreps_out="2x0e+1x1o+1x2e", **kwargs):
    assert hasattr(binding, "E3nnLowRankEdgeBlock"), "7.1 readout not implemented"
    return binding.E3nnLowRankEdgeBlock(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        edge_feats_irreps=kwargs.pop("edge_feats_irreps", "2x0e"),
        **kwargs,
    ).double()


def test_scalar_formula_has_radial_gates_and_both_linear_and_bilinear_terms():
    # Catches a missing gate/linear branch, wrong TP pairing or a nonzero bias.
    model = operation("1x0e", "1x0e", edge_feats_irreps="1x0e", rank=1)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.left_projections[0].weight.fill_(2)
        model.right_projections[0].weight.fill_(3)
        model.output_linears[0].weight.fill_(5)
        model.linears[0].weight.fill_(7)
        model.radial_gates[0][0].weight[0, 0] = 1
        model.radial_gates[0][2].weight[0, 0] = 4
        model.radial_gates[0][2].weight[1, 0] = 6
    a, b, r = (
        torch.tensor([[value]], dtype=torch.float64) for value in (2.0, 3.0, 1.0)
    )
    value = model(edge_messages=(a, b), edge_feats=(r, r))
    silu = 1 / (1 + math.exp(-1))
    want = 180 * math.tanh(4 * silu) + 35 * math.tanh(6 * silu)
    torch.testing.assert_close(value, torch.tensor([[want]], dtype=torch.float64))
    # Independent radial dependence, not merely a constant learned pair weight.
    assert not torch.equal(
        value, model(edge_messages=(a, b), edge_feats=(r / 2, r / 2))
    )


@pytest.mark.parametrize("determinant", [1, -1])
@pytest.mark.parametrize("two_sources", [False, True])
def test_o3_with_repeated_segments_and_named_sources(determinant, two_sources):
    # Rotation/reflection covariance fails if gates use nonscalars or channels mix m.
    torch.manual_seed(7)
    inputs = [o3.Irreps("1x0e+2x1o+2x0e+1x2e")]
    if two_sources:
        inputs.append(o3.Irreps("2x0e+1x3o"))
    out_irreps = o3.Irreps("2x0e+1x1o+2x2e+1x3o+1x4e+1x6e")
    model = operation(inputs, out_irreps, rank=2)
    rotation = (
        determinant
        * o3.matrix_z(torch.tensor(0.37, dtype=torch.float64))
        @ o3.matrix_y(torch.tensor(0.81, dtype=torch.float64))
    )
    kwargs, rotated = {}, {}
    names = ["node_feats", "edge_messages"] if two_sources else ["edge_messages"]
    for name, irreps in zip(names, inputs):
        kwargs[name] = tuple(
            torch.randn(3, irreps.dim, dtype=torch.float64) for _ in range(2)
        )
        rotated[name] = tuple(
            value @ irreps.D_from_matrix(rotation).T for value in kwargs[name]
        )
    radial = torch.randn(3, 2, dtype=torch.float64)
    kwargs["edge_feats"] = rotated["edge_feats"] = (radial, radial)
    value = model(**kwargs)
    torch.testing.assert_close(
        model(**rotated),
        value @ out_irreps.D_from_matrix(rotation).T,
        atol=2e-8,
        rtol=2e-8,
    )
    # Keyword order must not swap source irreps.
    torch.testing.assert_close(
        model(**dict(reversed(list(kwargs.items())))), value, rtol=0, atol=0
    )


def test_zero_radial_removes_node_reuse_tail_and_is_continuous():
    torch.manual_seed(3)
    model = operation(["3x0e+1x1o", "1x0e"], rank=2)
    a, b = (torch.randn(4, 6, dtype=torch.float64) for _ in range(2))
    c, d = (torch.randn(4, 1, dtype=torch.float64) for _ in range(2))
    radial = torch.ones(4, 2, dtype=torch.float64)

    def predict(scale):
        return model(
            node_feats=(a, b),
            edge_messages=(c, d),
            edge_feats=(radial * scale, radial * scale),
        )

    assert torch.count_nonzero(predict(0)) == 0
    assert predict(1).abs().max() > 1e-5
    assert predict(1e-7).abs().max() < 1e-5
    assert predict(1e-7).norm() < predict(1e-5).norm()


@pytest.mark.parametrize("n", [0, 3])
def test_empty_and_nonempty_backward_are_finite(n):
    torch.manual_seed(9)
    model = operation(rank=2)
    a, b = (
        torch.randn(n, 8, dtype=torch.float64, requires_grad=True) for _ in range(2)
    )
    r = torch.randn(n, 2, dtype=torch.float64, requires_grad=True)
    value = model(edge_messages=(a, b), edge_feats=(r, r))
    assert value.shape == (n, 10)
    value.square().sum().backward()
    for tensor in (a, b, r, *model.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    if n:
        assert all(tensor.grad.abs().sum() > 0 for tensor in (a, b, r))
        assert any(param.grad.abs().sum() > 0 for param in model.parameters())


def test_backward_matches_finite_differences():
    torch.manual_seed(12)
    model = operation("1x0e+1x1o", "1x0e+1x1o", rank=1)
    a, b = (
        torch.randn(2, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)
    )
    r = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, y, z: model(edge_messages=(x, y), edge_feats=(z, z)), (a, b, r)
    )


@pytest.mark.parametrize("rank", [0, -1, True, 1.5, None])
def test_invalid_rank_rejected(rank):
    with pytest.raises(ValueError, match="rank"):
        operation(rank=rank)


@pytest.mark.parametrize("radials", ["1x1o", "1x0o", "", "1x0e+1x0o"])
def test_non_even_scalar_radials_rejected(radials):
    with pytest.raises(ValueError, match="0e"):
        operation(edge_feats_irreps=radials)


@pytest.mark.parametrize("inputs", [[], [""], ["1x0e"] * 3])
def test_invalid_input_sources_rejected(inputs):
    with pytest.raises(ValueError, match="input"):
        operation(inputs)


def test_input_shape_and_missing_or_unknown_source_errors():
    model = operation(rank=1)
    a, r = torch.ones(2, 8, dtype=torch.float64), torch.ones(2, 2, dtype=torch.float64)
    for kwargs in (
        {"edge_messages": (a, a)},
        {"edge_messages": (a, a[:, :1]), "edge_feats": (r, r)},
        {"edge_messages": (a, a), "edge_feats": (r, r[:, :1])},
        {"edge_messages": (a, a), "edge_feats": r},
        {"unknown": (a, a), "edge_feats": (r, r)},
        {"node_feats": (a, a), "edge_messages": (a, a), "edge_feats": (r, r)},
    ):
        with pytest.raises(
            (ValueError, TypeError), match="input|source|edge_feats|shape|tuple"
        ):
            model(**kwargs)


def test_unreachable_output_is_structural_zero():
    model = operation("2x0e", "1x3o", rank=1)
    a, r = torch.ones(2, 2, dtype=torch.float64), torch.ones(2, 2, dtype=torch.float64)
    assert torch.count_nonzero(model(edge_messages=(a, a), edge_feats=(r, r))) == 0


@pytest.mark.parametrize("components", [1, 2])
def test_homonuclear_exchange_and_coefficients(components):
    # The operation is ordered; the enclosing block must symmetrize its two calls.
    torch.manual_seed(8)
    cls = type(operation(rank=2))
    basis = PointBasis("A", R=2, basis="1x0e+1x1o+1x2e+1x3o")
    block = E3nnIrrepsMatrixBlock(
        i_basis=basis,
        j_basis=basis,
        symmetry="ij",
        symm_transpose=True,
        n_matrix_components=components,
        operation_cls=cls,
        irreps={
            "node_feats_irreps": None,
            "edge_messages_irreps": o3.Irreps("2x0e+2x1o+2x2e+2x3o"),
            "edge_feats_irreps": o3.Irreps("2x0e"),
        },
        rank=2,
    ).double()
    a, b = (torch.randn(3, 32, dtype=torch.float64) for _ in range(2))
    r, s = (torch.randn(3, 2, dtype=torch.float64) for _ in range(2))
    forward = block(edge_messages=(a, b), edge_feats=(r, s))
    reverse = block(edge_messages=(b, a), edge_feats=(s, r))
    torch.testing.assert_close(forward, reverse.transpose(1, 2), rtol=1e-12, atol=1e-12)
    coefficients = block.forward_coefficients(edge_messages=(a, b), edge_feats=(r, s))
    torch.testing.assert_close(
        block.coefficients_to_block(coefficients), forward, atol=1e-12, rtol=1e-12
    )
    assert forward.abs().max() > 1e-5


def test_rank_bottleneck_reduces_parameters_for_wide_features():
    # This is a fixture-specific capacity/cost check, not a speed or VRAM claim.
    irreps_in, irreps_out = "32x0e+32x1o+32x2e", "4x0e+4x1o+4x2e"
    low = operation(irreps_in, irreps_out, rank=2)
    full = binding.E3nnLinearEdgeBlock(irreps_in, irreps_out)
    assert sum(p.numel() for p in low.parameters()) < sum(
        p.numel() for p in full.parameters()
    )
