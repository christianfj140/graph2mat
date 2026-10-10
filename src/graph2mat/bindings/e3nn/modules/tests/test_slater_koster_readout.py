"""Independent orbital oracles for the optional SK readout."""

import importlib
import itertools

import pytest
import torch

from graph2mat import PointBasis
from e3nn import o3
from graph2mat.bindings.e3nn.irreps_tools import ReducedTensorProducts
import graph2mat.bindings.e3nn as binding
from graph2mat.bindings.e3nn.modules.matrixblock import E3nnIrrepsMatrixBlock


@pytest.fixture(autouse=True)
def double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def couplings(*args):
    name = "graph2mat.bindings.e3nn.modules.slater_koster"
    assert importlib.util.find_spec(name) is not None, "SK calibration missing"
    return importlib.import_module(name)._sk_couplings(*args)


def basis(l, name="A"):
    return PointBasis(name, R=2, basis=[(1, l, (-1) ** l)])


def shell_ranges(point):
    offset = 0
    for mul, ir in point.e3nn_irreps:
        for _ in range(mul):
            yield offset, offset + ir.dim, ir
            offset += ir.dim


def axial(la, lb, q, dtype):
    value = torch.zeros(2 * la + 1, 2 * lb + 1, dtype=dtype)
    value[la - q, lb - q] = 1
    value[la + q, lb + q] = 1
    return value


@pytest.mark.parametrize("la,lb", list(itertools.product(range(4), repeat=2)))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_each_unit_integral_matches_independent_rotated_orbitals(la, lb, dtype):
    # Wrong phase, parity selection or normalization breaks this orbital transport.
    torch.set_default_dtype(dtype)
    a, b = basis(la), basis(lb, "B")
    rtp = ReducedTensorProducts("ij", i=a.e3nn_irreps, j=b.e3nn_irreps)
    segments, transform = couplings(a, b, rtp.irreps_out, rtp.change_of_basis)[0]
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    assert torch.linalg.matrix_rank(transform) == min(la, lb) + 1
    for angles in ((0., 0., 0.), (.31, .73, -.29), (1.1, -1.4, .4)):
        R = o3.angles_to_matrix(*(torch.tensor(x) for x in angles))
        n = R @ torch.tensor([0., 1., 0.])
        for q in range(min(la, lb) + 1):
            expected = (a.e3nn_irreps.D_from_matrix(R) @ axial(la, lb, q, dtype)
                        @ b.e3nn_irreps.D_from_matrix(R).T)
            coefficients = torch.zeros(rtp.irreps_out.dim)
            for k, (start, stop, L) in enumerate(segments):
                coefficients[start:stop] = transform[k, q] * o3.spherical_harmonics(
                    L, n, normalize=True, normalization="component")
            actual = torch.einsum("k,kij->ij", coefficients, rtp.change_of_basis)
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_repeated_radial_shells_never_mix_their_integrals():
    # Grouping by l alone writes integrals into the wrong radial shell pair.
    a = PointBasis("A", R=2, basis="2x0e+2x1o+1x2e+1x3o")
    b = PointBasis("B", R=2, basis="1x0e+2x1o+1x3o")
    rtp = ReducedTensorProducts("ij", i=a.e3nn_irreps, j=b.e3nn_irreps)
    pairs = list(itertools.product(shell_ranges(a), shell_ranges(b)))
    groups = couplings(a, b, rtp.irreps_out, rtp.change_of_basis)
    assert len(groups) == len(pairs)
    for ((ra, ea, ia), (rb, eb, ib)), (segments, transform) in zip(pairs, groups):
        for q in range(min(ia.l, ib.l) + 1):
            coeff = torch.zeros(rtp.irreps_out.dim)
            for k, (start, stop, L) in enumerate(segments):
                coeff[start:stop] = transform[k, q] * o3.spherical_harmonics(
                    L, torch.tensor([0., 1., 0.]), True, normalization="component")
            actual = torch.einsum("k,kij->ij", coeff, rtp.change_of_basis)
            expected = torch.zeros_like(actual)
            expected[ra:ea, rb:eb] = axial(ia.l, ib.l, q, expected.dtype)
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("broken", ["scale", "permute", "nan", "shape"])
def test_uncertified_q_is_rejected(broken):
    a, b = basis(1), basis(2)
    rtp = ReducedTensorProducts("ij", i=a.e3nn_irreps, j=b.e3nn_irreps)
    Q = rtp.change_of_basis.clone()
    if broken == "scale":
        Q *= 2
    elif broken == "permute":
        Q[[0, -1]] = Q[[-1, 0]]
    elif broken == "nan":
        Q[0, 0, 0] = float("nan")
    else:
        Q = Q[:-1]
    with pytest.raises(ValueError, match="Q|calibrat"):
        couplings(a, b, rtp.irreps_out, Q)


def sk_block(la=1, lb=1, components=1, inputs="2x0e+1x1o", **kwargs):
    assert hasattr(binding, "E3nnSlaterKosterEdgeBlock"), "SK operation missing"
    return E3nnIrrepsMatrixBlock(
        i_basis=basis(la), j_basis=basis(lb, "B"), symmetry="ij",
        operation_cls=binding.E3nnSlaterKosterEdgeBlock,
        irreps={"node_feats_irreps": o3.Irreps(inputs), "edge_feats_irreps": o3.Irreps("2x0e")},
        n_matrix_components=components, rank=1, **kwargs,
    )


def inputs_for(block, n=2):
    source = block.operation.residual.input_irreps[0]
    a, b = (torch.randn(n, source.dim) for _ in range(2))
    radial = torch.randn(n, 2)
    vectors = torch.randn(n, 3)
    return dict(node_feats=(a, b), edge_feats=(radial, radial), edge_vectors=(vectors, -vectors))


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("la,lb", [(0, 0), (0, 1), (1, 1)])
def test_literal_ss_sp_pp_integrals(la, lb, components):
    block = sk_block(la, lb, components)
    op = block.operation
    v = torch.tensor([[.3, -.2, .8], [-.4, .7, .1]])
    n = v / v.norm(dim=-1, keepdim=True)
    V = torch.arange(1., 1 + 2 * components * (min(la, lb) + 1)).reshape(2, components, -1)
    coeff = op._sk_coefficients(V, v)
    actual = block.coefficients_to_block(coeff)
    expected = []
    for c in range(components):
        if lb == 0:
            value = V[:, c, :1, None]
        elif la == 0:
            value = (V[:, c, :1] * n)[:, None, :]
        else:
            sigma, pi = V[:, c, 0], V[:, c, 1]
            value = pi[:, None, None] * torch.eye(3) + (sigma-pi)[:, None, None] * n[:, :, None] * n[:, None, :]
        expected.append(value)
    want = expected[0] if components == 1 else torch.stack(expected, -1)
    torch.testing.assert_close(actual, want, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("det", [1, -1])
def test_full_readout_o3_and_keyword_order(det):
    block = sk_block(2, 3, 2)
    data = inputs_for(block)
    R = det * o3.angles_to_matrix(*(torch.tensor(x) for x in (.3, .8, -.4)))
    rotated = {**data}
    source = block.operation.residual.input_irreps[0]
    rotated["node_feats"] = tuple(x @ source.D_from_matrix(R).T for x in data["node_feats"])
    rotated["edge_vectors"] = tuple(v @ R.T for v in data["edge_vectors"])
    value = block(**data)
    expected = torch.einsum("ai,nijc,bj->nabc", block.i_irreps.D_from_matrix(R), value,
                            block.j_irreps.D_from_matrix(R))
    torch.testing.assert_close(block(**rotated), expected, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(block(**dict(reversed(list(data.items())))), value, atol=0, rtol=0)


def test_both_branches_are_effective_and_radial_zero_kills_node_tail():
    torch.manual_seed(17)
    block = sk_block()
    op = block.operation
    data = inputs_for(block)
    full = op(**data)
    residual = op.residual(node_feats=data["node_feats"], edge_feats=data["edge_feats"])
    assert residual.abs().max() > 1e-5
    assert (full-residual).abs().max() > 1e-5
    zero = torch.zeros_like(data["edge_feats"][0])
    assert torch.count_nonzero(op(**{**data, "edge_feats": (zero, zero)})) == 0
    tiny = zero + 1e-9
    assert op(**{**data, "edge_feats": (tiny, tiny)}).abs().max() < 1e-7


def test_controlled_branch_ablation_retains_nonaxial_environment_residual():
    from copy import deepcopy
    torch.manual_seed(17)
    block = sk_block()
    data = inputs_for(block, 1)
    op = block.operation
    residual = op.residual(node_feats=data["node_feats"], edge_feats=data["edge_feats"])
    unit = torch.eye(op.n_integrals).reshape(op.n_integrals, 1, op.n_integrals)
    angular = op._sk_coefficients(unit, data["edge_vectors"][0].expand(op.n_integrals, -1)).T
    projection = angular @ torch.linalg.lstsq(angular, residual.T).solution
    assert (residual.T-projection).norm() > 1e-5
    sk_only, residual_only = deepcopy(op), deepcopy(op)
    with torch.no_grad():
        for p in sk_only.residual.parameters():
            p.zero_()
        for p in residual_only.sk_net.parameters():
            p.zero_()
    torch.testing.assert_close(residual_only(**data), residual, atol=0, rtol=0)
    assert sk_only(**data).norm() > 1e-5
    torch.testing.assert_close(sk_only(**data) + residual_only(**data), op(**data))


def test_two_sources_route_by_name_and_only_displaced_0e_condition_sk():
    torch.manual_seed(41)
    point = basis(1)
    node, edge = o3.Irreps("1x1o+1x0o+2x0e"), o3.Irreps("1x2e+1x0e")
    block = E3nnIrrepsMatrixBlock(point, point, "ij", binding.E3nnSlaterKosterEdgeBlock,
        irreps={"node_feats_irreps": node, "edge_messages_irreps": edge,
                "edge_feats_irreps": o3.Irreps("2x0e")}, rank=1)
    op = block.operation
    with torch.no_grad():
        for p in op.residual.parameters():
            p.zero_()
    vectors = torch.tensor([[.3, -.5, .7]])
    radial = torch.ones(1, 2)
    data = {"node_feats": tuple(torch.randn(1, node.dim) for _ in range(2)),
            "edge_messages": tuple(torch.randn(1, edge.dim) for _ in range(2)),
            "edge_feats": (radial, radial), "edge_vectors": (vectors, -vectors)}
    original = op(**data)
    torch.testing.assert_close(op(**dict(reversed(list(data.items())))), original, atol=0, rtol=0)
    changed = {k: tuple(x.clone() for x in pair) for k, pair in data.items()}
    for pair, stop in ((changed["node_feats"], 4), (changed["edge_messages"], 5)):
        for value in pair:
            value[:, :stop] += 3
    torch.testing.assert_close(op(**changed), original, atol=0, rtol=0)
    changed["node_feats"][0][:, 4:] += 2
    assert not torch.allclose(op(**changed), original)


@pytest.mark.parametrize("case", ["zero", "nan", "inf", "same", "shape", "complex", "dtype", "source_nan"])
def test_bad_geometry_and_inputs_fail_explicitly(case):
    block = sk_block()
    data = inputs_for(block)
    v = data["edge_vectors"][0].clone()
    if case in ("zero", "nan", "inf"):
        v[0] = {"zero": 0., "nan": float("nan"), "inf": float("inf")}[case]
    elif case == "shape":
        v = v[:, :2]
    elif case == "complex":
        v = v.to(torch.complex128)
    elif case == "dtype":
        v = v.float()
    elif case == "source_nan":
        data["node_feats"][0][0, 0] = float("nan")
    data["edge_vectors"] = (v, v if case == "same" else -v)
    with pytest.raises(ValueError):
        block(**data)


def test_empty_and_tiny_nonzero_vectors_do_not_get_clamped():
    block = sk_block(components=2)
    assert block(**inputs_for(block, 0)).shape == (0, 3, 3, 2)
    data = inputs_for(block)
    normal = block(**data)
    tiny = tuple(v * 1e-30 for v in data["edge_vectors"])
    torch.testing.assert_close(block(**{**data, "edge_vectors": tiny}), normal, atol=2e-12, rtol=2e-12)


def test_gradient_inputs_and_parameters_without_mutation():
    from torch.func import functional_call
    block = sk_block(0, 1, inputs="1x0e")
    op = block.operation
    a, b, r, v = (torch.randn(1, k, requires_grad=True) for k in (1, 1, 2, 3))
    def f(a, b, r, v):
        return op(node_feats=(a, b), edge_feats=(r, r), edge_vectors=(v, -v))
    assert torch.autograd.gradcheck(f, (a, b, r, v), atol=1e-5, rtol=1e-4)
    names, values = zip(*op.named_parameters())
    params = tuple(p.detach().clone().requires_grad_() for p in values)
    kwargs = dict(node_feats=(a, b), edge_feats=(r, r), edge_vectors=(v, -v))
    def parameter_f(*new):
        return functional_call(op, dict(zip(names, new)), (), kwargs)
    assert torch.autograd.gradcheck(parameter_f, params, atol=1e-5, rtol=1e-4, fast_mode=True)


def test_nonzero_output_shift_rejected_but_scaling_preserves_cutoff():
    block = sk_block(output_scaleshift=True)
    data = inputs_for(block)
    with pytest.raises(ValueError, match="shift"):
        block.set_output_scaleshift(torch.randn(4, block._irreps_out.dim), True)
    block.set_output_scaleshift(torch.randn(4, block._irreps_out.dim), False)
    zero = torch.zeros_like(data["edge_feats"][0])
    assert torch.count_nonzero(block(**{**data, "edge_feats": (zero, zero)})) == 0
    block.output_shift[0] = 1
    with pytest.raises(ValueError, match="shift"):
        block(**data)


@pytest.mark.parametrize("la,lb", [(0, 3), (1, 2), (2, 3), (3, 3)])
def test_transverse_frame_gauge_and_pole_continuity(la, lb):
    block = sk_block(la, lb, 2)
    op = block.operation
    V = torch.randn(1, 2, op.n_integrals)
    axis = torch.zeros(2*la+1, 2*lb+1, 2)
    for q in range(min(la, lb)+1):
        axis[la-q, lb-q] = V[0, :, q]
        axis[la+q, lb+q] = V[0, :, q]
    for angles in ((0., 0., 0.), (.3, .8, -.2), (.4, 3.141592653589, .6)):
        R = o3.angles_to_matrix(*(torch.tensor(x) for x in angles))
        n = R @ torch.tensor([0., 1., 0.])
        actual = block.coefficients_to_block(op._sk_coefficients(V, n[None]))[0]
        for phi in (.41, -.72):
            frame = R @ o3.matrix_y(torch.tensor(phi))
            expected = torch.einsum("ai,ijc,bj->abc", block.i_irreps.D_from_matrix(frame),
                                    axis, block.j_irreps.D_from_matrix(frame))
            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        nearby = n + torch.tensor([1e-9, 0., -1e-9])
        value = block.coefficients_to_block(op._sk_coefficients(V, nearby[None]))[0]
        assert (value-actual).abs().max() < 1e-7


def test_radial_envelope_value_and_derivative_at_cutoff_with_fixed_nodes():
    from graph2mat.models.mace import pair_envelope
    block = sk_block(output_scaleshift=True)
    block.set_output_scaleshift(torch.randn(4, block._irreps_out.dim), False)
    data = inputs_for(block, 1)
    values = []
    for radius in (1.9, 1.9999, 2., 2.1):
        r = torch.tensor([[radius]], requires_grad=True)
        radial = pair_envelope(r, r.new_tensor([[2.]]), 6).expand(-1, 2)
        out = block(**{**data, "edge_feats": (radial, radial)})
        derivative = torch.autograd.grad(out.sum(), r)[0]
        values.append(out.detach().abs().max())
        if radius >= 2:
            assert torch.count_nonzero(out) == 0
            assert torch.count_nonzero(derivative) == 0
        assert torch.isfinite(derivative).all()
    assert values[0] > 1e-10 and values[1] < 1e-8


@pytest.mark.parametrize("components", [0, 3, True])
def test_unsupported_components_fail_before_use(components):
    with pytest.raises(ValueError, match="C1/C2"):
        sk_block(components=components)


@pytest.mark.parametrize("bad_basis", ["1x1e", "1x4e"])
def test_unsupported_orbitals_fail_explicitly(bad_basis):
    with pytest.raises(ValueError, match="s/p/d/f|parity"):
        a = PointBasis("A", R=2, basis=bad_basis)
        E3nnIrrepsMatrixBlock(a, a, "ij", binding.E3nnSlaterKosterEdgeBlock,
                             irreps={"node_feats_irreps": o3.Irreps("1x0e"),
                                     "edge_feats_irreps": o3.Irreps("1x0e")})


def test_high_l_sk_coverage_warning_keeps_unreachable_residual_channels():
    with pytest.warns(UserWarning, match="always be zero") as recorded:
        block = sk_block(3, 3, inputs="1x0e")
    message = next(str(w.message) for w in recorded if "always be zero" in str(w.message))
    assert "1x1e+1x3e+1x5e" in message and "6e" not in message
    V = torch.zeros(1, 1, 4)
    V[..., 0] = 1
    c = block.operation._sk_coefficients(V, torch.tensor([[.3, .2, .5]]))
    slices = [s for s, (_, ir) in zip(block._irreps_out.slices(), block._irreps_out) if ir.l == 6]
    assert slices and c[:, slices[0]].abs().max() > 1e-3


def test_float32_subnormal_direction_and_noncontiguous_features_are_preserved():
    torch.set_default_dtype(torch.float32)
    block = sk_block()
    data = inputs_for(block)
    a, b = (torch.randn(2, 10)[:, ::2] for _ in range(2))
    data["node_feats"] = (a, b)
    snapshot = {k: tuple(x.clone() for x in pair) for k, pair in data.items()}
    expected = block(**data)
    scaled = tuple(v * 1e-20 for v in data["edge_vectors"])
    torch.testing.assert_close(block(**{**data, "edge_vectors": scaled}), expected,
                               atol=2e-5, rtol=2e-5)
    for key in data:
        for actual, original in zip(data[key], snapshot[key]):
            torch.testing.assert_close(actual, original, atol=0, rtol=0)
