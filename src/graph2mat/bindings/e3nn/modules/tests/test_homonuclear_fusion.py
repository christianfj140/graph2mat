"""Weight fusion must preserve the original polynomial and its raw gradients."""

import pytest
import torch
import copy

from graph2mat.bindings.e3nn import E3nnLinearEdgeBlock
from graph2mat import PointBasis
from graph2mat.bindings.e3nn.modules.matrixblock import E3nnIrrepsMatrixBlock


FEATURES = "2x0e+1x1o+1x0e+1x2e+1x3o"


def make_block(basis="1x0e+1x1o", components=1, sources=1, dtype=torch.float64, **kwargs):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        point = PointBasis("A", R=2, basis=basis)
        irreps = {"node_feats_irreps": FEATURES}
        if sources == 2:
            irreps["edge_messages_irreps"] = FEATURES
        options = dict(symmetry="ij", symm_transpose=True, operation_cls=E3nnLinearEdgeBlock,
                       irreps=irreps, n_matrix_components=components)
        options.update(kwargs)
        return E3nnIrrepsMatrixBlock(point, point, **options)
    finally:
        torch.set_default_dtype(previous)


def explicit_coefficients(block, **pairs):
    """Independent oracle: two original operations and an orbital transpose."""
    forward = block.coefficients_to_block(block.operation(**pairs))
    reverse = block.coefficients_to_block(block.operation(**{k: (b, a) for k, (a, b) in pairs.items()}))
    matrix = 0.5 * (forward + reverse.transpose(1, 2))
    if block.n_matrix_components > 1:
        coefficients = torch.einsum("ixy,nxyc->nci", block.change_of_basis, matrix)
        coefficients = coefficients.reshape(matrix.shape[0], block._irreps_out.dim)
    else:
        coefficients = torch.einsum("ixy,nxy->ni", block.change_of_basis, matrix)
    return matrix, coefficients


def prepare(block):
    single = block._single_irreps_out.dim
    index = torch.cat([block._coefficient_transpose_index + c * single
                       for c in range(block.n_matrix_components)])
    sign = block._coefficient_transpose_sign.repeat(block.n_matrix_components)
    return block.operation._prepare_homonuclear_fusion(index, sign)


@pytest.fixture(scope="module", params=["1x0e", "1x0e+1x1o", "1x0e+1x1o+1x2e",
                                       "1x0e+1x1o+1x3o", "2x0e+2x1o+1x2e+1x3o"])
def orbital_basis(request):
    return request.param


@pytest.mark.parametrize("components,sources", [(1, 1), (2, 2)])
@pytest.mark.parametrize("n", [0, 1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_operation_equivalence_and_raw_gradients(orbital_basis, components, sources, n, dtype):
    torch.manual_seed(81)
    block = make_block(orbital_basis, components, sources, dtype)
    assert prepare(block) is None
    size = block.operation.tensor_products[0].irreps_in1.dim
    pairs = {key: tuple(torch.randn(n, size, dtype=dtype, requires_grad=True) for _ in range(2))
             for key in ("node_feats", "edge_messages")[:sources]}
    # Deliberately reverse keyword order; sources must still match their modules.
    pairs = dict(reversed(list(pairs.items())))
    _, expected = explicit_coefficients(block, **pairs)
    actual = block.operation(_homonuclear=True, **pairs)
    atol, rtol = (1e-12, 1e-10) if dtype == torch.float64 else (5e-6, 5e-5)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    inputs = [x for pair in pairs.values() for x in pair] + list(block.operation.parameters())
    cotangent = torch.randn_like(expected)
    reference_grad = torch.autograd.grad((expected * cotangent).sum(), inputs, allow_unused=True)
    actual_grad = torch.autograd.grad((actual * cotangent).sum(), inputs, allow_unused=True)
    for reference, result in zip(reference_grad, actual_grad):
        assert (reference is None) == (result is None)
        if reference is not None:
            torch.testing.assert_close(result, reference, atol=atol, rtol=rtol)


def test_raw_parameter_gradcheck():
    operation = E3nnLinearEdgeBlock("2x0e+1x1o", "1x0e+1x1e").double()
    assert operation._prepare_homonuclear_fusion(torch.arange(4), torch.tensor([1., -1., -1., -1.])) is None
    names, parameters = zip(*operation.named_parameters())
    a, b = (torch.randn(1, 5, dtype=torch.float64, requires_grad=True) for _ in range(2))

    def evaluate(a, b, *weights):
        return torch.func.functional_call(operation, dict(zip(names, weights)), (),
                                          {"_homonuclear": True, "node_feats": (a, b)})

    assert torch.autograd.gradcheck(evaluate, (a, b, *parameters), fast_mode=True)


@pytest.mark.parametrize("index,sign", [(torch.tensor([0, 0]), torch.ones(2)),
                                       (torch.arange(2), torch.tensor([1., 0.])),
                                       (torch.tensor([1, 0]), torch.tensor([1., -1.]))])
def test_rejected_map_clears_previous_certificate(index, sign):
    operation = E3nnLinearEdgeBlock("2x0e", "2x0e")
    assert operation._prepare_homonuclear_fusion(torch.arange(2), torch.ones(2)) is None
    assert operation._prepare_homonuclear_fusion(index, sign) is not None
    assert not operation._homonuclear_ready
    assert all(layer._homonuclear_index.numel() == 0 for layer in [*operation.tensor_products, *operation.linears])


def test_real_unequal_path_normalization_rejected():
    from e3nn import o3
    operation = E3nnLinearEdgeBlock("1x0e+1x1o", "1o")
    operation.tensor_products[0] = o3.FullyConnectedTensorProduct(
        "1x0e+1x1o", "1x0e+1x1o", "1o", path_normalization="path",
        in1_var=[1., 4.], in2_var=[1., 9.])
    reason = operation._prepare_homonuclear_fusion(torch.arange(3), torch.ones(3))
    assert "normalization" in reason


def test_cross_segment_or_magnetic_permutation_rejected():
    operation = E3nnLinearEdgeBlock("1o", "1o+1o")
    reason = operation._prepare_homonuclear_fusion(torch.tensor([3, 4, 5, 0, 1, 2]), torch.ones(6))
    assert "segments" in reason
    operation = E3nnLinearEdgeBlock("1o", "1o")
    assert operation._prepare_homonuclear_fusion(torch.tensor([1, 0, 2]), torch.ones(3)) is not None


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("api", ["forward", "forward_coefficients", "forward_with_coefficients"])
@pytest.mark.parametrize("n", [0, 3])
def test_block_apis_use_one_orientation(components, api, n):
    reference = make_block(components=components)
    block = make_block(components=components, fuse_homonuclear=True)
    block.load_state_dict(reference.state_dict(), strict=True)
    assert list(block.state_dict()) == list(reference.state_dict())
    assert [(k, tuple(v.shape)) for k, v in block.named_parameters()] == [
        (k, tuple(v.shape)) for k, v in reference.named_parameters()]
    size = block.operation.tensor_products[0].irreps_in1.dim
    pairs = {"node_feats": (torch.randn(n, size, dtype=torch.float64), torch.randn(n, size, dtype=torch.float64))}
    matrix, coefficients = explicit_coefficients(reference, **pairs)
    calls = []
    hooks = [layer.register_forward_pre_hook(lambda module, args: calls.append((type(module).__name__, args[0].shape[0])))
             for layer in block.operation.tensor_products]
    op_calls = []
    hooks.append(block.operation.register_forward_hook(lambda *args: op_calls.append(1)))
    try:
        result = getattr(block, api)(**pairs)
    finally:
        for hook in hooks:
            hook.remove()
    expected = (matrix, coefficients) if api == "forward_with_coefficients" else matrix if api == "forward" else coefficients
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-10)
    assert len(op_calls) == 1
    assert [rows for _, rows in calls] == [n]
    legacy_rows = []
    hook = reference.operation.tensor_products[0].register_forward_pre_hook(
        lambda module, args: legacy_rows.append(args[0].shape[0]))
    reference(**pairs)
    hook.remove()
    assert legacy_rows == [2 * n]


class NonlinearEdge(E3nnLinearEdgeBlock):
    def forward(self, **kwargs):
        return super().forward(**kwargs).tanh()


@pytest.mark.parametrize("options", [{"output_scaleshift": True}, {"symmetry": "ij=ji"},
                                    {"operation_cls": NonlinearEdge}])
def test_static_fallback_preserves_matrix(options):
    with pytest.warns(UserWarning, match="Homonuclear fusion disabled"):
        block = make_block(fuse_homonuclear=True, **options)
    reference = copy.deepcopy(block)
    reference.fuse_homonuclear = False
    reference._configure_homonuclear_fusion()
    if block.output_scaleshift:
        with torch.no_grad():
            block.output_scale.fill_(1.7)
            block.output_shift.fill_(0.2)
        reference.load_state_dict(block.state_dict())
    a, b = (torch.randn(3, block.operation.tensor_products[0].irreps_in1.dim, dtype=torch.float64) for _ in range(2))
    assert not block._fused_homonuclear
    torch.testing.assert_close(block(node_feats=(a, b)), reference(node_feats=(a, b)))


def test_heteronuclear_and_default_are_silent():
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        block = make_block(symm_transpose=False, fuse_homonuclear=True)
        default = make_block()
    assert not block._fused_homonuclear and not default._fused_homonuclear
    assert not any("Homonuclear" in str(w.message) for w in caught)


def test_loaded_q_recertification_and_deepcopy():
    block = make_block(fuse_homonuclear=True)
    original = copy.deepcopy(block.state_dict())
    clone = copy.deepcopy(block)
    state = copy.deepcopy(original)
    q = state["change_of_basis"]
    # Mix opposite transpose eigenvectors, within copies of the same 1o irrep.
    start = 0
    for mul, ir in block._single_irreps_out:
        if str(ir) == "1o" and mul >= 2:
            break
        start += mul * ir.dim
    rotation = torch.eye(q.shape[0], dtype=q.dtype)
    cosine, sine = torch.cos(torch.tensor(0.3)), torch.sin(torch.tensor(0.3))
    for m in range(3):
        i, j = start + m, start + 3 + m
        rotation[i, i] = rotation[j, j] = cosine
        rotation[i, j], rotation[j, i] = sine, -sine
    state["change_of_basis"] = torch.einsum("ij,jxy->ixy", rotation, q)
    with pytest.warns(UserWarning, match="loaded Q"):
        clone.load_state_dict(state)
    assert not clone._fused_homonuclear and block._fused_homonuclear
    reference = make_block()
    reference.load_state_dict(state)
    size = block.operation.tensor_products[0].irreps_in1.dim
    pairs = {"node_feats": (torch.randn(3, size, dtype=torch.float64), torch.randn(3, size, dtype=torch.float64))}
    torch.testing.assert_close(clone(**pairs), reference(**pairs))
    clone.load_state_dict(original)
    assert clone._fused_homonuclear
    torch.testing.assert_close(clone(**pairs), block(**pairs))


def test_missing_modified_map_trainable_q_and_bool_validation():
    block = make_block(fuse_homonuclear=True)
    block._coefficient_transpose_index = None
    with pytest.warns(UserWarning, match="fixed Q"):
        block._configure_homonuclear_fusion()
    assert not block._fused_homonuclear
    block = make_block(fuse_homonuclear=True)
    block._coefficient_transpose_sign.zero_()
    with pytest.warns(UserWarning, match="invalid transpose"):
        block._configure_homonuclear_fusion()
    assert not block._fused_homonuclear
    block = make_block(fuse_homonuclear=True)
    block.change_of_basis.requires_grad_()
    with pytest.warns(UserWarning, match="fixed Q"):
        block._configure_homonuclear_fusion()
    assert not block._fused_homonuclear
    with pytest.raises(TypeError, match="bool"):
        make_block(fuse_homonuclear=1)


def test_runtime_autocast_falls_back_once():
    import warnings
    block = make_block(dtype=torch.float32, fuse_homonuclear=True)
    size = block.operation.tensor_products[0].irreps_in1.dim
    pair = (torch.randn(3, size), torch.randn(3, size))
    rows = []
    hook = block.operation.tensor_products[0].register_forward_pre_hook(lambda module, args: rows.append(args[0].shape[0]))
    with warnings.catch_warnings(record=True) as caught, torch.autocast("cpu", dtype=torch.bfloat16):
        warnings.simplefilter("always")  # "once per block" is the block's own flag, not the registry's
        block(node_feats=pair)
        block(node_feats=pair)
    hook.remove()
    assert rows == [6, 6]
    assert sum("Homonuclear fusion disabled" in str(w.message) for w in caught) == 1
    assert all("_homonuclear" not in key for key in block.state_dict())
    block.double()
    assert block._fused_homonuclear


@pytest.mark.parametrize("operation_name", ["E3nnSimpleEdgeBlock", "E3nnLowRankEdgeBlock", "E3nnSlaterKosterEdgeBlock"])
def test_other_readouts_are_never_selected(operation_name):
    from graph2mat.bindings import e3nn as binding
    options = {} if operation_name == "E3nnSimpleEdgeBlock" else {"edge_feats_irreps": "2x0e"}
    with pytest.warns(UserWarning, match="exact E3nnLinearEdgeBlock"):
        block = make_block(operation_cls=getattr(binding, operation_name), fuse_homonuclear=True, **options)
    assert not block._fused_homonuclear


def test_positional_and_global_arguments_keep_legacy_errors():
    block = make_block(fuse_homonuclear=True)
    reference = make_block()
    pair = (torch.ones(1, block.operation.tensor_products[0].irreps_in1.dim, dtype=torch.float64),) * 2
    with pytest.warns(UserWarning, match="keyword tensor pairs"), pytest.raises(TypeError):
        block(pair)
    with pytest.raises(TypeError):
        reference(pair)
    with pytest.raises(AssertionError):
        block(node_feats=pair, global_value=1)
    with pytest.raises(AssertionError):
        reference(node_feats=pair, global_value=1)


def test_double_q_small_incompatible_perturbation_is_not_certified():
    block = make_block(fuse_homonuclear=True)
    state = copy.deepcopy(block.state_dict())
    state["change_of_basis"][0, 0, 1] += 1e-8
    with pytest.warns(UserWarning, match="loaded Q"):
        block.load_state_dict(state)
    assert not block._fused_homonuclear


def test_scaled_q_preserves_combined_api_by_fallback():
    reference = make_block()
    state = copy.deepcopy(reference.state_dict())
    state["change_of_basis"] *= 2
    reference.load_state_dict(state)
    candidate = make_block(fuse_homonuclear=True)
    candidate.load_state_dict(state)
    size = candidate.operation.tensor_products[0].irreps_in1.dim
    pair = (torch.randn(3, size, dtype=torch.float64), torch.randn(3, size, dtype=torch.float64))
    torch.testing.assert_close(candidate.forward_with_coefficients(node_feats=pair),
                               reference.forward_with_coefficients(node_feats=pair), rtol=1e-10, atol=1e-12)
    assert not candidate._fused_homonuclear


@pytest.mark.parametrize("api", ["forward", "forward_coefficients", "forward_with_coefficients"])
def test_mismatched_endpoint_rows_preserve_legacy_values(api):
    block = make_block(basis="1x0e", irreps={"node_feats_irreps": "1x0e"}, fuse_homonuclear=True)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.fill_(1.)
    reference = copy.deepcopy(block)
    reference.fuse_homonuclear = False
    reference._configure_homonuclear_fusion()
    pair = (torch.tensor([[2.]], dtype=torch.float64), torch.tensor([[3.], [4.], [5.]], dtype=torch.float64))
    expected = getattr(reference, api)(node_feats=pair)
    actual = getattr(block, api)(node_feats=pair)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("api", ["forward", "forward_coefficients", "forward_with_coefficients"])
def test_mismatched_source_rows_preserve_legacy_error(api):
    block = make_block(basis="1x0e", irreps={"node_feats_irreps": "1x0e", "edge_messages_irreps": "1x0e"},
                       fuse_homonuclear=True)
    reference = copy.deepcopy(block)
    reference.fuse_homonuclear = False
    reference._configure_homonuclear_fusion()
    args = {"node_feats": (torch.ones(1, 1, dtype=torch.float64),) * 2,
            "edge_messages": (torch.ones(3, 1, dtype=torch.float64),) * 2}
    with pytest.raises(RuntimeError):
        getattr(reference, api)(**args)
    with pytest.raises(RuntimeError):
        getattr(block, api)(**args)


def test_scalar_projection_has_literal_result():
    operation = E3nnLinearEdgeBlock("2x0e", "1x0e").double()
    with torch.no_grad():
        operation.tensor_products[0].weight.copy_(torch.tensor([1., 2., 3., 4.]))
        operation.linears[0].weight.zero_()
    assert operation._prepare_homonuclear_fusion(torch.tensor([0]), torch.tensor([1.])) is None
    a = torch.tensor([[2., 1.]], dtype=torch.float64)
    b = torch.tensor([[3., 4.]], dtype=torch.float64)
    torch.testing.assert_close(operation(_homonuclear=True, node_feats=(a, b)),
                               torch.tensor([[24.75]], dtype=torch.float64))


def test_negative_cg_exchange_phase():
    operation = E3nnLinearEdgeBlock("1o", "1e").double()
    a, b = torch.eye(3, dtype=torch.float64)[:2]
    pair = (a[None], b[None])
    expected = operation(node_feats=pair)
    assert expected.abs().max() > 0
    assert operation._prepare_homonuclear_fusion(torch.arange(3), -torch.ones(3)) is None
    torch.testing.assert_close(operation(_homonuclear=True, node_feats=pair), expected)
