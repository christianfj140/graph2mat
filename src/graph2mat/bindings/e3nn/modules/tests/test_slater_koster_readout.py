"""Independent orbital oracles for the optional SK readout."""

import importlib
import itertools

import pytest
import torch

from graph2mat import PointBasis
from e3nn import o3
from graph2mat.bindings.e3nn.irreps_tools import ReducedTensorProducts


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
