"""MatrixMACE with magnetic moments as input (roadmap 8.1, magnetism).

Symmetry oracles: rotating positions and moments (proper and improper rotations, the moment
being an axial vector) rotates every orbital block with the Wigner D matrices of the orbitals
and the vector channels as Cartesian axial vectors; time reversal (m -> -m) keeps the even
channels ``[a0, bx, by, bz]`` and flips the odd ones ``[b0, ax, ay, az]``; with collinear moments
s -> -s swaps up and down. Rotations and permutations are checked on the predictions read back
through the data processor and sisl, so they do not rely on the readout's internal layout.
"""
import numpy as np
import pytest
import sisl
import torch
from e3nn import get_optimization_defaults, o3, set_optimization_defaults
from mace import modules as mace_modules
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor, OrbitalConfiguration
from graph2mat.bindings.e3nn import E3nnLinearEdgeBlock, E3nnSimpleNodeBlock, E3nnSpinOrbitFullMatrixBlock
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.core.data.spin_orbit import SOC_FULL_TIME_REVERSAL_PARITY, siesta_to_soc_full_channels
from graph2mat.models.mace import MatrixMACE

# Campaign recipe of roadmap 8.1 (auditorias_mejoras/run_soc_campaign.py): both parities up to l=2
# on the nodes, l=3 on the edges, correlation 3, two interactions.
CAMPAIGN = dict(hidden="16x0e+16x0o+16x1o+16x1e+16x2e+16x2o",
                edge_hidden="8x0e+8x0o+8x1o+8x1e+8x2e+8x2o+8x3o+8x3e", max_ell=3, correlation=3)
TINY = dict(hidden="2x0e+2x0o+2x1o+2x1e+2x2e+2x2o", edge_hidden="2x0e+2x0o+2x1o+2x1e", max_ell=2, correlation=2)
EVEN, ODD = slice(0, 4), slice(4, 8)
VECTOR = [slice(1, 4), slice(5, 8)]  # (bx, by, bz) and (ax, ay, az)


@pytest.fixture(scope="module", autouse=True)
def float64_without_torchscript():
    dtype, jit = torch.get_default_dtype(), get_optimization_defaults()["jit_script_fx"]
    torch.set_default_dtype(torch.float64)
    set_optimization_defaults(jit_script_fx=False)  # as LitMACEMatrixModel does with spin-orbit
    try:
        yield
    finally:
        torch.set_default_dtype(dtype)
        set_optimization_defaults(jit_script_fx=jit)


def _shell(l, zeta=1):
    return [sisl.AtomicOrbital(n=l + 1, l=l, m=m, zeta=zeta, R=2.1) for m in range(-l, l + 1)]


# s+d and s+p species for the symmetry checks (cheap reduced tensor products).
ATOMS = [sisl.Atom("Fe", orbitals=_shell(0) + _shell(2)), sisl.Atom("O", orbitals=_shell(0) + _shell(1))]
# Two d shells (Fe-like 3d with double zeta): on-site d1 x d2 blocks need l=5.
CAMPAIGN_ATOMS = [sisl.Atom("Fe", orbitals=_shell(2) + _shell(2, zeta=2)), ATOMS[1]]


def make_mace(table, hidden, max_ell, correlation, seed=0, **_):
    n = len(table.basis)
    torch.manual_seed(seed)
    return mace_modules.MACE(
        r_max=2 * table.maxR(), num_bessel=3, num_polynomial_cutoff=3, max_ell=max_ell,
        interaction_cls=mace_modules.RealAgnosticResidualInteractionBlock,
        interaction_cls_first=mace_modules.RealAgnosticResidualInteractionBlock,
        num_interactions=2, num_elements=n, hidden_irreps=o3.Irreps(hidden), MLP_irreps=o3.Irreps("1x0e"),
        atomic_energies=torch.zeros(n), avg_num_neighbors=3.0, atomic_numbers=torch.arange(n),
        correlation=correlation, gate=None,
    )


def matrix_mace(table, magnetic_moments, edge_hidden, **kwargs):
    mace = make_mace(table, **kwargs)
    kwargs = {k: v for k, v in kwargs.items() if k not in ("hidden", "max_ell", "correlation", "seed")}
    return MatrixMACE(mace, unique_basis=table.basis, symmetric=True, edge_hidden_irreps=o3.Irreps(edge_hidden),
                      node_operation=E3nnSimpleNodeBlock, edge_operation=E3nnLinearEdgeBlock,
                      lean_mace=True, magnetic_moments=magnetic_moments, **kwargs)  # as LitMACEMatrixModel


def build(kind, atoms, **kwargs):
    """Model (eval mode) and its data processor: ``kind`` None, "collinear" (spin_h_only) or "vector"
    (spin_orbit_full)."""
    table = AtomicTableWithEdges(atoms)
    n = {"vector": 8, "collinear": 2, None: 1}[kind]
    block = {"matrix_block_cls": E3nnSpinOrbitFullMatrixBlock} if kind == "vector" else {}
    model = matrix_mace(table, kind, n_matrix_components=n, **block, **kwargs)
    processor = MatrixDataProcessor(
        basis_table=table, out_matrix="hamiltonian", symmetric_matrix=True, sub_point_matrix=False,
        n_matrix_components=n, magnetic_moments=kind,
        matrix_component_policy={"vector": "spin_orbit_full", "collinear": "spin_h_only", None: "h_only"}[kind],
    )
    return model.eval(), processor


def data_of(processor, geometry, moments):
    config = OrbitalConfiguration.from_geometry(geometry, magnetic_moments=moments)
    return TorchBasisMatrixData.from_config(config, data_processor=processor)


def labels(model, processor, geometry, moments):
    with torch.no_grad():
        out = model(data_of(processor, geometry, moments))
    return out["node_labels"].numpy(), out["edge_labels"].numpy()


def channels(model, processor, geometry, moments):
    """Predicted ``(no, no_s, n)`` channels read back from the reconstructed sisl Hamiltonian:
    ``[a0, bx, by, bz, b0, ax, ay, az]`` (spin_orbit_full) or ``[up, down]`` (spin_h_only)."""
    data = data_of(processor, geometry, moments)
    with torch.no_grad():
        predictions = model(data)
    H = processor.matrix_from_data(data, predictions, out_format="sisl_H")
    values = np.stack([H.tocsr(i).toarray() for i in range(processor.n_matrix_components)], -1)
    return siesta_to_soc_full_channels(values) if values.shape[-1] == 8 else values


def assert_equivariant(model, processor, geometry, m, vector):
    """Rotate positions, cell and moments (axial vectors, or scalars if not ``vector``): the blocks
    rotate with the orbitals' D matrices and the vector channels as Cartesian axial vectors."""
    original = channels(model, processor, geometry, m)
    assert np.isfinite(original).all()
    for k in range(original.shape[-1]):
        assert np.abs(original[..., k]).max() > 1e-6
    cob = processor.basis_table.change_of_basis
    for Q in rotations():
        rotated = geometry.copy()
        rotated.xyz[:] = geometry.xyz @ Q.T
        rotated.cell[:] = geometry.cell @ Q.T
        axial = np.linalg.det(Q) * Q  # the moment does not change under inversion
        actual = channels(model, processor, rotated, m @ axial.T if vector else m)
        D = block_diag(*[
            processor.basis_table.basis[int(s)].e3nn_irreps.D_from_matrix(torch.as_tensor(cob @ Q @ cob.T)).numpy()
            for s in geometry.atoms.species
        ])
        expected = np.concatenate([
            np.einsum("ij,jkc,lk->ilc", D, cell, D)
            for cell in np.split(original, original.shape[1] // geometry.no, axis=1)
        ], axis=1)
        for v in VECTOR if vector else []:
            expected[..., v] = expected[..., v] @ axial.T
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.fixture(scope="module")
def vector_model():
    return build("vector", ATOMS, **TINY)


def molecule(atoms=ATOMS, seed=0):
    """Two Fe and two O atoms with every block type (Fe/O on-site, Fe-Fe, Fe-O and O-O edges)."""
    rng = np.random.default_rng(seed)
    xyz = np.array([[0.0, 0.0, 0.0], [1.9, 0.3, 0.2], [0.4, 1.5, -0.3], [1.2, 0.9, 1.4]]) + rng.normal(0, 0.05, (4, 3))
    return sisl.Geometry(xyz, atoms=[atoms[0], atoms[0], atoms[1], atoms[1]], lattice=[30.0, 30.0, 30.0])


def chain(atoms=ATOMS):
    geometry = sisl.Geometry([[0.0, 0.0, 0.0], [1.4, 0.5, 0.3]], atoms=atoms, lattice=[3.1, 30.0, 30.0])
    geometry.set_nsc([3, 1, 1])
    return geometry


def moments_of(geometry, seed=1):
    return np.random.default_rng(seed).normal(0, 2.0, (geometry.na, 3))


def rotations():
    """Six random proper rotations and four improper ones (three -R and the inversion)."""
    proper = list(Rotation.random(6, random_state=np.random.default_rng(0)).as_matrix())
    return proper + [-q for q in proper[:3]] + [-np.eye(3)]


@pytest.mark.parametrize("geometry", [molecule(), chain()], ids=["molecule", "chain"])
def test_joint_equivariance_of_positions_and_axial_moments(vector_model, geometry):
    assert_equivariant(*vector_model, geometry, moments_of(geometry), vector=True)


def test_time_reversal_is_exact(vector_model):
    model, processor = vector_model
    geometry = molecule()
    m = moments_of(geometry)
    plus, minus = labels(model, processor, geometry, m), labels(model, processor, geometry, -m)
    for p, q in zip(plus, minus):
        assert np.abs(p[:, EVEN]).max() > 1e-6 and np.abs(p[:, ODD]).max() > 1e-6
        np.testing.assert_allclose(q, p * SOC_FULL_TIME_REVERSAL_PARITY, rtol=0, atol=1e-10)


def test_zero_moments_give_exactly_zero_odd_channels(vector_model):
    model, processor = vector_model
    geometry = molecule()
    for out in labels(model, processor, geometry, np.zeros((geometry.na, 3))):
        assert np.abs(out[:, EVEN]).max() > 1e-6
        assert np.all(out[:, ODD] == 0)


def test_atom_permutation(vector_model):
    model, processor = vector_model
    geometry, perm = molecule(), [2, 0, 3, 1]
    m = moments_of(geometry)
    original = channels(model, processor, geometry, m)
    permuted = channels(model, processor, geometry.sub(perm), m[perm])
    orbitals = np.concatenate([np.arange(geometry.firsto[a], geometry.firsto[a + 1]) for a in perm])
    np.testing.assert_allclose(permuted, original[np.ix_(orbitals, orbitals)], rtol=1e-8, atol=1e-10)


def test_prediction_depends_on_the_moments(vector_model):
    # Same geometry, ferromagnetic vs one moment flipped: every channel group changes.
    model, processor = vector_model
    geometry = molecule()
    m = np.tile([0.0, 0.0, 2.2], (geometry.na, 1))
    flipped = m.copy()
    flipped[1] *= -1
    a, b = channels(model, processor, geometry, m), channels(model, processor, geometry, flipped)
    for group in (EVEN, ODD):
        assert np.abs(a[..., group] - b[..., group]).max() > 1e-4


def test_onsite_blocks_see_the_neighbour_moments(vector_model):
    # Only m_1 changes: the on-site block of atom 0 changes (neighbour term of the magnetic layer).
    model, processor = vector_model
    geometry = molecule()
    m = moments_of(geometry)
    other = m.copy()
    other[1] = [1.5, -0.7, 0.4]
    onsite = slice(geometry.firsto[0], geometry.firsto[1])
    a, b = (channels(model, processor, geometry, x)[onsite, onsite] for x in (m, other))
    for group in (EVEN, ODD):
        assert np.abs(a[..., group] - b[..., group]).max() > 1e-6


def test_campaign_irreps_reach_every_coefficient():
    """With the campaign irreps no output irrep of the eight channels is structurally zero: the
    readout coefficients of every block, combined as time reversal combines the two passes (even
    channels (F(m) + F(-m)) / 2, odd ones (F(m) - F(-m)) / 2), are non-zero in every irrep instance
    (the 0o finding of roadmap 8.1)."""
    model, processor = build("vector", CAMPAIGN_ATOMS, **CAMPAIGN)
    readout = model.matrix_readouts
    blocks = [b for b in [*readout.self_interactions, *readout.interactions.values()] if b is not None]
    captured = {id(b): [] for b in blocks}
    hooks = [b.operation.register_forward_hook(lambda mod, args, out, key=id(b): captured[key].append(out))
             for b in blocks]
    try:
        geometry = molecule(CAMPAIGN_ATOMS)
        labels(model, processor, geometry, moments_of(geometry))
    finally:
        for hook in hooks:
            hook.remove()
    missing = []
    for block in blocks:
        plus, minus = captured[id(block)]
        groups = [(block._single_irreps_out, (plus + minus) / 2), (block._g_irreps_out, (plus + minus) / 2),
                  (block._b0_irreps_out, (plus - minus) / 2), (block._a_irreps_out, (plus - minus) / 2)]
        start = 0
        for name, (irreps, coefficients) in zip(["a0", "b", "b0", "a"], groups):
            for mul, ir in irreps:
                for _ in range(mul):
                    norm = coefficients[:, start:start + ir.dim].norm()
                    if norm < 1e-8 * coefficients.abs().max():
                        missing.append(f"{block.i_irreps} x {block.j_irreps} {name} {ir}")
                    start += ir.dim
        assert start == block._irreps_out.dim
    assert not missing, missing


# ---------------------------------------------------------------------------
# Collinear moments (spin_h_only: up, down)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def collinear_model():
    return build("collinear", ATOMS, **TINY)


def test_collinear_spin_flip_swaps_up_and_down(collinear_model):
    model, processor = collinear_model
    geometry = molecule()
    s = np.array([2.2, -1.9, 0.4, 0.0])
    plus, minus = labels(model, processor, geometry, s), labels(model, processor, geometry, -s)
    for p, q in zip(plus, minus):
        assert np.abs(p[:, 0] - p[:, 1]).max() > 1e-6  # up != down with moments
        np.testing.assert_allclose(q, p[:, ::-1], rtol=0, atol=1e-12)
    for out in labels(model, processor, geometry, np.zeros(geometry.na)):
        np.testing.assert_array_equal(out[:, 0], out[:, 1])


def test_collinear_prediction_depends_on_the_moments(collinear_model):
    model, processor = collinear_model
    geometry = molecule()
    ferro, anti = np.full(geometry.na, 2.2), np.array([2.2, -2.2, 0.0, 0.0])
    for a, b in zip(labels(model, processor, geometry, ferro), labels(model, processor, geometry, anti)):
        assert np.abs(a - b).max() > 1e-4


@pytest.mark.parametrize("geometry", [molecule(), chain()], ids=["molecule", "chain"])
def test_collinear_equivariance_with_scalar_moments(collinear_model, geometry):
    # Collinear moments are scalars (0e): rotations, proper or not, leave them unchanged.
    s = np.random.default_rng(1).normal(0, 2.0, geometry.na)
    assert_equivariant(*collinear_model, geometry, s, vector=False)


# ---------------------------------------------------------------------------
# Without moments, and fail-closed options
# ---------------------------------------------------------------------------

def test_without_moments_the_model_has_no_magnetic_layer_and_ignores_moments():
    model, plain = build(None, ATOMS[1:], **TINY)
    assert model.magnetic_layer is None
    assert not [k for k in model.state_dict() if "magnetic" in k]
    geometry = sisl.Geometry([[0.0, 0.0, 0.0], [1.1, 0.4, 0.1]], atoms=ATOMS[1], lattice=[30.0, 30.0, 30.0])
    magnetic = MatrixDataProcessor(basis_table=plain.basis_table, symmetric_matrix=True, sub_point_matrix=False,
                                   magnetic_moments="vector")
    expected = labels(model, plain, geometry, None)
    for a, b in zip(labels(model, magnetic, geometry, moments_of(geometry)), expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kind,kwargs,match", [
    ("spin", {}, "magnetic_moments must be"),
    ("vector", {"n_matrix_components": 8}, "E3nnSpinOrbitFullMatrixBlock"),
    ("collinear", {"n_matrix_components": 1}, "n_matrix_components=2"),
    ("vector", {"readout_per_interaction": True}, "readout_per_interaction"),
    ("vector", {"return_coefficients": True}, "return_coefficients"),
    ("vector", {"long_range": "charges"}, "long_range"),
    ("vector", {"hamiltonian_local_context": True}, "hamiltonian_local_context"),
    ("collinear", {"n_matrix_components": 2, "matrix_targets": ["hamiltonian"]}, "matrix_targets"),
], ids=["kind", "vector-block", "collinear-n", "per-interaction", "coefficients", "long-range", "context",
        "multitarget"])
def test_rejected_options(kind, kwargs, match):
    if kind == "vector" and "n_matrix_components" not in kwargs:
        kwargs = {"n_matrix_components": 8, "matrix_block_cls": E3nnSpinOrbitFullMatrixBlock, **kwargs}
    with pytest.raises(ValueError, match=match):
        matrix_mace(AtomicTableWithEdges(ATOMS[1:]), kind, **TINY, **kwargs)


def test_forward_requires_matching_moments(collinear_model):
    model, processor = collinear_model
    geometry = molecule()
    plain = MatrixDataProcessor(basis_table=processor.basis_table, symmetric_matrix=True, sub_point_matrix=False,
                                n_matrix_components=2, matrix_component_policy="spin_h_only")
    with pytest.raises(ValueError, match="magnetic_moments"):
        labels(model, plain, geometry, None)
    vector = MatrixDataProcessor(basis_table=processor.basis_table, symmetric_matrix=True, sub_point_matrix=False,
                                 n_matrix_components=2, matrix_component_policy="spin_h_only",
                                 magnetic_moments="vector")
    with pytest.raises(ValueError, match=r"magnetic_moments.*shape"):
        labels(model, vector, geometry, moments_of(geometry))
    with pytest.raises(ValueError, match="coefficients_only"):
        model(data_of(processor, geometry, np.ones(geometry.na)), coefficients_only=True)
