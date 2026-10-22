"""Independent counterexamples from the tools audit; dense physical oracles."""
import numpy as np
import pytest
import scipy.sparse as sp
import sisl

from graph2mat.tools.analysis.breakdown import min_image_distances, summarize_breakdown
from graph2mat.tools.analysis.metrics import canonical_bands, gamma_metrics
from graph2mat.tools.siesta.delta import combine_hamiltonians
from graph2mat.tools.siesta.dm_pilot import aligned_error


def _two_atoms():
    return sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=sisl.Atom(1, R=2.),
                         lattice=sisl.Lattice([10., 10., 10.], nsc=[1, 1, 1]))


def _chain(order):
    geometry = _two_atoms()
    geometry.set_nsc([3, 1, 1])
    geometry.lattice.sc_off = geometry.sc_off[order]
    h = sisl.Hamiltonian(geometry, orthogonal=False)
    h[0, 0], h[1, 1] = (1., 1.), (2., 1.)
    h[0, 1, (1, 0, 0)] = (2., .1)
    h[1, 0, (-1, 0, 0)] = (2., .1)
    return h


def test_combine_matches_physical_kspace_with_permuted_images():
    a, b = _chain([0, 1, 2]), _chain([0, 2, 1])
    result = combine_hamiltonians(a, b, -1.)
    for k in ([0., 0., 0.], [.23, 0., 0.]):
        np.testing.assert_allclose(result.Hk(k=k, format='array'), 0., atol=1e-12)
        np.testing.assert_allclose(result.Sk(k=k, format='array'), a.Sk(k=k, format='array'))
    np.testing.assert_array_equal(b.sc_off, [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])


def test_combine_rejects_different_structures():
    a, b = _chain([0, 1, 2]), _chain([0, 1, 2])
    b.geometry.xyz[1, 0] += .5
    with pytest.raises(ValueError, match='geometry'):
        combine_hamiltonians(a, b, 1.)


def test_aligned_error_uses_fixed_union_support_for_offdiagonal_error():
    geometry = _two_atoms()
    ref = sisl.Hamiltonian.fromsp(geometry, sp.csr_matrix([[1., .2], [.2, 2.]]),
                                  S=sp.eye(2, format='csr'))
    pred = sisl.Hamiltonian.fromsp(geometry, sp.csr_matrix([[1., .3], [.3, 2.]]),
                                   S=sp.eye(2, format='csr'))
    assert aligned_error(pred, ref) == pytest.approx(np.sqrt(.02 / 4))


def test_canonical_bands_rejects_empty_overlap_subspace():
    h = sisl.Hamiltonian.fromsp(_two_atoms(), sp.diags([-.5, .5]), S=sp.eye(2, format='csr'))
    with pytest.raises(ValueError, match='subspace'):
        canonical_bands(h, h, [[0., 0., 0.]], threshold=2.)


def test_summarize_exact_prediction_has_zero_shares():
    row = dict(total_sum=0., total_n=4, raw_sum=0., gauge_c_meV=0., onsite_sum=0., onsite_n=2)
    summary = summarize_breakdown([row])
    assert summary['H_MAE_gauge_aligned_meV'] == 0.
    assert summary['categories']['onsite'] == dict(share_of_error=0., share_of_elements=.5, mean_error_meV=0.)


@pytest.mark.parametrize('cell,xyz,pbc,want', [
    ([10., 10., 10.], [31., 0., 0.], [True, False, False], 1.),
    ([10., 10., 10.], [31., 0., 0.], [False, False, False], 31.),
    ([[10., 0., 0.], [51., 1., 0.], [0., 0., 10.]], [30.5, .5, 0.], [True, True, False], np.sqrt(.5)),
    ([[100., 1., 0.], [99., 1., 0.], [0., 0., 10.]], [0., .04, 0.], [True, True, True], .04),
])
def test_minimum_image_handles_unwrapped_and_nonreduced_cells(cell, xyz, pbc, want):
    geometry = sisl.Geometry([[0., 0., 0.], xyz], atoms=sisl.Atom(1), lattice=cell)
    geometry.lattice.pbc = pbc
    assert min_image_distances(geometry)[0, 1] == pytest.approx(want)


def test_collinear_metrics_observe_down_channel_with_independent_oracle(tmp_path):
    geometry = _two_atoms()
    ref = sisl.Hamiltonian.fromsp(geometry, [sp.diags([-.5, .5]), sp.diags([-.5, .5])],
                                  S=sp.eye(2, format='csr'), spin='polarized')
    pred = sisl.Hamiltonian.fromsp(geometry, [sp.diags([-.5, .5]), sp.diags([-1.5, 1.5])],
                                   S=sp.eye(2, format='csr'), spin='polarized')
    ref.write(str(tmp_path / 'ref.TSHS'))
    pred.write(str(tmp_path / 'pred.TSHS'))
    metrics = gamma_metrics(tmp_path / 'pred.TSHS', tmp_path / 'ref.TSHS', window_ev=10.)
    assert metrics['H_MAE_meV'] == pytest.approx(250.)
    assert metrics['H_RMSE_meV'] == pytest.approx(500.)
    assert metrics['spectral_err_meV'] == pytest.approx(1000. / np.sqrt(2.))
