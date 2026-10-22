"""Hermitian spinor oracles, independent of Graph2Mat channel conversion helpers."""
import numpy as np
import pytest
import sisl

from graph2mat.tools.conftest import write_fdf
from graph2mat.tools.siesta.delta import add_h0, combine_hamiltonians, compute_h0, project_time_reversal
from graph2mat.tools.server._serialization import write_prediction


def _soc(magnetic=0.):
    geometry = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=sisl.Atom(6, R=2.),
                             lattice=sisl.Lattice([2.5, 10., 10.], nsc=[3, 1, 1]))
    h = sisl.Hamiltonian(geometry, spin='spin-orbit', orthogonal=False)
    h[0, 0] = (.1 + magnetic, .1 - magnetic, 0., 0., 0., 0., 0., 0., 1.)
    h[1, 1] = (-.3, -.3, 0., 0., 0., 0., 0., 0., 1.)
    forward = np.array([-.2, -.2, .04, .03, .05, -.05, -.04, .03])
    backward = np.array([-.2, -.2, -.04, -.03, -.05, .05, .04, -.03])
    h[0, 1], h[1, 0] = (*forward, .1), (*backward, .1)
    h[0, 1, (1, 0, 0)] = (*(.3 * forward), .02)
    h[1, 0, (-1, 0, 0)] = (*(.3 * backward), .02)
    return h


def _oracle(kx):
    block = np.array([[-.2 + .05j, .04 + .03j], [-.04 + .03j, -.2 - .05j]])
    off = block * (1. + .3 * np.exp(2j * np.pi * kx))
    h = np.diag([.1, .1, -.3, -.3]).astype(complex)
    h[:2, 2:], h[2:, :2] = off, off.conj().T
    return h


@pytest.mark.parametrize('complex_storage', [False, True])
def test_soc_projection_preserves_hermitian_trs_oracle(complex_storage):
    h = _soc(magnetic=1e-4)
    if complex_storage:
        h = h.astype(np.complex128)
    projected, residual = project_time_reversal(h)
    assert residual == pytest.approx(1e-4 / .3)
    for kx in (0., .23):
        actual = projected.Hk(k=[kx, 0., 0.], format='array')
        np.testing.assert_allclose(actual, _oracle(kx), atol=1e-12)
        np.testing.assert_allclose(actual, actual.conj().T, atol=1e-12)
        np.testing.assert_allclose(projected.Sk(k=[kx, 0., 0.], format='array'),
                                   h.Sk(k=[kx, 0., 0.], format='array'))


def test_combine_same_soc_in_real_and_complex_storage_is_zero():
    h = _soc()
    result = combine_hamiltonians(h, h.astype(np.complex128), -1.)
    for kx in (0., .23):
        np.testing.assert_allclose(result.Hk(k=[kx, 0., 0.], format='array'), 0., atol=1e-12)


def test_soc_projection_refuses_complex_overlap_instead_of_discarding_it():
    h = _soc().astype(np.complex128)
    h[0, 0, h.S_idx] = 1. + .01j
    with pytest.raises(ValueError, match='real overlap'):
        project_time_reversal(h)


@pytest.mark.parametrize('suffix', ['.nc', '.TSHS'])
def test_soc_writer_and_h0_reconstruction_preserve_all_channels_and_overlap(tmp_path, suffix):
    base = _soc()
    residual = sisl.Hamiltonian.fromsp(base.geometry, [.2 * base.tocsr(i) for i in range(8)], spin='spin-orbit')
    pred_path, base_path = tmp_path / ('pred' + suffix), tmp_path / ('base' + suffix)
    write_prediction(base, base_path, fermi_level=0.)
    write_prediction(residual, pred_path, fermi_level=0., orthogonal_basis=True)
    result = sisl.get_sile(str(add_h0(pred_path, base_path))).read_hamiltonian()
    assert result.spin.is_spinorbit
    for kx in (0., .23):
        hk = result.Hk(k=[kx, 0., 0.], format='array')
        np.testing.assert_allclose(hk, 1.2 * _oracle(kx), atol=1e-10)
        np.testing.assert_allclose(hk, hk.conj().T, atol=1e-12)
        np.testing.assert_allclose(result.Sk(k=[kx, 0., 0.], format='array'),
                                   base.Sk(k=[kx, 0., 0.], format='array'), atol=1e-12)


def test_soc_h0_cache_checks_changed_initialization_input(tmp_path, fake_siesta, monkeypatch):
    h = _soc()
    ref = tmp_path / 'reference'
    fdf = write_fdf(ref, h.geometry, extra='Spin spin-orbit\n')
    source = ref / 'sample.TSHS'
    h.write(str(source))
    monkeypatch.setenv('FAKE_SIESTA_TSHS', str(source))
    out = compute_h0(ref, tmp_path / 'h0-run', siesta=fake_siesta)
    monkeypatch.setenv('FAKE_SIESTA_FAIL', '1')
    assert compute_h0(ref, tmp_path / 'h0-run', siesta=fake_siesta) == out
    fdf.write_text(fdf.read_text().replace('Spin spin-orbit', 'Spin non-collinear'))
    with pytest.raises(ValueError, match='cache'):
        compute_h0(ref, tmp_path / 'h0-run', siesta=fake_siesta)
