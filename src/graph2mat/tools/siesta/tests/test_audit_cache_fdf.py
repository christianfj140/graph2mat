import numpy as np
import pytest
import sisl

from graph2mat.tools.conftest import make_hamiltonian, write_fdf
from graph2mat.tools.siesta.delta import compute_h0, h0_fdf
from graph2mat.tools.siesta.dm_pilot import run_from_dm
from graph2mat.tools.siesta.label_noise import noise_fdf
from graph2mat.tools.siesta.runner import system_label


def test_system_label_resolves_fdf_include(tmp_path):
    (tmp_path / 'label.fdf').write_text('System.Label included\n')
    fdf = tmp_path / 'RUN.fdf'
    fdf.write_text('%include label.fdf\n')
    assert system_label(fdf) == 'included'


@pytest.mark.parametrize('text', ['%include settings.fdf\n', 'Spin < settings.fdf\n'])
def test_h0_rewrite_refuses_unresolved_external_settings(text):
    with pytest.raises(ValueError, match='external|another file'):
        h0_fdf(text)


def test_h0_replaces_initspin_using_full_fdf_label_rules(tmp_path):
    text = ('Spin spin-orbit\n%block DM._InitSpin\n 1 +\n%endblock\n')
    out = h0_fdf(text, n_atoms=1)
    path = tmp_path / 'RUN.fdf'
    path.write_text(out)
    assert sisl.get_sile(str(path)).get('DM.InitSpin') == ['1 0.0']


def test_noise_rewrite_refuses_external_settings():
    with pytest.raises(ValueError, match='external|another file'):
        noise_fdf('%include atoms.fdf\n', True, '1.d-6')


def test_noise_stops_at_unnamed_endblock():
    text = ('%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n%endblock\n'
            '%block LatticeVectors\n10 0 0\n0 10 0\n0 0 10\n%endblock\n'
            '%block DM.InitSpin\n1 0 0 1\n%endblock\n')
    out = noise_fdf(text, True, '1.d-6', np.array([.1, .2, .3]))
    assert '0.100000000000 0.200000000000 0.300000000000 1' in out
    assert '%block LatticeVectors\n10 0 0\n0 10 0\n0 0 10\n%endblock' in out
    assert '%block DM.InitSpin\n1 0 0 1\n%endblock' in out


def _reference(tmp_path, monkeypatch):
    h = make_hamiltonian()
    ref = tmp_path / 'reference'
    write_fdf(ref, h.geometry)
    source = ref / 'sample.TSHS'
    h.write(str(source))
    monkeypatch.setenv('FAKE_SIESTA_TSHS', str(source))
    dm = sisl.DensityMatrix.fromsp(h.geometry, h.tocsr(0) * .1, S=h.tocsr(h.S_idx))
    return ref, dm


def test_dm_cache_reuses_identical_inputs(tmp_path, fake_siesta, monkeypatch):
    ref, dm = _reference(tmp_path, monkeypatch)
    work = tmp_path / 'dm-run'
    first = run_from_dm(ref, work, dm, siesta=fake_siesta)
    monkeypatch.setenv('FAKE_SIESTA_FAIL', '1')
    assert run_from_dm(ref, work, dm, siesta=fake_siesta) == first


@pytest.mark.parametrize('change', ['dm', 'pseudo', 'fdf', 'output'])
def test_dm_cache_rejects_changed_inputs_or_output(tmp_path, fake_siesta, monkeypatch, change):
    ref, dm = _reference(tmp_path, monkeypatch)
    work = tmp_path / 'dm-run'
    out = run_from_dm(ref, work, dm, siesta=fake_siesta)
    if change == 'dm':
        dm = dm.copy()
        dm[0, 0] = (.7, 1.)
    elif change == 'pseudo':
        (ref / 'C.psf').write_text('different pseudo\n')
    elif change == 'fdf':
        fdf = ref / 'RUN.fdf'
        fdf.write_text(fdf.read_text() + 'MeshCutoff 600 Ry\n')
    else:
        make_hamiltonian(onsite=4.).write(str(out))
    preserved = out.read_bytes()
    with pytest.raises(ValueError, match='cache'):
        run_from_dm(ref, work, dm, siesta=fake_siesta)
    assert out.read_bytes() == preserved


def test_h0_cache_refuses_unverified_legacy_file(tmp_path, monkeypatch):
    ref, _ = _reference(tmp_path, monkeypatch)
    work = tmp_path / 'h0-run'
    work.mkdir()
    (work / 'sample.TSHS').write_bytes((ref / 'sample.TSHS').read_bytes())
    preserved = (work / 'sample.TSHS').read_bytes()
    with pytest.raises(ValueError, match='cache'):
        compute_h0(ref, work)
    assert (work / 'sample.TSHS').read_bytes() == preserved


def test_dm_result_with_wrong_geometry_never_lands(tmp_path, fake_siesta, monkeypatch):
    ref, dm = _reference(tmp_path, monkeypatch)
    moved = make_hamiltonian()
    moved.geometry.xyz[0, 0] += .5
    source = tmp_path / 'moved.TSHS'
    moved.write(str(source))
    monkeypatch.setenv('FAKE_SIESTA_TSHS', str(source))
    with pytest.raises(RuntimeError, match='geometry'):
        run_from_dm(ref, tmp_path / 'dm-run', dm, siesta=fake_siesta)
    assert not (tmp_path / 'dm-run').exists()


def test_corrupt_h0_result_never_lands(tmp_path, fake_siesta, monkeypatch):
    ref, _ = _reference(tmp_path, monkeypatch)
    source = tmp_path / 'broken.TSHS'
    source.write_bytes(b'not a matrix')
    monkeypatch.setenv('FAKE_SIESTA_TSHS', str(source))
    with pytest.raises(sisl.SileError):
        compute_h0(ref, tmp_path / 'h0-run', siesta=fake_siesta)
    assert not (tmp_path / 'h0-run').exists()


def test_cache_hash_works_without_python311_file_digest(tmp_path, monkeypatch):
    import hashlib
    from graph2mat.tools.siesta.runner import _sha256

    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    path = tmp_path / 'input'
    path.write_bytes(b'abc')
    assert _sha256(path) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
