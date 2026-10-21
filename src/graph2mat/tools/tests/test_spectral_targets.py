import numpy as np
import pytest
import sisl

from graph2mat.tools.conftest import make_hamiltonian
from graph2mat.tools.spectral_targets import (
    NULL_FILE,
    WINDOW_FILE,
    null_space_columns,
    spectral_window_columns,
    write_projection_files,
)


@pytest.fixture
def label(tmp_path):
    H = make_hamiltonian(geometry=sisl.geom.graphene(atoms=sisl.Atom(6, R=1.6)).tile(2, 0))
    H.write(str(tmp_path / "x.TSHS"))
    return tmp_path / "x.TSHS"


def test_window_columns_are_s_orthonormal_eigenvectors_in_the_window(label):
    H = sisl.get_sile(str(label)).read_hamiltonian()
    S = H.Sk(format="array")
    e_all = np.linalg.eigvals(np.linalg.solve(S, H.Hk(format="array"))).real
    C = spectral_window_columns(label, window_ev=3.0)
    assert C.shape[1] == np.sum(np.abs(e_all) <= 3.0) > 0
    np.testing.assert_allclose(C.T @ S @ C, np.eye(C.shape[1]), atol=1e-10)
    assert spectral_window_columns(label, window_ev=0.0).shape[1] == 0


def test_null_space_scaling(label):
    S = sisl.get_sile(str(label)).read_hamiltonian().Sk(format="array")
    lam = np.sort(np.linalg.eigvalsh(S))
    threshold = np.median(lam)
    kept = lam[lam < threshold]
    C = null_space_columns(label, threshold=threshold, floor=lam.min() * 2)
    assert C.shape[1] == len(kept) > 0
    # Columns: u / sqrt(max(lambda, floor)) -> C^T S C = lambda / max(lambda, floor) on the diagonal.
    expected = kept / np.maximum(kept, lam.min() * 2)
    np.testing.assert_allclose(np.sort(np.diag(C.T @ S @ C)), np.sort(expected), atol=1e-10)


def test_write_projection_files_pads_to_the_widest(tmp_path):
    dirs = [tmp_path / "a", tmp_path / "b"]
    for d in dirs:
        d.mkdir()
    width = write_projection_files([np.ones((4, 1)), np.ones((4, 3))], dirs, WINDOW_FILE)
    assert width == 3
    a = np.load(dirs[0] / WINDOW_FILE)["C"]
    assert a.shape == (4, 3) and a.dtype == np.float32 and a[:, 1:].sum() == 0
    assert sorted(p.name for p in dirs[0].iterdir()) == [WINDOW_FILE]
    with pytest.raises(ValueError):
        write_projection_files([np.ones((4, 1))], dirs, NULL_FILE)
