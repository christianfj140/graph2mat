"""Read-only audit reproductions on synthetic matrices; no real SIESTA or HTTP."""
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import sisl

from graph2mat.tools.analysis.metrics import canonical_bands, gamma_metrics
from graph2mat.tools.analysis.breakdown import summarize_breakdown
from graph2mat.tools.siesta.delta import combine_hamiltonians, compute_h0
from graph2mat.tools.siesta.dm_pilot import aligned_error
from graph2mat.tools.siesta.runner import system_label
from graph2mat.tools.server.api_client import ServerClient


def chain(order):
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s", R=2.)])
    g = sisl.Geometry([[0., 0., 0.], [.7, 0., 0.]], atoms=[atom, atom],
                      lattice=sisl.Lattice([1.5, 10., 10.], nsc=[3, 1, 1]))
    g.lattice.sc_off = g.lattice.sc_off[order]
    h = sisl.Hamiltonian(g, orthogonal=False)
    for i in range(2):
        h[i, i] = (i + 1., 1.)
    h[0, 1] = (3., 0.)
    h[1, 0] = (3., 0.)
    h[0, 1, (1, 0, 0)] = (2., 0.)
    h[1, 0, (-1, 0, 0)] = (2., 0.)
    return h


def run():
    out = {}
    with tempfile.TemporaryDirectory(prefix="g2m-audit-tools-") as directory:
        tmp = Path(directory)
        a, b = chain([0, 1, 2]), chain([0, 2, 1])
        k = [.23, 0., 0.]
        expected = a.Hk(k=k, format="array") - b.Hk(k=k, format="array")
        result = combine_hamiltonians(a, b, -1.)
        actual = result.Hk(k=k, format="array")
        out["delta_sc_off"] = {
            "expected_max": float(np.max(np.abs(expected))),
            "actual_max": float(np.max(np.abs(actual))),
            "actual_hermiticity": float(np.max(np.abs(actual - actual.conj().T))),
        }
        assert out["delta_sc_off"]["expected_max"] < 1e-12
        assert out["delta_sc_off"]["actual_max"] > 1.

        atom = sisl.Atom(1, R=1.)
        g = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=[atom, atom],
                          lattice=[10., 10., 10.])
        g.set_nsc([1, 1, 1])
        ref = sisl.Hamiltonian.fromsp(g, sp.csr_matrix([[1., .2], [.2, 2.]]), S=sp.eye(2, format="csr"))
        pred = sisl.Hamiltonian.fromsp(g, sp.csr_matrix([[1., .3], [.3, 2.]]), S=sp.eye(2, format="csr"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actual_aligned = aligned_error(pred, ref)
        out["aligned_error_offdiagonal"] = {"actual": str(actual_aligned), "expected_dense_rms": float(np.sqrt(.02 / 4))}
        assert not np.isfinite(actual_aligned)

        # Only the spin-down matrix differs. Both matrices have well-conditioned S.
        spin_ref = sisl.Hamiltonian.fromsp(g, [sp.diags([-.5, .5]), sp.diags([-.5, .5])],
                                         S=sp.eye(2, format="csr"), spin="polarized")
        spin_pred = sisl.Hamiltonian.fromsp(g, [sp.diags([-.5, .5]), sp.diags([-1.5, 1.5])],
                                          S=sp.eye(2, format="csr"), spin="polarized")
        spin_ref.write(str(tmp / "spin-ref.TSHS"))
        spin_pred.write(str(tmp / "spin-pred.TSHS"))
        metrics = gamma_metrics(tmp / "spin-pred.TSHS", tmp / "spin-ref.TSHS", window_ev=10.)
        out["metrics_ignore_spin_down"] = {"reported": metrics, "actual_down_max_eV": 1.}
        assert metrics["H_MAE_meV"] == 0.
        assert metrics["spectral_err_meV"] == 0.

        full_bands = canonical_bands(ref, ref, [[0., 0., 0.]], threshold=0.)
        supposedly_empty = canonical_bands(ref, ref, [[0., 0., 0.]], threshold=2.)
        out["canonical_zero_kept"] = {"all_S_eigenvalues": [1., 1.], "threshold": 2., "actual_bands": supposedly_empty.tolist()}
        np.testing.assert_allclose(supposedly_empty, full_bands)

        fdf = tmp / "normalized-label.fdf"
        fdf.write_text("System.Label sample\n")
        out["system_label_normalization"] = {
            "runner": system_label(fdf),
            "sisl": sisl.get_sile(str(fdf)).get("SystemLabel", "siesta"),
        }
        assert out["system_label_normalization"] == {"runner": "siesta", "sisl": "sample"}

        # A process may write a file before returning failure. The second call must
        # not treat that failed run's artifact as a certified cached H0.
        sample = tmp / "sample"
        sample.mkdir()
        ref.write(str(sample / "sample.TSHS"))
        g.write(sample / "RUN.fdf")
        (sample / "RUN.fdf").write_text("SystemLabel sample\n" + (sample / "RUN.fdf").read_text())
        (sample / "H.psf").write_text("fake pseudo\n")
        moved = ref.copy()
        moved.geometry.xyz[0, 0] += .5
        moved.write(str(tmp / "moved.TSHS"))
        fake = tmp / "failed_siesta.py"
        fake.write_text("from pathlib import Path\nimport shutil, sys\nsys.stdin.read()\nshutil.copy(" + repr(str(tmp / "moved.TSHS")) + ", 'sample.TSHS')\nsys.exit(7)\n")
        command = f"{sys.executable} {fake}"
        first_code = None
        try:
            compute_h0(sample, tmp / "h0", siesta=command)
        except subprocess.CalledProcessError as exc:
            first_code = exc.returncode
        cached = compute_h0(sample, tmp / "h0", siesta=command)
        accepted = sisl.get_sile(str(cached)).read_geometry()
        moved_ang = float(np.abs(accepted.xyz - ref.geometry.xyz).max())
        out["h0_failed_cache"] = {"first_exit": first_code, "second_returns": cached.name, "accepted_geometry_error_Ang": moved_ang, "geometry_tol_Ang": 1e-4}
        assert first_code == 7 and moved_ang > .4

        dm = sisl.DensityMatrix.fromsp(g, sp.eye(2, format="csr"))
        dmfile = tmp / "prediction.DM"
        dm.write(str(dmfile))
        geomfile = tmp / "geometry.xyz"
        g.write(geomfile)

        class Response:
            content = dmfile.read_bytes()
            def raise_for_status(self):
                pass

        with patch("graph2mat.tools.server.api_client.requests.post", return_value=Response()):
            client = ServerClient(url="http://unused.local")
            obj = client.predict(g, sisl.DensityMatrix, model="test")
            try:
                from_path = client.predict(geomfile, sisl.DensityMatrix, model="test")
                from_path_result = {"geometry": from_path.geometry.xyz.tolist()}
            except Exception as exc:
                from_path_result = {"exception": type(exc).__name__, "message": str(exc)}
        out["client_dm_geometry_path"] = {"object_geometry": obj.geometry.xyz.tolist(), "path_result": from_path_result}

        # A perfect prediction is a normal input of the aggregation API.
        row = {"total_sum": 0., "total_n": 4, "raw_sum": 0., "gauge_c_meV": 0., "onsite_sum": 0., "onsite_n": 2}
        try:
            perfect = summarize_breakdown([row])
            out["perfect_breakdown"] = {"result": perfect}
        except Exception as exc:
            out["perfect_breakdown"] = {"exception": type(exc).__name__, "message": str(exc)}
        assert out["perfect_breakdown"].get("exception") == "ZeroDivisionError"

    print(json.dumps(out, indent=2, allow_nan=True))


if __name__ == "__main__":
    run()
