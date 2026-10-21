#!/usr/bin/env python3
"""Roadmap 8.1, magnetic phase, F5: campaign on the bcc Fe bench sets (auditorias_mejoras/roadmap_magnetismo.md).

Materials: ``fe`` (collinear, policy ``spin_h_only`` + ``magnetic_moments="collinear"``) and ``fe_soc``
(spin-orbit, policy ``spin_orbit_full`` + ``magnetic_moments="vector"``). Labels and nominal moments come
from g2m_bench_materials (F4); samples listed in ``moment_audit.json["excluded"]`` (SCF left the nominal
pattern or did not converge) are skipped everywhere.

Arms:
- ``delta``: H - H0 (H0: one SIESTA iteration from atomic densities with the same nominal moments).
- ``delta_nomom``: the same labels and H0, the network sees zero moments (control: what the moments add
  beyond H0). With zero input the exact time-reversal readout predicts no odd channels.
- ``direct`` (fe_soc only): H itself.

Metrics per test structure (D5 of the roadmap, fixed before any result): H MAE/max per channel group
(collinear: up, down, (up - down)/2; spin-orbit: a0, b, b0, a), with the c*S gauge shift removed from the
spin-independent part; band RMSE (aligned) at Gamma and 6 seeded k points for reference levels within
[E_F - 3, E_F + 2] eV; per-atom moments recomputed from H (same k grid and Fermi-Dirac smearing as the
labels, fixed electron count) against the same computation on the reference H, in mu_B and, with
spin-orbit, in degrees. The same band and moment metrics for the prediction with the odd channels of the
network output set to zero (ablation) and, for Delta arms, for H0 alone.

Usage: run_magnetic_campaign.py prepare <material> | run <material> --arm delta --seed 0 [--updates N]
       | summary <material> | check <material>   (check: moments from the reference H vs SIESTA's)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")  # small dense eigenproblems; SIESTA labels may share the CPU
import numpy as np  # noqa: E402

MD = Path(os.environ.get("MDVS_ROOT", Path(__file__).resolve().parents[2] / "MD_vs_AtomicDisplacement"))
sys.path.insert(0, str(MD / "Comparison/scripts"))
import dataset_design_curves_v1 as c  # noqa: E402
import g2m_bench_materials as bench  # noqa: E402
import g2m_delta as d  # noqa: E402

s4 = c.s4
UPDATES, BATCH = 8000, 2
HIDDEN = "16x0e + 16x0o + 16x1o + 16x1e + 16x2e + 16x2o"  # recipe of the Bi2Se3 SOC campaign
EDGE_HIDDEN = "8x0e + 8x0o + 8x1o + 8x1e + 8x2e + 8x2o + 8x3o + 8x3e"
MODEL = {"num_interactions": 2, "correlation": 3, "max_ell": 3, "optim_lr": 0.0018}
VALENCE_PER_ATOM = 16  # Fe psml z-pseudo
WINDOW = (-3.0, 2.0)  # eV around E_F
CANON_THR = 1e-2
ARMS = {"fe": ("delta", "delta_nomom"), "fe_soc": ("delta", "delta_nomom", "direct")}


def runs_dir(material: str) -> Path:
    return c.OUT / "runs" / "bench_mag" / material


def excluded(material: str) -> set:
    return set(json.loads((bench.WORK / material / "moment_audit.json").read_text())["excluded"])


def samples(material: str, split: str, labelled_only: bool = False) -> list:
    """The split's samples minus the audit's exclusions; labelled_only (smoke runs before the audit):
    every sample whose label exists."""
    if labelled_only:
        return [s for s in bench.samples(material, split) if Path(s.reference_matrix).exists()
                and not (Path(s.reference_dir) / "LABEL_FAILED").exists()]
    bad = excluded(material)
    return [s for s in bench.samples(material, split) if s.sample_id not in bad]


def _read(path: Path, fdf: Path):
    import sisl

    return sisl.get_sile(str(path)).read_hamiltonian(geometry=sisl.get_sile(str(fdf)).read_geometry())


def _components(H, nsc) -> list:
    if not np.array_equal(H.nsc, nsc):
        H = H.copy()
        H.set_nsc(nsc)
    n = 2 if H.spin.is_polarized else 8
    return [H.tocsr(i) for i in range(n)]


def _nominal(sample) -> np.ndarray:
    return np.asarray(json.loads((Path(sample.reference_dir) / "magnetic_moments.json").read_text())["nominal"])


# --- labels ----------------------------------------------------------------------------------

def arm_sample(material: str, sample, arm: str):
    """Copy of ``sample`` with label H - H0 (delta arms) or H (direct), its S, and the moments the network
    sees (nominal; zeros for delta_nomom)."""
    import sisl
    from scipy.sparse import csr_array

    work = runs_dir(material) / f"{arm}_samples" / sample.sample_id
    matrix = work / Path(sample.reference_matrix).name
    if not matrix.exists():
        H = _read(sample.reference_matrix, sample.run_fdf)
        comps = _components(H, H.nsc)
        if arm != "direct":
            H0 = _read(d.compute_h0(sample), sample.run_fdf)
            comps = [a - b for a, b in zip(comps, _components(H0, H.nsc))]
        work.mkdir(parents=True, exist_ok=True)
        shutil.copy(sample.run_fdf, work / "RUN.fdf")
        for ion in Path(sample.reference_dir).glob("*.ion.xml"):
            shutil.copy(ion, work)
        sisl.Hamiltonian.fromsp(H.geometry, [csr_array(m) for m in comps], S=H.tocsr(H.S_idx),
                                spin=H.spin).write(str(matrix))
        nominal = _nominal(sample)
        (work / "magnetic_moments.json").write_text(json.dumps(
            {"nominal": (0 * nominal if arm == "delta_nomom" else nominal).tolist()}))
    return dataclasses.replace(sample, run_fdf=work / "RUN.fdf", reference_dir=work, reference_matrix=matrix)


def prepare(material: str) -> None:
    n = 0
    for arm in ARMS[material]:
        for split in ("train", "validation", "test"):
            for s in samples(material, split):
                arm_sample(material, s, arm)
                n += 1
    print("prepared", n, flush=True)


# --- physics from H --------------------------------------------------------------------------

def _kgrid(sample) -> list:
    import re

    block = re.search(r"%block kgrid_Monkhorst_Pack(.*?)%endblock", Path(sample.run_fdf).read_text(), re.S).group(1)
    n = [int(line.split()[i]) for i, line in enumerate(block.strip().splitlines())]
    return [np.array([i / n[0], j / n[1], k / n[2]]) for i in range(n[0]) for j in range(n[1]) for k in range(n[2])]


def _etemp(sample) -> float:
    import re

    return float(re.search(r"(?m)^ElectronicTemperature\s+(\S+)\s+eV", Path(sample.run_fdf).read_text()).group(1))


def _states(H, S, k):
    """Eigenvalues and S-orthonormal eigenvectors at k: per spin (collinear, list of 2) or spinor."""
    import scipy.linalg

    sk = S.Sk(k=k, format="array")  # spin-orbit matrices: already the spinor S (2 no, spin-interleaved)
    if H.spin.is_polarized:
        return [scipy.linalg.eigh(H.Hk(k=k, spin=s, format="array"), sk) for s in (0, 1)], sk
    return [scipy.linalg.eigh(H.Hk(k=k, format="array"), sk)], sk


def _fermi(levels, n_el, kT):
    from scipy.optimize import brentq
    from scipy.special import expit

    flat = np.concatenate([np.ravel(e) for e in levels])
    weight = 1.0 / len(levels)
    count = lambda mu: weight * expit(-(flat - mu) / kT).sum() - n_el  # noqa: E731
    return brentq(count, flat.min() - 1, flat.max() + 1, xtol=1e-10)


def moments_from_h(H, S, kpts, kT, n_el) -> tuple:
    """(E_F, per-atom moments) from H on the label's k grid with Fermi-Dirac smearing: Mulliken spin of
    the occupied states, (na,) collinear (up - down) or (na, 3) spin-orbit."""
    from scipy.special import expit

    sols = [_states(H, S, k) for k in kpts]
    levels = [np.concatenate([e for e, _ in sol]) for sol, _ in sols]
    mu = _fermi(levels, n_el, kT)
    no, na = H.no, H.geometry.na
    atom = np.array([H.geometry.o2a(o) for o in range(no)])
    if H.spin.is_polarized:
        q = np.zeros((2, no))
        for sol, sk in sols:
            for s, (e, v) in enumerate(sol):
                f = expit(-(e - mu) / kT)
                q[s] += np.real(np.einsum("in,in,n->i", v.conj(), sk @ v, f))
        m = (q[0] - q[1]) / len(kpts)
        return mu, np.bincount(atom, m, minlength=na)
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    m = np.zeros((no, 3))
    for sol, sk in sols:
        e, v = sol[0]
        f = expit(-(e - mu) / kT)
        sv = sk @ v
        a, b = v.reshape(no, 2, -1), sv.reshape(no, 2, -1)  # orbital-major, spin-interleaved
        m += np.real(np.einsum("isn,psq,iqn,n->ip", a.conj(), pauli, b, f))
    m /= len(kpts)
    return mu, np.stack([np.bincount(atom, m[:, p], minlength=na) for p in range(3)], -1)


def _hks(H, k):
    return [H.Hk(k=k, spin=s, format="array") for s in (0, 1)] if H.spin.is_polarized else [H.Hk(k=k, format="array")]


def _levels(H, S, kpts, thr: float = 0.0):
    """Sorted levels at each k; thr > 0: canonical orthogonalization keeping, at every k, the same number
    of largest S eigenvalues (the smallest count above thr on the k set), as the SOC campaign: S of bcc Fe
    is ill-conditioned (smallest eigenvalue 7.6e-5 at Gamma, magnetic F0)."""
    if thr <= 0:
        return np.array([np.sort(np.concatenate([e for e, _ in _states(H, S, k)[0]])) for k in kpts])
    decomposed = [np.linalg.eigh(S.Sk(k=k, format="array")) for k in kpts]
    n_keep = min(int((w > thr).sum()) for w, _ in decomposed)
    out = []
    for k, (w, u) in zip(kpts, decomposed):
        keep = np.argsort(w)[-n_keep:]
        x = u[:, keep] / np.sqrt(w[keep])
        out.append(np.sort(np.concatenate([np.linalg.eigvalsh(x.conj().T @ h @ x) for h in _hks(H, k)])))
    return np.array(out)


def band_error(H, ref, S, kpts, mu_ref, thr: float = 0.0) -> float:
    e_ref, e = _levels(ref, S, kpts, thr), _levels(H, S, kpts, thr)
    window = (e_ref >= mu_ref + WINDOW[0]) & (e_ref <= mu_ref + WINDOW[1])
    diff = e - e_ref
    aligned = diff - diff[window].mean()
    return 1e3 * float(np.sqrt(np.mean(aligned[window] ** 2)))


def moment_error(m, m_ref) -> dict:
    if m.ndim == 1:
        return {"moment_mae_muB": float(np.abs(m - m_ref).mean()), "moment_max_muB": float(np.abs(m - m_ref).max())}
    cos = np.sum(m * m_ref, 1) / np.linalg.norm(m, axis=1) / np.linalg.norm(m_ref, axis=1)
    return {"moment_mae_muB": float(np.linalg.norm(m - m_ref, axis=1).mean()),
            "moment_abs_mae_muB": float(np.abs(np.linalg.norm(m, axis=1) - np.linalg.norm(m_ref, axis=1)).mean()),
            "angle_max_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1))).max())}


def channels(comps) -> dict:
    """Channel groups of a list of csr components: collinear {up, down, diff} or spin-orbit {a0, b, b0, a}."""
    x = np.stack([m.toarray() for m in comps], -1)
    if x.shape[-1] == 2:
        return {"up": x[..., 0], "down": x[..., 1], "diff": (x[..., 0] - x[..., 1]) / 2}
    from graph2mat.core.data.spin_orbit import siesta_to_soc_full_channels

    ch = siesta_to_soc_full_channels(x)
    return {"a0": ch[..., 0], "b": ch[..., 1:4], "b0": ch[..., 4], "a": ch[..., 5:8]}


def odd_zeroed(comps) -> list:
    """The time-reversal-odd part removed: up = down = their mean, or b0 = a = 0."""
    from scipy.sparse import csr_array

    if len(comps) == 2:
        mean = (comps[0] + comps[1]) / 2
        return [csr_array(mean), csr_array(mean)]
    from graph2mat.core.data.spin_orbit import siesta_to_soc_full_channels, soc_full_channels_to_siesta

    ch = siesta_to_soc_full_channels(np.stack([m.toarray() for m in comps], -1))
    ch[..., 4:] = 0
    x = soc_full_channels_to_siesta(ch)
    return [csr_array(x[..., i]) for i in range(8)]


def _matrix(ref, comps):
    import sisl
    from scipy.sparse import csr_array

    return sisl.Hamiltonian.fromsp(ref.geometry, [csr_array(m) for m in comps], S=ref.tocsr(ref.S_idx), spin=ref.spin)


# --- prediction and evaluation ---------------------------------------------------------------

def predict(model, processor, geometry, moments) -> object:
    import torch

    from graph2mat import OrbitalConfiguration
    from graph2mat.bindings.torch import TorchBasisMatrixData

    config = OrbitalConfiguration.from_geometry(geometry, magnetic_moments=moments)
    with torch.no_grad():
        data = TorchBasisMatrixData.new(config, data_processor=processor, labels=False)
        data["ptr"] = torch.tensor([0, len(data["point_types"])])
        data["batch"] = torch.zeros(len(data["point_types"]), dtype=torch.long)
        return processor.matrix_from_data(data, predictions=model(data))


def evaluate(material: str, arm: str, ckpt: Path, tests=None) -> dict:
    from graph2mat.bindings.torch.load import load_from_lit_ckpt

    model, processor = load_from_lit_ckpt(ckpt, as_torch=True)
    model.eval()
    rng = np.random.default_rng(0)
    band_k = [np.zeros(3), *rng.uniform(-0.5, 0.5, (6, 3))]
    out = {}
    for s in samples(material, "test") if tests is None else tests:
        ref = _read(s.reference_matrix, s.run_fdf)
        S = ref
        moments = 0 * _nominal(s) if arm == "delta_nomom" else _nominal(s)
        geometry = processor.add_basis_to_geometry(ref.geometry)
        net = _components(predict(model, processor, geometry, moments), ref.nsc)
        base = _components(_read(d.h0_path(s), s.run_fdf), ref.nsc) if arm != "direct" else [0 * m for m in net]
        ref_c = _components(ref, ref.nsc)
        cases = {"model": [a + b for a, b in zip(net, base)],
                 "ablation_odd_zero": [a + b for a, b in zip(odd_zeroed(net), base)]}
        if arm != "direct":
            cases["h0_only"] = base
        ch_p, ch_r = channels(cases["model"]), channels(ref_c)
        support = np.abs(np.stack([m.toarray() for m in ref_c], -1)).sum(-1) != 0
        s_dense = ref.tocsr(ref.S_idx).toarray()
        even = "a0" if "a0" in ch_r else "up"
        d_even = (ch_p[even] - ch_r[even]) if even == "a0" else ((ch_p["up"] + ch_p["down"]) - (ch_r["up"] + ch_r["down"])) / 2
        shift = float((s_dense[support] * d_even[support]).sum() / (s_dense[support] ** 2).sum())
        row = {"kind": s.dim, "n_atoms": ref.geometry.na}
        for g in ch_r:
            diff = ch_p[g] - ch_r[g]
            if g in ("a0", "up", "down"):
                diff = diff - shift * s_dense
            diff = diff[support]
            row[f"{g}_mae_meV"] = 1e3 * float(np.abs(diff).mean())
            row[f"{g}_max_meV"] = 1e3 * float(np.abs(diff).max())
        kpts, kT, n_el = _kgrid(s), _etemp(s), VALENCE_PER_ATOM * ref.geometry.na
        mu_ref, m_ref = moments_from_h(ref, S, kpts, kT, n_el)
        for name, comps in cases.items():
            H = _matrix(ref, comps)
            _, m = moments_from_h(H, S, kpts, kT, n_el)
            row[name] = {"bands_rmse_aligned_meV": band_error(H, ref, S, band_k, mu_ref),
                         "canon_bands_rmse_aligned_meV": band_error(H, ref, S, band_k, mu_ref, CANON_THR),
                         **moment_error(m, m_ref)}
        out[s.sample_id] = row
        print(s.sample_id, json.dumps(row), flush=True)
    return out


def check(material: str) -> None:
    """Moments recomputed from the reference H against SIESTA's converged Mulliken moments (validation of
    moments_from_h), on the first test structures."""
    for s in samples(material, "test")[:2]:
        ref = _read(s.reference_matrix, s.run_fdf)
        _, m = moments_from_h(ref, ref, _kgrid(s), _etemp(s), VALENCE_PER_ATOM * ref.geometry.na)
        conv = np.asarray(json.loads((Path(s.reference_dir) / "magnetic_moments.json").read_text())["converged"])
        print(s.sample_id, "max |m_H - m_SIESTA| =", float(np.abs(m - conv).max()), "mu_B", flush=True)


# --- training --------------------------------------------------------------------------------

def run(material: str, arm: str, seed: int, updates: int | None) -> Path:
    if arm not in ARMS[material]:
        raise ValueError(f"{material}: arm {arm!r} not in {ARMS[material]}")
    soc = material == "fe_soc"
    job = f"{material}__{arm}__ts{seed}" + (f"__u{updates}" if updates else "")
    run_dir = runs_dir(material) / job
    result = run_dir / "result.json"
    if result.exists():
        return result
    s4.DEFAULT_BASIS_GLOB = str(bench.BASIS / material / "*.ion.xml")
    accelerator = s4.torch_backend_preflight()["effective_backend"]
    # Smoke (--updates before the moment audit): the first labelled training structures only, to time the
    # build, the steps and the evaluation while the labels are still running.
    smoke = updates is not None and not (bench.WORK / material / "moment_audit.json").exists()
    if smoke:
        labelled = samples(material, "train", labelled_only=True)[:9]
        if len(labelled) < 2:
            raise RuntimeError("smoke run needs two labelled training structures")
        train = [arm_sample(material, s, arm) for s in labelled[:-1]]
        val = [arm_sample(material, labelled[-1], arm)]
    else:
        train = [arm_sample(material, s, arm) for s in samples(material, "train")]
        val = [arm_sample(material, s, arm) for s in samples(material, "validation")]
    n_updates = updates or UPDATES
    epochs = max(1, n_updates // -(-len(train) // BATCH))
    done = run_dir / "train_done.json"
    if done.exists():
        info = c.read_json(done)
        checkpoint, seconds, peak = Path(info["checkpoint"]), info["seconds"], info["peak"]
    else:
        shutil.rmtree(run_dir, ignore_errors=True)
        config = s4.build_graph2mat_config(run_dir, train, val, run_name=job, accelerator=accelerator,
                                           training_seed=seed, **{**c.TRAIN_KW, "precision": "32-true",
                                                                   "batch_size": BATCH, "max_epochs": epochs})
        kind = "vector" if soc else "collinear"
        c.apply_config_patch(config, {
            "lr_scheduler": c._cosine(epochs),
            "data": {"batch_size": BATCH, "matrix_component_policy": "spin_orbit_full" if soc else "spin_h_only",
                     "magnetic_moments": kind, "symmetric_matrix": True, "n_matrix_components": 8 if soc else 2},
            "model": {**MODEL, "loss": "graph2mat.metrics.elementwise_mse", "hidden_irreps": HIDDEN,
                      "edge_hidden_irreps": EDGE_HIDDEN, "spin_orbit": soc, "magnetic_moments": kind,
                      "symmetric_matrix": True},
            "trainer": {"max_epochs": epochs, "check_val_every_n_epoch": max(1, epochs // 200),
                        "precision": "32-true", "callbacks": [
                            {"class_path": "ModelCheckpoint", "init_args": {
                                "monitor": "val_loss", "mode": "min", "save_top_k": 1,
                                "filename": "best-epoch{epoch}-step{step}"}}]},
        })
        checkpoint, seconds, peak = c.run_training(config, run_dir)
        c.write_json(done, {"checkpoint": str(checkpoint), "seconds": seconds, "peak": peak})
    tests = evaluate(material, arm, Path(checkpoint), tests=[labelled[-1]] if smoke else None)
    c.write_json(result, {"job_id": job, "material": material, "arm": arm, "training_seed": seed,
                          "updates": n_updates, "checkpoint": str(checkpoint), "train_seconds": seconds,
                          "peak_gpu_mib": peak, "n_train": len(train), "n_val": len(val),
                          "hidden_irreps": HIDDEN, "edge_hidden_irreps": EDGE_HIDDEN, "tests": tests})
    return result


def summary(material: str) -> dict:
    rows = [c.read_json(f) for f in sorted(runs_dir(material).glob("*/result.json")) if "__u" not in f.parent.name]
    groups = {}
    for r in rows:
        for t in r["tests"].values():
            kind = "ood" if t["kind"].startswith("ood") else "test"
            groups.setdefault(f"{r['arm']}/{t['kind'] if kind == 'ood' else 'in_distribution'}", []).append(t)
    table = {}
    for key, ts in groups.items():
        entry = {"n": len(ts)}
        for name in ts[0]:
            if name.endswith("_meV") or name.endswith("_muB"):
                entry[name] = float(np.mean([t[name] for t in ts]))
            elif isinstance(ts[0][name], dict):
                entry[name] = {m: float(np.mean([t[name][m] for t in ts])) for m in ts[0][name]}
        table[key] = entry
    path = runs_dir(material) / "summary.json"
    path.write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))
    return table


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("prepare", "run", "summary", "check"))
    p.add_argument("material", choices=sorted(ARMS))
    p.add_argument("--arm", default="delta")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--updates", type=int)
    a = p.parse_args()
    runs_dir(a.material).mkdir(parents=True, exist_ok=True)
    if a.cmd == "prepare":
        prepare(a.material)
    elif a.cmd == "run":
        print("done", run(a.material, a.arm, a.seed, a.updates), flush=True)
    elif a.cmd == "check":
        check(a.material)
    else:
        summary(a.material)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
