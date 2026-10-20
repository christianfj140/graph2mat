#!/usr/bin/env python3
"""Roadmap 8.1 F5: SOC campaign on the Bi2Se3 bench set (plan docs/superpowers/plans/2026-10-19-soc-f4-f5.md).

The materials bench (g2m_bench_benchmark.py) cannot run SOC: it predicts through MatrixWriter, which
rejects spin_orbit on purpose, and its band metrics assume a scalar H. This runner reuses the bench
training config and ``c.run_training`` and adds what SOC needs:

- ``prepare``: Delta labels H - H0 per structure (8 sisl components + the label's S), projected
  onto the time-reversal (TRS) channels. The raw SCF TRS residual of H (~1e-5 eV) is of the order of
  SOC_TRS_RTOL relative to the much smaller H - H0, so the projection is recorded, not silent:
  every residual is written to ``delta_manifest.json``.
- ``run --arm delta --seed N``: trains `spin_orbit=True` MACE (symmetric matrices, hidden irreps
  with unnatural parity so that every g irrep is reachable) on H - H0, then evaluates the test
  structures in-process: H = prediction + H0 with the reference S (``attach_overlap``).
- Metrics per test structure, against the full SIESTA H: h0 error after the c*S gauge shift and g
  error (meV, reference support); spinor bands (canonical orthogonalization, S eigenvalues > 1e-2,
  as the bench) at Gamma and 6 seeded k points: gap at Gamma, band RMSE in [VBM-2, CBM+2] eV
  (aligned); Bi Mulliken weight of VBM/CBM at Gamma (band inversion). The same metrics for the
  prediction with g set to zero ("no SOC") and for the reference with g = 0 show how much SOC
  matters.
- ``summary``: mean over seeds per test kind.

Usage: run_soc_campaign.py prepare | run --arm delta --seed 0 [--updates N] | summary
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

import numpy as np

MD = Path("/home/christian/repositorios/MD_vs_AtomicDisplacement")
sys.path.insert(0, str(MD / "Comparison/scripts"))
import dataset_design_curves_v1 as c  # noqa: E402
import g2m_bench_benchmark as bb  # noqa: E402
import g2m_bench_materials as bench  # noqa: E402
import g2m_delta as d  # noqa: E402

s4 = c.s4
MATERIAL = "bi2se3"
RUNS = c.OUT / "runs" / "bench_soc" / MATERIAL
DELTA = RUNS / "delta_samples"
BASIS_GLOB = str(bench.BASIS / MATERIAL / "*.ion.xml")
UPDATES, BATCH = 8000, 2
HIDDEN = "16x0e + 16x0o + 16x1o + 16x1e + 16x2e + 16x2o"  # MACE needs one multiplicity for all irreps
EDGE_HIDDEN = "8x0e + 8x0o + 8x1o + 8x1e + 8x2e + 8x2o + 8x3o + 8x3e"
# Speed probe 2026-10-20 (2 hexagonal cells, RTX 5090): l<=3 nodes with correlation 3 and 3 interactions
# took 858 s to build and >2 s/step; this recipe builds in ~450 s and steps in 0.21 s, 1.1 GB. Only the
# on-site 5e part of d-d g blocks stays unreachable (11/630); with Delta, g(H-H0) = 0 exactly.
MODEL = {"num_interactions": 2, "correlation": 3, "max_ell": 3, "optim_lr": 0.0018}
VALENCE_PER_FORMULA, ATOMS_PER_FORMULA = 2 * 15 + 3 * 16, 5
CANON_THR = 1e-2


# --- labels ----------------------------------------------------------------------------------

def _read(path: Path, fdf: Path):
    import sisl

    return sisl.get_sile(str(path)).read_hamiltonian(geometry=sisl.get_sile(str(fdf)).read_geometry())


def _components(H, nsc) -> list:
    if not np.array_equal(H.nsc, nsc):
        H = H.copy()
        H.set_nsc(nsc)
    return [H.tocsr(i) for i in range(8)]


def delta_sample(sample) -> tuple:
    """Copy of ``sample`` whose label is the TRS projection of H - H0 (with the label's S)."""
    import sisl
    from graph2mat.core.data.spin_orbit import soc_channels_to_siesta, soc_trs_residual

    work = DELTA / sample.sample_id
    matrix = work / Path(sample.reference_matrix).name
    if not matrix.exists():
        h0_path = d.compute_h0(sample)
        H = _read(sample.reference_matrix, sample.run_fdf)
        H0 = _read(h0_path, sample.run_fdf)
        diff = [a - b for a, b in zip(_components(H, H.nsc), _components(H0, H.nsc))]
        stacked = np.stack([m.toarray() for m in diff], -1)
        residual = soc_trs_residual(stacked)
        h0c = (stacked[..., 0] + stacked[..., 1]) / 2
        g = [(stacked[..., 3] + stacked[..., 7]) / 2, (stacked[..., 2] - stacked[..., 6]) / 2,
             (stacked[..., 4] - stacked[..., 5]) / 2]
        projected = soc_channels_to_siesta(np.stack([h0c, *g], -1))
        from scipy.sparse import csr_array

        comps = [csr_array(projected[..., i] * (np.abs(stacked).sum(-1) != 0)) for i in range(8)]
        work.mkdir(parents=True, exist_ok=True)
        shutil.copy(sample.run_fdf, work / "RUN.fdf")
        for ion in Path(sample.reference_dir).glob("*.ion.xml"):
            shutil.copy(ion, work)
        sisl.Hamiltonian.fromsp(H.geometry, comps, S=H.tocsr(H.S_idx), spin=sisl.Spin("so")).write(str(matrix))
        (work / "trs_residual.json").write_text(json.dumps({"relative_trs_residual": residual}))
    residual = json.loads((work / "trs_residual.json").read_text())["relative_trs_residual"]
    return dataclasses.replace(sample, run_fdf=work / "RUN.fdf", reference_dir=work, reference_matrix=matrix), residual


def prepare() -> Path:
    rows = {}
    for split in ("train", "validation", "test"):
        for s in bench.samples(MATERIAL, split):
            _, residual = delta_sample(s)
            rows[s.sample_id] = residual
    path = RUNS / "delta_manifest.json"
    path.write_text(json.dumps({"max_relative_trs_residual": max(rows.values()), "residuals": rows}, indent=1))
    print("prepared", len(rows), "max residual", max(rows.values()), flush=True)
    return path


# --- metrics ---------------------------------------------------------------------------------

def _with_channels(H, S, h0, g):
    import sisl
    from graph2mat.core.data.spin_orbit import soc_channels_to_siesta
    from scipy.sparse import csr_array

    stacked = soc_channels_to_siesta(np.stack([h0, *g], -1))
    return sisl.Hamiltonian.fromsp(H.geometry, [csr_array(stacked[..., i]) for i in range(8)], S=S,
                                   spin=sisl.Spin("so"))


def _channels(H, nsc):
    comps = np.stack([m.toarray() for m in _components(H, nsc)], -1)
    return ((comps[..., 0] + comps[..., 1]) / 2,
            [(comps[..., 3] + comps[..., 7]) / 2, (comps[..., 2] - comps[..., 6]) / 2, (comps[..., 4] - comps[..., 5]) / 2])


def _levels(H, S_ref, kpts, thr):
    """Eigenvalues at each k. thr > 0: canonical orthogonalization keeping, at every k, the same
    number of largest S eigenvalues (the smallest count above thr on the k set), as the bench."""
    import scipy.linalg

    if thr <= 0:
        return np.array([scipy.linalg.eigh(H.Hk(k=k, format="array"), S_ref.Sk(k=k, format="array"),
                                           eigvals_only=True) for k in kpts])
    decomposed = [np.linalg.eigh(S_ref.Sk(k=k, format="array")) for k in kpts]
    n_keep = min(int((w > thr).sum()) for w, _ in decomposed)
    out = []
    for k, (w, u) in zip(kpts, decomposed):
        keep = np.argsort(w)[-n_keep:]
        x = u[:, keep] / np.sqrt(w[keep])
        out.append(np.linalg.eigvalsh(x.conj().T @ H.Hk(k=k, format="array") @ x))
    return np.array(out)


def _bi_weights(H, S_ref, n_occ):
    import scipy.linalg

    hk, sk = H.Hk(format="array"), S_ref.Sk(format="array")
    e, v = scipy.linalg.eigh(hk, sk)
    bi = np.repeat([H.geometry.atoms[H.geometry.o2a(o)].Z == 83 for o in range(H.no)], 2)
    w = np.real(np.conj(v) * (sk @ v))
    frac = w[bi].sum(0) / w.sum(0)
    return float(frac[n_occ - 1]), float(frac[n_occ])


def band_metrics(H, ref, kpts) -> dict:
    n_occ = VALENCE_PER_FORMULA * ref.geometry.na // ATOMS_PER_FORMULA
    out = {}
    for tag, thr in (("", 0.0), ("canon_", CANON_THR)):
        e_ref, e = _levels(ref, ref, kpts, thr), _levels(H, ref, kpts, thr)
        n = int((e_ref[0] < 0).sum())  # sisl puts E_F at 0 (insulator: same count at every k)
        if thr <= 0 and n != n_occ:
            raise ValueError(f"{n} states below E_F, expected {n_occ} valence states")
        vbm, cbm = e_ref[:, n - 1].max(), e_ref[:, n].min()
        window = (e_ref >= vbm - 2.0) & (e_ref <= cbm + 2.0)
        diff = e - e_ref
        aligned = diff - diff[window].mean()
        out |= {f"{tag}gap_gamma_ref_eV": float(e_ref[0, n] - e_ref[0, n - 1]),
                f"{tag}gap_gamma_err_meV": 1e3 * float((e[0, n] - e[0, n - 1]) - (e_ref[0, n] - e_ref[0, n - 1])),
                f"{tag}bands_rmse_aligned_meV": 1e3 * float(np.sqrt(np.mean(aligned[window] ** 2)))}
    out["bi_weight_vbm_cbm"] = _bi_weights(H, ref, n_occ)
    return out


def evaluate(model_ckpt: Path, delta: bool) -> dict:
    from graph2mat.bindings.torch.load import load_from_lit_ckpt

    model, processor = load_from_lit_ckpt(model_ckpt, as_torch=True)
    model.eval()
    rng = np.random.default_rng(0)
    kpts = [np.zeros(3), *rng.uniform(-0.5, 0.5, (6, 3))]
    out = {}
    for s in bench.samples(MATERIAL, "test"):
        ref = _read(s.reference_matrix, s.run_fdf)
        geometry = processor.add_basis_to_geometry(ref.geometry)
        pred = processor.torch_predict(model, geometry)
        h0_p, g_p = _channels(pred, ref.nsc)
        if delta:
            h0_0, g_0 = _channels(_read(d.h0_path(s), s.run_fdf), ref.nsc)
            h0_p, g_p = h0_p + h0_0, [a + b for a, b in zip(g_p, g_0)]
        S = ref.tocsr(ref.S_idx)
        h0_r, g_r = _channels(ref, ref.nsc)
        support = np.abs(np.stack([h0_r, *g_r], -1)).sum(-1) != 0
        s_dense = S.toarray()
        dh0 = h0_p - h0_r
        shift = float((s_dense[support] * dh0[support]).sum() / (s_dense[support] ** 2).sum())
        dg = np.stack([a - b for a, b in zip(g_p, g_r)], -1)
        Hp = _with_channels(ref, S, h0_p, g_p)
        row = {"kind": s.dim,
               "h0_mae_gauge_meV": 1e3 * float(np.abs(dh0 - shift * s_dense)[support].mean()),
               "g_mae_meV": 1e3 * float(np.abs(dg[support]).mean()),
               "g_max_meV": 1e3 * float(np.abs(dg[support]).max()),
               "soc": band_metrics(Hp, ref, kpts),
               "pred_without_g": band_metrics(_with_channels(ref, S, h0_p, [0 * x for x in g_p]), ref, kpts),
               "reference_without_g": band_metrics(_with_channels(ref, S, h0_r, [0 * x for x in g_r]), ref, kpts),
               "reference": {"bi_weight_vbm_cbm": _bi_weights(ref, ref, VALENCE_PER_FORMULA * ref.geometry.na // 5)}}
        out[s.sample_id] = row
        print(s.sample_id, json.dumps(row), flush=True)
    return out


# --- training --------------------------------------------------------------------------------

def run(arm: str, seed: int, updates: int | None) -> Path:
    if arm != "delta":
        raise ValueError("only the Delta arm runs before roadmap 8.2 (direct arm needs long readout edges)")
    job = f"{MATERIAL}__soc_{arm}__ts{seed}" + (f"__u{updates}" if updates else "")
    run_dir = RUNS / job
    result = run_dir / "result.json"
    if result.exists():
        return result
    s4.DEFAULT_BASIS_GLOB = BASIS_GLOB
    accelerator = s4.torch_backend_preflight()["effective_backend"]
    train = [delta_sample(s)[0] for s in bench.samples(MATERIAL, "train")]
    val = [delta_sample(s)[0] for s in bench.samples(MATERIAL, "validation")]
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
        c.apply_config_patch(config, {
            "lr_scheduler": c._cosine(epochs),
            "data": {"batch_size": BATCH, "matrix_component_policy": "spin_orbit", "symmetric_matrix": True,
                     "n_matrix_components": 4},
            "model": {**MODEL, "loss": "graph2mat.metrics.elementwise_mse", "hidden_irreps": HIDDEN,
                      "edge_hidden_irreps": EDGE_HIDDEN, "spin_orbit": True, "symmetric_matrix": True},
            "trainer": {"max_epochs": epochs, "check_val_every_n_epoch": max(1, epochs // 200),
                        "precision": "32-true", "callbacks": [
                            {"class_path": "ModelCheckpoint", "init_args": {
                                "monitor": "val_loss", "mode": "min", "save_top_k": 1,
                                "filename": "best-epoch{epoch}-step{step}"}}]},
        })
        checkpoint, seconds, peak = c.run_training(config, run_dir)
        c.write_json(done, {"checkpoint": str(checkpoint), "seconds": seconds, "peak": peak})
    tests = evaluate(Path(checkpoint), delta=True)
    c.write_json(result, {"job_id": job, "arm": arm, "training_seed": seed, "updates": n_updates,
                          "checkpoint": str(checkpoint), "train_seconds": seconds, "peak_gpu_mib": peak,
                          "hidden_irreps": HIDDEN, "edge_hidden_irreps": EDGE_HIDDEN, "tests": tests})
    return result


def summary() -> dict:
    rows = [c.read_json(f) for f in sorted(RUNS.glob("*/result.json")) if "__u" not in f.parent.name]
    out = {}
    for r in rows:
        for t in r["tests"].values():
            out.setdefault(r["arm"], {}).setdefault(t["kind"], []).append(t)
    table = {}
    for arm, kinds in out.items():
        for kind, ts in kinds.items():
            mean = lambda f: float(np.mean([f(t) for t in ts]))  # noqa: E731
            table[f"{arm}/{kind}"] = {
                "n": len(ts),
                "h0_mae_gauge_meV": mean(lambda t: t["h0_mae_gauge_meV"]),
                "g_mae_meV": mean(lambda t: t["g_mae_meV"]),
                **{f"{case}_{m}": mean(lambda t, case=case, m=m: abs(t[case][m]))
                   for case in ("soc", "pred_without_g", "reference_without_g")
                   for m in ("gap_gamma_err_meV", "canon_gap_gamma_err_meV", "bands_rmse_aligned_meV",
                             "canon_bands_rmse_aligned_meV")},
            }
    path = RUNS / "summary.json"
    path.write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))
    return table


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("prepare", "run", "summary"))
    p.add_argument("--arm", default="delta")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--updates", type=int)
    a = p.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "run":
        print("done", run(a.arm, a.seed, a.updates), flush=True)
    else:
        summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
