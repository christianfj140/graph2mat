#!/usr/bin/env python3
"""Magnetic phase of roadmap 8.1, F0: SIESTA feasibility on bcc Fe (roadmap_magnetismo.md, Task 1).

Fully relativistic PseudoDojo psml for every run (collinear and spin-orbit share the basis).
Nominal moments go to DM.InitSpin (|m| = 2.2 mu_B, sign or angles per atom). Cases:

- relax: two-atom cubic cell, collinear ferromagnet, variable cell -> lattice constant.
- conv_*: 16-atom rattled supercell, collinear ferromagnet, k mesh and smearing convergence.
- sym_*: two-atom rattled cell with spin-orbit, ferromagnet along a generic axis n, and its
  images under time reversal (m -> -m), C4 about z (structure and m rotated) and inversion
  (structure inverted, m kept: m is axial). SCF and one-iteration H0 versions.
- colA_*: 16-atom collinear spin-flip patterns (+H0): metastability and size of H - H0.
- socB_*: 16-atom spin-orbit patterns along random axes (+H0).
- frozen_n1: converged DM of socB_fm_z rigidly rotated to n1, one iteration: H[rotated DM]
  against the SCF along n1 (socB_fm_n1); decides whether frozen non-collinear data (B') are usable.

Usage: magnetismo_f0_fe.py relax | run | analyze   (outputs in experimentos_magnetismo_f0_fe_20261021/)
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MD = Path(os.environ.get("MDVS_ROOT", Path(__file__).resolve().parents[2] / "MD_vs_AtomicDisplacement"))
sys.path.insert(0, str(MD / "Comparison/scripts"))
from g2m_autorun import cpu_temp  # noqa: E402
from g2m_bench_materials import MESH, RELAX, SINGLE_POINT, fdf_text  # noqa: E402
from g2m_delta import h0_fdf  # noqa: E402

PSEUDO = MD / "materials/bench_common/pseudos_fr/Fe.psml"
SIESTA = os.environ.get("SIESTA", "siesta")
OUT = Path(__file__).resolve().parent / "experimentos_magnetismo_f0_fe_20261021"
CPUS, NPROC, WORKERS, MAX_TEMP = "8-23", 4, 3, 79.0  # E-cores, as the materials bench
ETEMP, M0 = 0.05, 2.2  # eV (metal); nominal moment, mu_B
N1 = np.array([np.sin(np.radians(60)) * np.cos(np.radians(30)), np.sin(np.radians(60)) * np.sin(np.radians(30)),
               np.cos(np.radians(60))])  # generic axis: theta 60, phi 30 degrees


def bcc(a: float, reps: int = 1):
    import ase.build

    return ase.build.bulk("Fe", "bcc", a=a, cubic=True).repeat((reps, reps, reps))


def rattled(atoms, sigma: float, seed: int):
    atoms = atoms.copy()
    atoms.positions += np.random.default_rng(seed).normal(0, sigma, atoms.positions.shape)
    return atoms


def angles(m):
    """|m|, theta, phi (degrees) of a Cartesian moment, as DM.InitSpin expects them."""
    r = float(np.linalg.norm(m))
    return r, float(np.degrees(np.arccos(np.clip(m[2] / r, -1, 1)))), float(np.degrees(np.arctan2(m[1], m[0])))


def fdf(atoms, spin: str, k: int, moments, *, run: str = SINGLE_POINT, h0: bool = False, frozen: bool = False,
        etemp: float = ETEMP) -> str:
    """spin: 'polarized' (moments: signed floats) or 'spin-orbit' (moments: (n, 3) Cartesian)."""
    text = fdf_text(atoms, "fe", [k, k, k], mesh=MESH, etemp=etemp, run=run, comment="magnetismo_f0_fe.py")
    # DM tolerance: the bench default 1e-5. With 1e-6 (as Bi2Se3) the SOC ferromagnet hovered at
    # dDmax ~1e-5 for 70+ iterations of 74 s; runs before 2026-10-21 04:30 used 1e-6.
    # Runs started after 05:50 stop at 120 SCF steps: some spin-flip patterns never settle.
    for old, new in (("SpinPolarized                 F", f"Spin                          {spin}"),
                     ("MaxSCFIterations              300", "MaxSCFIterations              120")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    if spin == "polarized":
        lines = [f" {i + 1} {m:+.6f}" for i, m in enumerate(moments)]
    else:
        lines = [" {} {:.6f} {:.6f} {:.6f}".format(i + 1, *angles(m)) for i, m in enumerate(moments)]
    # Small cells, many k points: k-parallel diagonalization is ~2.5x faster here (6 SCF steps of conv_k6, 2026-10-21).
    text += "Diag.ParallelOverK            T\n"
    text += "Write.MullikenPop             1\n%block DM.InitSpin\n" + "\n".join(lines) + "\n%endblock DM.InitSpin\n"
    if h0 or frozen:
        text = h0_fdf(text)
    if frozen:  # one iteration from the DM file in the work directory instead of atomic densities
        assert text.count("DM.UseSaveDM  F") == 1
        text = text.replace("DM.UseSaveDM  F", "DM.UseSaveDM  T")
    return text


def run(name: str, text: str, atoms, moments, setup=None) -> None:
    work = OUT / name
    if (work / "timing.json").exists():
        return
    while cpu_temp() >= MAX_TEMP:  # ponytail: wait between runs only; a single run is short
        time.sleep(30)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    (work / "RUN.fdf").write_text(text)
    shutil.copy(PSEUDO, work)
    atoms.write(work / "structure.extxyz")
    (work / "moments.json").write_text(json.dumps(np.asarray(moments).tolist()))
    if setup is not None:
        setup(work)
    cmd = ["taskset", "-c", CPUS, "mpirun", "-np", str(NPROC), "--bind-to", "none", SIESTA]
    start = time.time()
    with (work / "RUN.fdf").open() as stdin, (work / "RUN.out").open("w") as stdout:
        proc = subprocess.run(cmd, cwd=work, stdin=stdin, stdout=stdout, stderr=subprocess.STDOUT,
                              env={**os.environ, "OMP_NUM_THREADS": "1"}, preexec_fn=lambda: os.nice(10))
    out = (work / "RUN.out").read_text(errors="ignore")
    info = {"wall_s": round(time.time() - start, 1), "returncode": proc.returncode,
            "scf_iterations": out.count("scf:") - 1, "converged": "SCF cycle converged" in out or "SCF Convergence by" in out,
            "max_cpu_temp": cpu_temp()}
    (work / "timing.json").write_text(json.dumps(info))
    print(name, info, flush=True)


def lattice_constant() -> float:
    import sisl

    geom = sisl.get_sile(str(OUT / "relax" / "fe.STRUCT_OUT")).read_geometry()
    return float(np.mean(np.diag(geom.cell)))


def cases() -> dict:
    """name: (fdf text, atoms, nominal moments, setup) for every run after the relaxation."""
    a = lattice_constant()
    out = {}
    # k mesh / smearing convergence (16 atoms, k=6 ~ k=12 on the cubic cell)
    sc = rattled(bcc(a, 2), 0.05, 1)
    fm = [M0] * len(sc)
    for k, et in ((4, ETEMP), (6, ETEMP), (8, ETEMP), (6, 0.1)):
        out[f"conv_k{k}_t{int(1000 * et)}"] = (fdf(sc, "polarized", k, fm, etemp=et), sc, fm, None)
    # symmetry checks on two atoms
    two = rattled(bcc(a, 1), 0.06, 2)
    centre = two.cell.sum(0) / 2
    c4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    images = {"base": (two, N1), "trev": (two, -N1),
              "c4": (_moved(two, centre, c4), c4 @ N1), "inv": (_moved(two, centre, -np.eye(3)), N1)}
    for tag, (atoms, n) in images.items():
        m = np.tile(M0 * n, (len(atoms), 1))
        for h0 in (False, True):
            out[f"sym_{tag}{'_h0' if h0 else ''}"] = (fdf(atoms, "spin-orbit", 12, m, h0=h0), atoms, m, None)
    # collinear spin flips (16 atoms)
    rng = np.random.default_rng(3)
    colA = {"fm": np.ones(16)}
    for n_flip in (1, 2, 4, 6, 8):
        s = np.ones(16)
        s[rng.choice(16, n_flip, replace=False)] = -1
        colA[f"flip{n_flip}"] = s
    layered = np.where(np.isclose(sc.get_scaled_positions()[:, 2] * 4 % 2, 0, atol=0.2), 1.0, -1.0)  # AFM (001) layers
    colA["afm001"] = layered
    for index, (tag, s) in enumerate(colA.items()):
        atoms = rattled(bcc(a, 2), 0.05, 22 + 2 * index)  # the seeds of the first runs (10 + len(out) then)
        for h0 in (False, True):
            out[f"colA_{tag}{'_h0' if h0 else ''}"] = (fdf(atoms, "polarized", 6, M0 * s, h0=h0), atoms, M0 * s, None)
        if tag in ("flip1", "flip4"):  # did not converge in 300 steps at 0.05 eV: does more smearing settle them?
            out[f"colA_{tag}_t100"] = (fdf(atoms, "polarized", 6, M0 * s, etemp=0.1), atoms, M0 * s, None)
    # spin-orbit patterns along axes
    axes = {"fm_z": (np.array([0, 0, 1.0]), 0), "fm_n1": (N1, 0), "flip3": (_unit(rng.normal(size=3)), 3),
            "flip6": (_unit(rng.normal(size=3)), 6)}
    soc_atoms = rattled(bcc(a, 2), 0.05, 20)
    for tag, (n, n_flip) in axes.items():
        s = np.ones(16)
        if n_flip:
            s[rng.choice(16, n_flip, replace=False)] = -1
        m = M0 * s[:, None] * n
        for h0 in (False, True):
            out[f"socB_{tag}{'_h0' if h0 else ''}"] = (fdf(soc_atoms, "spin-orbit", 6, m, h0=h0), soc_atoms, m, None)
        if tag == "flip3":  # the F4 spin-orbit recipe candidate: k 4x4x4, 0.1 eV smearing
            out["socB_flip3_k4_t100"] = (fdf(soc_atoms, "spin-orbit", 4, m, etemp=0.1), soc_atoms, m, None)
    m = np.tile(M0 * N1, (16, 1))
    out["frozen_n1"] = (fdf(soc_atoms, "spin-orbit", 6, m, frozen=True), soc_atoms, m, _rotated_dm_from("socB_fm_z", N1))
    return out


def _unit(v):
    return v / np.linalg.norm(v)


def _moved(atoms, centre, R):
    atoms = atoms.copy()
    atoms.positions = (atoms.positions - centre) @ R.T + centre
    return atoms


def _rotated_dm_from(source: str, n):
    """Setup hook: write the converged DM of ``source`` with its spin rigidly rotated from z to n."""
    def setup(work: Path) -> None:
        import sisl

        src = OUT / source
        geom = sisl.get_sile(str(src / "fe.TSHS")).read_geometry()
        dm = sisl.get_sile(str(src / "fe.DM")).read_density_matrix(geometry=geom)
        _, theta, _ = angles(n)
        axis = _unit(np.cross([0.0, 0.0, 1.0], n))  # rotating z by theta about z x n gives n
        # Sense of sisl's rotation checked in analyze() through the Mulliken moments of the result.
        dm.spin_rotate((theta, axis)).write(str(work / "fe.DM"))
    return setup


# --- analysis -------------------------------------------------------------------------------
GROUPS = {"a0": [0], "b": [1, 2, 3], "b0": [4], "a": [5, 6, 7]}  # channel groups of the full spinor
ODD = np.array([1, 1, 1, 1, -1, -1, -1, -1.0])  # time-reversal parity of [a0, bx, by, bz, b0, ax, ay, az]


def read(name):
    import sisl

    geometry = sisl.get_sile(str(OUT / name / "RUN.fdf")).read_geometry()  # TSHS carries no species
    return sisl.get_sile(str(OUT / name / "fe.TSHS")).read_hamiltonian(geometry=geometry)


def full_channels(H):
    """[a0, bx, by, bz, b0, ax, ay, az] of a spin-orbit matrix, (no, no_s, 8) dense. Written from the
    Pauli decomposition H = sum_mu sigma_mu (a_mu + i b_mu), independently of graph2mat's code."""
    c = [H.tocsr(i).toarray() for i in range(8)]
    return np.stack([(c[0] + c[1]) / 2, (c[3] + c[7]) / 2, (c[2] - c[6]) / 2, (c[4] - c[5]) / 2,
                     (c[4] + c[5]) / 2, (c[2] + c[6]) / 2, (c[7] - c[3]) / 2, (c[0] - c[1]) / 2], -1)


def blocks(H, channels):
    """{(i, j, sc_off): (n_i, n_j, C)} for every stored atom pair."""
    out, geom = {}, H.geometry
    no = geom.no
    for isc, off in enumerate(geom.lattice.sc_off):
        sub = channels[:, isc * no:(isc + 1) * no]
        for i in range(geom.na):
            oi = slice(geom.firsto[i], geom.firsto[i + 1])
            for j in range(geom.na):
                b = sub[oi, geom.firsto[j]:geom.firsto[j + 1]]
                if np.any(b):
                    out[(i, j, tuple(int(x) for x in off))] = b
    return out


def mulliken(name):
    """Per-atom moments from the run's DM and S with sisl: (na,) collinear, (na, 3) spin-orbit."""
    import sisl

    H = read(name)
    dm = sisl.get_sile(str(OUT / name / "fe.DM")).read_density_matrix(geometry=H.geometry)
    n = 2 if dm.spin.is_polarized else 8
    dm = sisl.DensityMatrix.fromsp(H.geometry, [dm.tocsr(i) for i in range(n)], S=H.tocsr(H.S_idx), spin=dm.spin)
    q = np.asarray(dm.mulliken("atom"))
    return q[1] if dm.spin.is_polarized else q[1:4].T


def _operation_check(base, image, R, controls):
    """max |error| per channel group of H(image) against H(base) moved by the (possibly improper)
    rotation R about the cell centre; m is axial, so vector channels rotate with det(R) R."""
    import torch

    import graph2mat.bindings.e3nn  # noqa: F401
    from graph2mat import AtomicTableWithEdges

    Hb, Hi = read(base), read(image)
    cb, ci = blocks(Hb, full_channels(Hb)), blocks(Hi, full_channels(Hi))
    cell = Hb.geometry.cell
    M = np.linalg.solve(cell.T, R @ cell.T)
    assert np.allclose(M, np.round(M))
    M = np.round(M).astype(int)
    table = AtomicTableWithEdges(Hb.geometry.atoms.atom)
    cob = table.change_of_basis
    D = [table.basis[int(sp)].e3nn_irreps.D_from_matrix(torch.as_tensor(cob @ R @ cob.T)).numpy()
         for sp in Hb.geometry.atoms.species]
    scale = max(np.abs(b).max() for b in cb.values())
    variants = {"axial (assumed)": np.linalg.det(R) * R, **controls}
    worst = {v: {g: 0.0 for g in GROUPS} for v in variants}
    missing = 0
    for (i, j, off), b in cb.items():
        t = ci.get((i, j, tuple(int(x) for x in M @ np.array(off))))
        if t is None:
            missing += 1
            continue
        moved = np.einsum("xy,yzc,wz->xwc", D[i], b, D[j])
        for name, Rs in variants.items():
            m = moved.copy()
            m[..., 1:4] = m[..., 1:4] @ Rs.T
            m[..., 5:8] = m[..., 5:8] @ Rs.T
            for g, idx in GROUPS.items():
                worst[name][g] = max(worst[name][g], float(np.abs(m[..., idx] - t[..., idx]).max()))
    return {"blocks": len(cb), "missing_images": missing, "max_channel_eV": float(scale), "max_error_eV": worst}


def _time_reversal_check(base, image):
    Hb, Hi = read(base), read(image)
    a, b = full_channels(Hb), full_channels(Hi)
    if a.shape != b.shape:
        return {"error": "supports differ"}
    err = np.abs(a * ODD - b)
    return {g: float(err[..., idx].max()) for g, idx in GROUPS.items()} | {"max_channel_eV": float(np.abs(a).max())}


def _group_max(ch):
    return {g: float(np.abs(ch[..., idx]).max()) for g, idx in GROUPS.items()}


def _delta(name):
    H, H0 = read(name), read(f"{name}_h0")
    if H.geometry.lattice.nsc.tolist() != H0.geometry.lattice.nsc.tolist():
        return "nsc differs between H and H0"
    if H.spin.is_polarized:
        d = [(H.tocsr(s) - H0.tocsr(s)).toarray() for s in range(2)]
        h = [H.tocsr(s).toarray() for s in range(2)]
        return {"max_abs_H_up_eV": float(np.abs(h[0]).max()), "max_abs_dH_up_eV": float(np.abs(d[0]).max()),
                "max_abs_dH_down_eV": float(np.abs(d[1]).max()),
                "max_abs_H_updown_diff_eV": float(np.abs(h[0] - h[1]).max() / 2),
                "max_abs_dH_updown_diff_eV": float(np.abs(d[0] - d[1]).max() / 2)}
    ch, ch0 = full_channels(H), full_channels(H0)
    return {"H": _group_max(ch), "H_minus_H0": _group_max(ch - ch0)}


def _beyond_graph(name):
    """Share of off-site |H|^2 and largest element beyond R_i + R_j (graph2mat's graph)."""
    import sisl

    H = read(name)
    rmax = sisl.get_sile(str(OUT / name / "Fe.ion.xml")).read_basis().maxR()
    h = H.tocsr(0).tocoo()
    na = H.geometry.na
    ia, ja_sc = H.o2a(h.row), H.o2a(h.col)
    ja, isc = ja_sc % na, h.col // H.no
    dist = np.linalg.norm(H.geometry.xyz[ja] + H.geometry.sc_off[isc] @ H.geometry.cell - H.geometry.xyz[ia], axis=1)
    off = dist > 1e-6
    beyond = off & (dist >= 2 * rmax)
    return {"cutoff_A": 2 * rmax, "share_offsite_H2": float(np.sum(h.data[beyond] ** 2) / np.sum(h.data[off] ** 2)),
            "max_meV": 1e3 * float(np.abs(h.data[beyond]).max()) if beyond.any() else 0.0,
            "max_dist_A": float(dist[beyond].max()) if beyond.any() else 0.0}


def _kpoint_conv():
    import sisl

    ref = read("conv_k8_t50")
    out = {}
    for name in ("conv_k4_t50", "conv_k6_t50", "conv_k6_t100"):
        H = read(name)
        for k in ([0.0, 0.0, 0.0], [0.13, 0.27, 0.31]):
            s = ref.Sk(k=k, format="array")
            d = [H.Hk(k=k, spin=sp, format="array") - ref.Hk(k=k, spin=sp, format="array") for sp in (0, 1)]
            c = float(np.real(np.vdot(s, d[0] + d[1])) / np.real(2 * np.vdot(s, s)))
            a = np.abs(np.stack(d) - c * s)
            out.setdefault(name, {})[str(k)] = {"shift_meV": 1e3 * c, "aligned_mae_meV": 1e3 * float(a.mean()),
                                                "aligned_max_meV": 1e3 * float(a.max())}
    del sisl
    return out


def _moments(name):
    nominal = np.array(json.loads((OUT / name / "moments.json").read_text()))
    m = mulliken(name)
    if m.ndim == 1:
        return {"nominal": nominal.tolist(), "converged": np.round(m, 4).tolist(),
                "sign_kept": bool(np.all(np.sign(m) == np.sign(nominal))), "min_abs": float(np.abs(m).min())}
    cos = np.sum(m * nominal, 1) / np.linalg.norm(m, axis=1) / np.linalg.norm(nominal, axis=1)
    return {"converged": np.round(m, 4).tolist(), "abs": np.round(np.linalg.norm(m, axis=1), 4).tolist(),
            "max_angle_to_nominal_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1))).max())}


def analyze() -> dict:
    out = {"timing": {p.parent.name: json.loads(p.read_text()) for p in sorted(OUT.glob("*/timing.json"))}}
    done = {n for n in out["timing"] if out["timing"][n]["returncode"] == 0}

    def section(key, fn, *needed):
        if all(n in done for n in needed):
            try:
                out[key] = fn()
            except Exception as error:  # noqa: BLE001  (report, keep the rest)
                out[key] = f"{type(error).__name__}: {error}"

    section("lattice_constant_A", lattice_constant, "relax")
    section("kpoint_convergence", _kpoint_conv, "conv_k4_t50", "conv_k6_t50", "conv_k8_t50", "conv_k6_t100")
    c4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    for suffix in ("", "_h0"):
        section(f"time_reversal{suffix}", lambda s=suffix: _time_reversal_check(f"sym_base{s}", f"sym_trev{s}"),
                f"sym_base{suffix}", f"sym_trev{suffix}")
        section(f"c4{suffix}", lambda s=suffix: _operation_check(f"sym_base{s}", f"sym_c4{s}", c4,
                                                                 {"spin unrotated (control)": np.eye(3)}),
                f"sym_base{suffix}", f"sym_c4{suffix}")
        section(f"inversion{suffix}", lambda s=suffix: _operation_check(
            f"sym_base{s}", f"sym_inv{s}", -np.eye(3), {"polar (control)": -np.eye(3)}),
                f"sym_base{suffix}", f"sym_inv{suffix}")
    for name in sorted(done):
        if name.startswith(("colA_", "socB_", "sym_base", "frozen")) and not name.endswith("_h0"):
            section(f"moments:{name}", lambda n=name: _moments(n), name)
        if name.startswith(("colA_", "socB_")) and not name.endswith("_h0"):
            section(f"delta:{name}", lambda n=name: _delta(n), name, f"{name}_h0")
    section("channels:socB_flip3", lambda: _group_max(full_channels(read("socB_flip3"))), "socB_flip3")
    section("beyond_graph:colA_fm", lambda: _beyond_graph("colA_fm"), "colA_fm")
    section("s_min_eig:colA_fm", lambda: float(np.linalg.eigvalsh(read("colA_fm").Sk(format="array")).min()), "colA_fm")
    section("frozen_vs_scf", lambda: {"H_frozen_minus_scf": _group_max(full_channels(read("frozen_n1"))
                                                                         - full_channels(read("socB_fm_n1"))),
                                      "H_scf": _group_max(full_channels(read("socB_fm_n1")))},
            "frozen_n1", "socB_fm_n1")
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    return out


def relax() -> None:
    atoms = bcc(2.84)
    run("relax", fdf(atoms, "polarized", 16, [M0] * 2, run=RELAX.format(variable_cell="T")), atoms, [M0] * 2)


def run_all() -> None:
    todo = cases()
    first = {n: v for n, v in todo.items() if n != "frozen_n1"}
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda kv: run(kv[0], *kv[1]), first.items()))
    run("frozen_n1", *todo["frozen_n1"])


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    OUT.mkdir(exist_ok=True)
    if cmd == "relax":
        relax()
    elif cmd == "run":
        run_all()
    elif cmd == "analyze":
        print(json.dumps(analyze(), indent=1))
    else:
        raise SystemExit(f"unknown command {cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
