#!/usr/bin/env python3
"""Roadmap 8.1 (spin-orbit coupling), phase F0: SIESTA feasibility on primitive Bi2Se3.

Design: docs/superpowers/specs/2026-10-19-soc-8-1-design.md (F0 authorized 2026-10-19).
Runs a handful of single points of the 5-atom rhombohedral cell (experimental lattice,
a = 4.143 A, c = 28.636 A, u_Bi = 0.4008, u_Se2 = 0.2117) with PseudoDojo nc-fr-04 psml, then
checks with sisl what graph2mat will rely on:

- the SOC .TSHS reads as an 8-component spin-orbit Hamiltonian plus S;
- raw Hermiticity of H(k) and raw time-reversal (TRS) residual of the stored elements;
- Kramers degeneracy; band gap and band inversion at Gamma against a run without SOC;
- the SOC channel g of H and of H - H0 (one non-self-consistent iteration, as g2m_delta);
- k-point and mesh convergence of the Gamma gap; wall time with and without SOC.

Usage: soc_f0_bi2se3.py [run|analyze|all]  (outputs in experimentos_soc_f0_bi2se3_20261019/)
"""
from __future__ import annotations

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
from g2m_delta import h0_fdf  # noqa: E402

PSEUDOS = MD / "materials/bench_common/pseudos_fr"
SIESTA = os.environ.get("SIESTA", "siesta")
OUT = Path(__file__).resolve().parent / "experimentos_soc_f0_bi2se3_20261019"
CPUS, NPROC, MAX_TEMP = "8-23", 4, 79.0  # E-cores, as the materials bench
A, C, U_BI, U_SE = 4.143, 28.636, 0.4008, 0.2117
VALENCE = 2 * 15 + 3 * 16  # psml z-pseudo: Bi 15 (5d 6s 6p), Se 16 (3d 4s 4p)

# name: spin flavour, Monkhorst-Pack divisions, mesh cutoff (Ry), extra fdf, H0 (non-SCF)
CASES = {
    "sr_k6": ("none", 6, 550, "", False),
    "soc_k4": ("spin-orbit", 4, 550, "", False),
    "soc_k6": ("spin-orbit", 6, 550, "", False),
    "soc_k8": ("spin-orbit", 8, 550, "", False),
    "soc_k6_m400": ("spin-orbit", 6, 400, "", False),
    "soc_k6_m700": ("spin-orbit", 6, 700, "", False),
    "soc_onsite_k6": ("spin-orbit+onsite", 6, 550, "", False),
    "soc_split_k6": ("spin-orbit", 6, 550, "SOC.Split.SR.SO T\n", False),
    "h0_soc_k6": ("spin-orbit", 6, 550, "", True),
    "h0_sr_k6": ("none", 6, 550, "", True),
}


def cell():
    """Rhombohedral primitive vectors (obverse setting) and Cartesian positions; inversion at 0."""
    hexagonal = np.array([[A, 0, 0], [-A / 2, A * np.sqrt(3) / 2, 0], [0, 0, C]])
    lattice = np.array([[2, 1, 1], [-1, 1, 1], [-1, -2, 1]]) / 3 @ hexagonal
    # Wyckoff 2c (x, x, x) in rhombohedral coordinates sits at x * (0, 0, c).
    z = {"Se1": [0.0], "Bi": [U_BI, -U_BI], "Se2": [U_SE, -U_SE]}
    symbols = ["Se", "Bi", "Bi", "Se", "Se"]
    positions = np.array([[0, 0, f * C] for f in (*z["Se1"], *z["Bi"], *z["Se2"])])
    return lattice, positions, symbols


def fdf(name: str) -> str:
    spin, k, mesh, extra, _ = CASES[name]
    lattice, positions, symbols = cell()
    species = ["Bi", "Se"]
    lines = [f"SystemName Bi2Se3 F0 {name}", "SystemLabel bi2se3", f"NumberOfAtoms {len(symbols)}",
             "NumberOfSpecies 2", "%block ChemicalSpeciesLabel", " 1 83 Bi", " 2 34 Se",
             "%endblock ChemicalSpeciesLabel", "LatticeConstant 1.0 Ang", "%block LatticeVectors",
             *(" " + " ".join(f"{x:.10f}" for x in v) for v in lattice), "%endblock LatticeVectors",
             "AtomicCoordinatesFormat Ang", "%block AtomicCoordinatesAndAtomicSpecies",
             *(" " + " ".join(f"{x:.10f}" for x in p) + f" {species.index(s) + 1}"
               for p, s in zip(positions, symbols)),
             "%endblock AtomicCoordinatesAndAtomicSpecies", "%block kgrid_Monkhorst_Pack",
             *(" " + " ".join(str(k) if i == j else "0" for j in range(3)) + " 0.0" for i in range(3)),
             "%endblock kgrid_Monkhorst_Pack"]
    # Bench settings (g2m_bench_materials.SETTINGS) with a tighter DM tolerance for SOC.
    settings = f"""XC.functional GGA
XC.authors PBE
PAO.BasisSize DZP
PAO.EnergyShift 0.005 Ry
PAO.Polarization.NonPerturbative T
MeshCutoff {mesh} Ry
ElectronicTemperature 0.025 eV
SolutionMethod diagon
MaxSCFIterations 300
DM.Tolerance 1.d-6
DM.MixingWeight 0.10
DM.NumberPulay 6
SCF.MustConverge T
Spin {spin}
Save.HS T
TS.HS.Save T
Write.Ion.XML T
MD.TypeOfRun CG
MD.NumCGsteps 0
MD.Steps 0
{extra}"""
    if spin.startswith("spin-orbit"):
        # SIESTA starts non-collinear runs ferromagnetic (0.36 mu_B here): a one-iteration H0
        # would keep that spurious moment and break time reversal. Start unpolarized.
        settings += "%block DM.InitSpin\n" + "".join(f" {i + 1} 0.0\n" for i in range(5)) + "%endblock DM.InitSpin\n"
    text = "\n".join(lines) + "\n" + settings
    return h0_fdf(text) if CASES[name][4] else text


def run(name: str) -> None:
    work = OUT / name
    if (work / "timing.json").exists():
        return
    while cpu_temp() >= MAX_TEMP:  # ponytail: wait between runs only; a single run is short
        time.sleep(30)
    work.mkdir(parents=True, exist_ok=True)
    (work / "RUN.fdf").write_text(fdf(name))
    for s in ("Bi", "Se"):
        shutil.copy(PSEUDOS / f"{s}.psml", work)
    cmd = ["taskset", "-c", CPUS, "mpirun", "-np", str(NPROC), "--bind-to", "none", SIESTA]
    start = time.time()
    with (work / "RUN.fdf").open() as stdin, (work / "RUN.out").open("w") as stdout:
        subprocess.run(cmd, cwd=work, stdin=stdin, stdout=stdout, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, preexec_fn=lambda: os.nice(10))
    scf = (work / "RUN.out").read_text().count("scf:") - 1
    (work / "timing.json").write_text(json.dumps({"wall_s": time.time() - start, "scf_lines": scf,
                                                  "max_cpu_temp": cpu_temp()}))
    print(f"{name}: {time.time() - start:.0f} s", flush=True)


# --- analysis -------------------------------------------------------------------------------

def read(name):
    import sisl

    # The TSHS carries no species; the fdf (with the .ion.xml files) does.
    geometry = sisl.get_sile(OUT / name / "RUN.fdf").read_geometry()
    sile = sisl.get_sile(OUT / name / "bi2se3.TSHS")
    return sile.read_hamiltonian(geometry=geometry), sile.read_fermi_level()


def channels(H):
    """Real-space SOC channels of a sisl spin-orbit matrix (design §4.1), each (no, no_s) dense."""
    c = [H.tocsr(i).toarray() for i in range(8)]
    trs = {"h0": (c[0] + c[1]) / 2, "gx": (c[3] + c[7]) / 2, "gy": (c[2] - c[6]) / 2, "gz": (c[4] - c[5]) / 2}
    odd = {"Im h0": (c[4] + c[5]) / 2, "Re hx": (c[2] + c[6]) / 2, "Re hy": (c[7] - c[3]) / 2,
           "Re hz": (c[0] - c[1]) / 2}
    return trs, odd


def gamma_analysis(H, n_occ):
    """Gap at Gamma and the Bi weight (Mulliken) of the top valence / bottom conduction state."""
    import scipy.linalg as la

    Hk, Sk = H.Hk(format="array"), H.Sk(format="array")
    e, v = la.eigh(Hk, Sk)
    spinor = Hk.shape[0] == 2 * H.no
    bi = np.array([H.geometry.atoms[H.geometry.o2a(o)].Z == 83 for o in range(H.no)])
    if spinor:
        bi = np.repeat(bi, 2)
    weights = np.real(np.conj(v) * (Sk @ v))
    w_bi = weights[bi].sum(0) / weights.sum(0)
    return {"gap_gamma_eV": float(e[n_occ] - e[n_occ - 1]), "vbm_bi_weight": float(w_bi[n_occ - 1]),
            "cbm_bi_weight": float(w_bi[n_occ]), "vbm_eV": float(e[n_occ - 1]), "cbm_eV": float(e[n_occ])}


def analyze() -> dict:
    import sisl

    out = {"timing": {n: json.loads((OUT / n / "timing.json").read_text()) for n in CASES
                      if (OUT / n / "timing.json").exists()}}
    rng = np.random.default_rng(0)
    ks = [np.zeros(3), *rng.uniform(-.5, .5, (3, 3))]
    for name in CASES:
        if not (OUT / name / "bi2se3.TSHS").exists():
            continue
        H, ef = read(name)
        soc = H.spin.is_spinorbit
        res = {"spin": str(H.spin), "dim": H.dim, "orthogonal": bool(H.orthogonal), "no": H.no,
               "nsc": H.geometry.lattice.nsc.tolist(), "fermi_eV": float(ef)}
        herm = []
        for k in ks:
            Hk = H.Hk(k=k, format="array")
            herm.append(np.abs(Hk - Hk.conj().T).max() / np.abs(Hk).max())
        res["hermiticity_rel_max"] = float(max(herm))
        n_occ = VALENCE if soc else VALENCE // 2
        res["gamma"] = gamma_analysis(H, n_occ)
        if soc:
            trs, odd = channels(H)
            scale = max(np.abs(x).max() for x in trs.values())
            res["trs_odd_rel_max"] = {k: float(np.abs(x).max() / scale) for k, x in odd.items()}
            res["channel_max_abs_eV"] = {k: float(np.abs(x).max()) for k, x in trs.items()}
            split = []
            for k in ks:
                e = H.eigh(k=k)
                split.append(np.abs(e[::2] - e[1::2]).max())
            res["kramers_max_split_eV"] = float(max(split))
        out[name] = res
    if all((OUT / n / "bi2se3.TSHS").exists() for n in ("soc_k6", "h0_soc_k6")):
        (H, _), (H0, _) = read("soc_k6"), read("h0_soc_k6")
        if H.geometry.lattice.nsc.tolist() == H0.geometry.lattice.nsc.tolist():
            g, g0 = channels(H)[0], channels(H0)[0]
            out["delta_hypothesis"] = {k: {"max_abs_H_eV": float(np.abs(g[k]).max()),
                                           "max_abs_H_minus_H0_eV": float(np.abs(g[k] - g0[k]).max()),
                                           "rms_H_minus_H0_eV": float(np.sqrt(np.mean((g[k] - g0[k]) ** 2)))}
                                       for k in ("gx", "gy", "gz", "h0")}
        else:
            out["delta_hypothesis"] = "nsc differs between H and H0; compare on a common support"
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(exist_ok=True)
    if mode in ("run", "all"):
        for name in CASES:
            run(name)
    if mode in ("analyze", "all"):
        print(json.dumps(analyze(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
