"""CPU closure audit of saved matrices; no inference, fit, queue writes or cutoffs."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse import load_npz

import relaunch_h2oclu_nullspace as recovery
from diagnose_h2oclu_ood import read_references, symmetric
from h2oclu_batching_metrics import matrix_metrics


def levels(H, S):
    """Independent full-basis Cholesky route, instead of generalized scipy.eigh."""
    L = np.linalg.cholesky(symmetric(S))
    B = solve_triangular(L, symmetric(H), lower=True)
    A = solve_triangular(L, B.T, lower=True).T
    return np.linalg.eigvalsh(symmetric(A))


def equal_metrics(actual, saved):
    for key, value in actual.items():
        if isinstance(value, dict):
            equal_metrics(value, saved[key])
        elif isinstance(value, (str, bool)) or value is None:
            assert value == saved[key], key
        else:
            np.testing.assert_allclose(value, saved[key], atol=1e-7, rtol=1e-10, err_msg=key)


def audit(root):
    # Analytic exact-reference and cS controls exercise the independent solver.
    S = np.diag([.002, .5, 1.])
    H = np.diag([-.004, .5, 4.])
    np.testing.assert_allclose(levels(H, S), [-2., 1., 4.], atol=1e-12)
    np.testing.assert_allclose(levels(H+2*S, S), [0., 3., 6.], atol=1e-12)
    a = recovery
    m, relaunch = a.read_relaunch(root)
    original = json.loads((root/"queue_receipt.json").read_text())
    queue = Path(relaunch["queue"])
    raw = queue.read_bytes()
    state = json.loads(queue.with_name("state.json").read_text())
    entries = [json.loads(x) for x in raw.splitlines() if x.strip()]
    for receipt in (original, relaunch):
        assert receipt["complete"]
        assert hashlib.sha256(raw[:receipt["old_bytes"]]).hexdigest() == receipt["old_sha256"]
        for entry in receipt["jobs"]:
            assert [x for x in entries if x["id"] == entry["id"]] == [entry]
    assert all(j["id"] in state["done"] and j["id"] not in state["failed"] for j in relaunch["jobs"])
    for j in relaunch["jobs"]:
        p = Path(f"/proc/{state['pids'].get(j['id'])}/cmdline")
        assert not p.exists() or str(root).encode() not in p.read_bytes(), "own process still live"
    references, metadata = read_references(m)
    summary = json.loads((root/"summary.json").read_text())
    files = {**m["files"], **relaunch["files"]}
    for path in (Path(__file__).resolve(), root/"summary.json", root/"relaunch_receipt.json"):
        files[str(path)] = a.sha256(path)
    results, records, costs, checkpoints = {}, [], [], []
    max_spectral_difference = 0.
    for arm, jobs in (("fixed", m["baselines"]), ("null", m["jobs"])):
        for job in jobs:
            directory = Path(job["root"])
            result = (json.loads(Path(job["result"]).read_text()) if arm == "fixed"
                      else a.campaign.checked_result(m, job))
            fit = result["training"]
            assert fit["samples_committed"] == 64000 and fit["steps"] == 8000
            assert len(fit["epochs"]) == 2000
            assert all(e["samples"] == 32 and e["updates"] == 4 for e in fit["epochs"])
            for path in (directory/"result.json", directory/"fit.json", directory/"evaluation_receipt.json",
                         Path(fit["checkpoint"]), Path(fit["recovery_checkpoint"])):
                files[str(path)] = a.sha256(path)
            if arm == "null":
                import torch
                checkpoint = torch.load(fit["checkpoint"], map_location="cpu", weights_only=False)
                hp = checkpoint["hyper_parameters"]
                assert hp["loss_kwargs"] == {"spectral_weight": 1e-4} and hp["n_output_components"] == 1
                callbacks = [v for v in checkpoint["callbacks"].values() if v.get("monitor") == "val_loss"]
                assert len(callbacks) == 1 and callbacks[0]["best_model_path"] == fit["checkpoint"]
                np.testing.assert_allclose(float(callbacks[0]["best_model_score"]), fit["val_loss"], atol=1e-12)
                checkpoints.append(dict(seed=job["seed"], selected_epoch=checkpoint["epoch"],
                    selected_step=checkpoint["global_step"], basic_val_loss=fit["val_loss"],
                    checkpoint=fit["checkpoint"], sha256=a.sha256(fit["checkpoint"])))
            attempts = json.loads((directory/"attempts.json").read_text())
            assert len(attempts) == 1 and attempts[0]["exit_reason"] == "success"
            telemetry = json.loads(Path(attempts[0]["telemetry"]).read_text())
            pss = [t["host"]["pss_sum_mib"] for t in telemetry if t["host"]["pss_sum_mib"] is not None]
            assert pss and all(np.isfinite(v) for v in pss)
            for path in (directory/"attempts.json", Path(attempts[0]["telemetry"])):
                files[str(path)] = a.sha256(path)
            costs.append(dict(arm=arm, seed=job["seed"], seconds=fit["seconds_all_attempts"],
                allocated_mib=fit["peak_allocated_mib"], reserved_mib=fit["peak_reserved_mib"],
                host_pss_peak_mib=max(pss), host_pss_missing_samples=len(telemetry)-len(pss),
                min_free_fraction=min(t["memory.free"]/t["memory.total"] for t in telemetry),
                max_gpu_temperature=max(t["temperature.gpu"] for t in telemetry),
                evaluation_cpu_seconds=result["evaluation"]["seconds"], attempts=1))
            results[arm, job["seed"]] = result
            for row in result["evaluation"]["rows"]:
                path = Path(row["matrix"])
                assert a.sha256(path) == row["matrix_sha256"]
                files[str(path)] = row["matrix_sha256"]
                P = load_npz(path).toarray()
                frame, owners, species, _ = references[row["sample_id"]]
                computed = matrix_metrics(P, frame["H"], frame["S"], owners, species)
                assert computed["spectral_valid"]
                equal_metrics(computed, row)
                ref = levels(frame["H"], frame["S"])
                pred = levels(P, frame["S"])-row["gauge_c_eV"]
                for calculated, stored in ((ref, row["reference_levels_eV"]),
                                           (pred, row["predicted_aligned_levels_eV"])):
                    assert len(calculated) == metadata[row["sample_id"]]["orbitals"]
                    max_spectral_difference = max(max_spectral_difference, float(np.max(abs(calculated-stored))))
                    np.testing.assert_allclose(calculated, stored, atol=2e-8, rtol=0)
                records.append(dict(arm=arm, seed=job["seed"], **{k: v for k, v in row.items()
                    if k not in ("reference_levels_eV", "predicted_aligned_levels_eV")}))
    verified_summary = {}
    regressions = []
    for split in ("validation", "test"):
        for metric in a.campaign.science.PRIMARY:
            pairs = [np.array([np.mean([r[metric] for r in results[arm, seed]["evaluation"]["rows"]
                                      if r["split"] == split]) for seed in range(3)]) for arm in ("fixed", "null")]
            f, n = pairs
            ratios = [n[list(ix)].mean()/f[list(ix)].mean() for ix in itertools.product(range(3), repeat=3)]
            calculated = dict(fixed_mean=float(f.mean()), null_mean=float(n.mean()), ratio_of_means=float(n.mean()/f.mean()),
                fixed_per_seed=f.tolist(), null_per_seed=n.tolist(),
                paired_seeds=[0, 1, 2], paired_ratios=(n/f).tolist(), paired_count=3, excluded_pairs=[],
                paired_bootstrap_ratio_ci95=np.quantile(ratios, [.025, .975]).tolist())
            equal_metrics(calculated, summary["metrics"][split+"/"+metric])
            verified_summary[split+"/"+metric] = calculated
    for seed in range(3):
        baseline = {r["sample_id"]: r for r in results["fixed", seed]["evaluation"]["rows"]}
        for row in results["null", seed]["evaluation"]["rows"]:
            for metric in a.campaign.science.PRIMARY:
                ratio = row[metric]/baseline[row["sample_id"]][metric]
                if ratio > 1.05:
                    regressions.append(dict(seed=seed, sample_id=row["sample_id"], split=row["split"],
                        metric=metric, ratio=ratio))
    assert len(records) == 42 and summary["completed"] == 3
    a.verify_files(files)
    a.read_relaunch(root)
    return dict(status="verified", created_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=a.sha256(root/"manifest.json"), files=files, references=metadata,
        candidate_checkpoints=checkpoints, matrix_count=42, candidate_matrix_count=21,
        max_independent_Cholesky_level_difference_eV=max_spectral_difference,
        metrics=verified_summary, rows=records, costs=costs, regressions_over_5_percent=regressions,
        queue_effective_ids=[j["id"] for j in relaunch["jobs"]], queue_prefixes_verified=True,
        limitations="3seeds, observed OOD/incomplete lineage, reference-S conditional, noncontemporaneous costs; no default promotion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=recovery.recovery.REPO/"auditorias_mejoras/experimentos_nullspace_h2oclu_20261017_v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = recovery.recovery.own_output(args.out)
    if output.exists():
        raise ValueError("audit output already exists; choose a new output path")
    report = audit(recovery.recovery.own_output(args.root))
    recovery.campaign.write_json(output, report)
    print("VERIFIED", report["matrix_count"], "matrices; max independent level discrepancy eV",
          report["max_independent_Cholesky_level_difference_eV"], "files", len(report["files"]))
