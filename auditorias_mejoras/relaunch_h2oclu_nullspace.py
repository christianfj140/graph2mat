"""Recover the existing campaign after its CPU evaluator import-order failure."""
import argparse
import hashlib
import json
from pathlib import Path

import run_cost_batching_comparison as recovery

# The frozen launcher imports its callback before its own extension bootstrap.
recovery.recovery_api()
import run_h2oclu_nullspace as campaign  # noqa: E402

from run_readout_comparison import append_queue, sha256, verify_files  # noqa: E402


def read_relaunch(root):
    root = recovery.own_output(root)
    m = campaign.read_campaign(root)
    receipt = json.loads(recovery.own_output(root/"relaunch_receipt.json").read_text())
    verify_files(receipt["files"])
    if receipt["authorization"] != "Corrige y relanza" or len(receipt["jobs"]) != 3:
        raise ValueError("three existing seeds only authorized for relaunch")
    for index, (entry, job) in enumerate(zip(receipt["jobs"], m["jobs"])):
        if (entry["id"] != f"{root.name}:{job['key']}:bootstrap1"
                or entry["script"] != str(Path(__file__).resolve())
                or entry["args"] != ["run", "--root", str(root), "--key", job["key"]]
                or entry["vram_mib"] != 4096 or entry["min_free_vram_frac"] != .30
                or (index and entry["after"] != [receipt["jobs"][index-1]["id"]])):
            raise ValueError("relaunch identity/command/dependencies changed")
    return m, receipt


def enqueue(root, queue=campaign.science.QUEUE, approved=False):
    if not approved:
        raise ValueError("explicit relaunch authorization required")
    root = recovery.own_output(root)
    m = campaign.read_campaign(root)
    queue = Path(queue)
    path = recovery.own_output(root/"relaunch_receipt.json")
    if not path.exists():
        old = json.loads((root/"queue_receipt.json").read_text())
        state = json.loads(queue.with_name("state.json").read_text())
        if (old["jobs"][0]["id"] not in state["failed"]
                or any(j["id"] in state["done"] for j in old["jobs"][1:])):
            raise ValueError("expected failed evaluation0 and two pending original seeds")
        for j in old["jobs"]:
            pid = state["pids"].get(j["id"])
            command = Path(f"/proc/{pid}/cmdline")
            if command.exists() and str(root).encode() in command.read_bytes():
                raise ValueError("original campaign still running")
        fit_path = recovery.own_output(root/"null_s0/fit.json")
        fit = json.loads(fit_path.read_text())
        checkpoint = recovery.own_output(fit["checkpoint"])
        if (fit["samples_committed"] != 64000 or fit["steps"] != 8000
                or root/"null_s0" not in checkpoint.parents):
            raise ValueError("completed seed0 must be preserved, not retrained")
        files = {str(p): sha256(p) for p in (Path(__file__).resolve(), root/"manifest.json",
                 root/"queue_receipt.json", fit_path, checkpoint)}
        previous = queue.read_bytes()
        dependencies = campaign.live_dependencies(
            [json.loads(x) for x in previous.splitlines() if x.strip()], state)
        jobs = []
        for job in m["jobs"]:
            identifier = f"{root.name}:{job['key']}:bootstrap1"
            jobs.append(dict(id=identifier, script=str(Path(__file__).resolve()),
                args=["run", "--root", str(root), "--key", job["key"]],
                after=dependencies, vram_mib=4096, min_free_vram_frac=.30))
            dependencies = [identifier]
        campaign.write_json(path, dict(authorization="Corrige y relanza", queue=str(queue),
            old_bytes=len(previous), old_sha256=hashlib.sha256(previous).hexdigest(), files=files,
            jobs=jobs, complete=False, appended_ids=[]))
    _, receipt = read_relaunch(root)
    if receipt["queue"] != str(queue):
        raise ValueError("relaunch queue changed")
    for entry in receipt["jobs"]:
        if hashlib.sha256(queue.read_bytes()[:receipt["old_bytes"]]).hexdigest() != receipt["old_sha256"]:
            raise ValueError("previous queue prefix changed")
        append_queue(queue, entry)
        if entry["id"] not in receipt["appended_ids"]:
            receipt["appended_ids"].append(entry["id"])
        campaign.write_json(path, receipt)
    receipt["complete"] = True
    campaign.write_json(path, receipt)
    return receipt


def run(root, key):
    root = recovery.own_output(root)
    m, _ = read_relaunch(root)
    job = next(j for j in m["jobs"] if j["key"] == key)
    directory = Path(job["root"])
    evidence = any(p.exists() for p in campaign.evaluation_paths(directory))
    if not evidence and not (directory/"fit.json").exists():
        recovery.guarded_fit(Path(job["config"]), directory)
    # No new CLI subprocess: this process already bootstrapped the scalar kernels.
    # Frozen evaluate validates completion/selection and refuses partial outputs.
    campaign.evaluate(root, key)
    campaign.summarize(root)
    print("DONE", key, "bootstrap1", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("enqueue", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key", choices=("null_s0", "null_s1", "null_s2"))
    parser.add_argument("--approved-relaunch", action="store_true")
    args = parser.parse_args()
    if args.action == "enqueue":
        enqueue(args.root, approved=args.approved_relaunch)
    else:
        run(args.root, args.key)
