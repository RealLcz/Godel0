#!/usr/bin/env python3
"""Supervisor for Godel0 evolve-20 runs: status, failure detection, resubmit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GODEL0_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = GODEL0_ROOT / "logs" / "supervisor_state.json"
DEFAULT_EXCLUDE_NODE = "cn34"


@dataclass
class SupervisorState:
    target_epochs: int = 20
    completed_epochs: int = 0
    phase: str = "preflight"  # preflight | evolve20
    active_job_id: int | None = None
    last_job_id: int | None = None
    run_dir: str | None = None
    config_path: str | None = None
    last_status: str = "unknown"
    last_error: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = ""

    def save(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls) -> SupervisorState:
        if not STATE_PATH.is_file():
            return cls()
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def slurm_state(job_id: int) -> dict[str, str]:
    proc = _run(["sacct", "-j", str(job_id), "--format=JobID,State,ExitCode,Elapsed", "-n", "-P"])
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = line.split("|") if line else []
    if len(parts) >= 4 and parts[0].endswith(str(job_id)):
        return {
            "job_id": str(job_id),
            "state": parts[1],
            "exit_code": parts[2],
            "elapsed": parts[3],
        }
    proc = _run(["squeue", "-j", str(job_id), "-h", "-o", "%T"])
    state = proc.stdout.strip() or "UNKNOWN"
    return {"job_id": str(job_id), "state": state, "exit_code": "", "elapsed": ""}


def parse_log_tail(log_path: Path, n: int = 80) -> str:
    if not log_path.is_file():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def detect_log_issues(text: str) -> list[str]:
    issues: list[str] = []
    patterns = [
        (r"Root bootstrap failed", "root_bootstrap_failed"),
        (r"Proposer batch failed", "proposer_batch_failed"),
        (r"Child build failed", "child_build_failed"),
        (r"Patch guard: Empty patch", "empty_patch"),
        (r"Level 1 failed", "level1_failed"),
        (r"Traceback \(most recent call last\)", "traceback"),
        (r"RuntimeError:", "runtime_error"),
        (r"CUDA out of memory|OutOfMemoryError", "gpu_oom"),
        (r"Engine core initialization failed", "vllm_init_failed"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            issues.append(label)
    return issues


def count_complete_epochs(run_dir: Path) -> int:
    archive = run_dir / "archive.jsonl"
    if not archive.is_file():
        return 0
    count = 0
    for line in archive.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "complete" and row.get("node_id") != "root":
            count += 1
    return count


def root_bootstrap_status(run_dir: Path) -> dict[str, Any]:
    node = run_dir / "nodes" / "root" / "node.json"
    summary = run_dir / "nodes" / "root" / "proposer" / "generation_summary.json"
    feedback_dir = run_dir / "nodes" / "root" / "proposer" / "trusted_feedback"
    out: dict[str, Any] = {"root_status": "missing", "validated_tasks": 0, "batch_complete": False}
    if node.is_file():
        data = json.loads(node.read_text(encoding="utf-8"))
        out["root_status"] = data.get("status")
        out["solved_task_count"] = data.get("solved_task_count")
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        out["validated_tasks"] = len(data.get("task_ids") or [])
        out["batch_complete"] = bool(data.get("complete"))
        out["candidates_validated"] = data.get("candidates_validated")
    if feedback_dir.is_dir():
        out["feedback_files"] = len(list(feedback_dir.glob("*.json")))
    return out


def infer_run_dir(state: SupervisorState) -> Path | None:
    if state.run_dir:
        p = Path(state.run_dir)
        if p.is_dir():
            return p
    if state.active_job_id:
        matches = sorted(GODEL0_ROOT.glob(f"runs_*/*_{state.active_job_id}"))
        if matches:
            return matches[-1]
    return None


def submit_job(
    config_path: Path,
    run_name_prefix: str,
    max_nodes: int | None = None,
    exclude_node: str = DEFAULT_EXCLUDE_NODE,
) -> int:
    env = [
        f"GODEL0_CONFIG={config_path}",
        f"GODEL0_RUN_NAME_PREFIX={run_name_prefix}",
    ]
    if max_nodes is not None:
        env.append(f"GODEL0_MAX_NODES={max_nodes}")
    cmd = [
        "sbatch",
        f"--exclude={exclude_node}",
        str(GODEL0_ROOT / "scripts/slurm/godel0_evolve20_repo_chain.slurm"),
    ]
    proc = subprocess.run(
        ["bash", "-lc", " ".join(env + cmd)],
        cwd=str(GODEL0_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "sbatch failed")
    m = re.search(r"Submitted batch job (\d+)", proc.stdout)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output: {proc.stdout}")
    return int(m.group(1))


def status_report(state: SupervisorState) -> dict[str, Any]:
    report: dict[str, Any] = {
        "phase": state.phase,
        "target_epochs": state.target_epochs,
        "completed_epochs": state.completed_epochs,
        "active_job_id": state.active_job_id,
        "last_status": state.last_status,
        "last_error": state.last_error,
    }
    if state.active_job_id:
        report["slurm"] = slurm_state(state.active_job_id)
        log_path = GODEL0_ROOT / "logs" / f"godel0_evolve20_{state.active_job_id}.log"
        tail = parse_log_tail(log_path)
        report["log_issues"] = detect_log_issues(tail)
        report["log_tail"] = tail.splitlines()[-8:]
    run_dir = infer_run_dir(state)
    if run_dir:
        report["run_dir"] = str(run_dir)
        report["epochs_in_run"] = count_complete_epochs(run_dir)
        report["root"] = root_bootstrap_status(run_dir)
        state.completed_epochs = max(state.completed_epochs, report["epochs_in_run"])
    return report


def cmd_status(_: argparse.Namespace) -> int:
    state = SupervisorState.load()
    print(json.dumps(status_report(state), indent=2))
    return 0


def cmd_check(_: argparse.Namespace) -> int:
    state = SupervisorState.load()
    report = status_report(state)
    print(json.dumps(report, indent=2))

    job_id = state.active_job_id
    if not job_id:
        return 0

    slurm = report.get("slurm", {})
    terminal = slurm.get("state") in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}
    issues = report.get("log_issues") or []

    if terminal:
        exit_code = slurm.get("exit_code", "")
        if slurm.get("state") != "COMPLETED" or exit_code not in {"", "0:0", "0"}:
            state.last_error = f"job_{job_id}_{slurm.get('state')}_{exit_code}"
            state.last_status = "failed"
            state.history.append({"event": "job_failed", "job_id": job_id, "slurm": slurm, "issues": issues})
            state.active_job_id = None
            state.save()
            print(f"SUPERVISOR_ALERT job_failed {json.dumps({'job_id': job_id, 'issues': issues})}")
            return 2

        run_dir = infer_run_dir(state)
        epochs = count_complete_epochs(run_dir) if run_dir else 0
        root = report.get("root", {})
        log_text = parse_log_tail(GODEL0_ROOT / "logs" / f"godel0_evolve20_{job_id}.log")
        match = re.search(r"Successful epochs: (\d+)", log_text)
        successful = int(match.group(1)) if match else 0

        state.completed_epochs = max(state.completed_epochs, epochs, successful)
        state.last_status = "completed"
        state.last_job_id = job_id
        state.active_job_id = None
        state.history.append({"event": "job_completed", "job_id": job_id, "epochs": state.completed_epochs})
        state.save()
        print(f"SUPERVISOR_ALERT job_completed {json.dumps({'job_id': job_id, 'epochs': state.completed_epochs})}")
        return 3

    if issues:
        print(f"SUPERVISOR_ALERT log_issues {json.dumps({'job_id': job_id, 'issues': issues})}")
        return 1
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    state = SupervisorState.load()
    state.target_epochs = args.target_epochs
    state.phase = args.phase
    state.active_job_id = args.job_id
    state.config_path = args.config
    if args.job_id:
        run_dir = infer_run_dir(state)
        if run_dir:
            state.run_dir = str(run_dir)
    state.save()
    print(json.dumps(asdict(state), indent=2))
    return 0


def cmd_submit_next(args: argparse.Namespace) -> int:
    state = SupervisorState.load()
    if state.active_job_id and slurm_state(state.active_job_id).get("state") in {"RUNNING", "PENDING"}:
        print(f"Job {state.active_job_id} still active; not submitting")
        return 1

    if state.phase == "preflight":
        config = GODEL0_ROOT / "configs/ansible_joint_epoch1_preflight.yaml"
        prefix = "ansible_joint_epoch1_preflight"
    else:
        config = GODEL0_ROOT / "configs/evolve20_ansible_formal.yaml"
        prefix = "ansible_evolve20_joint_v1"

    job_id = submit_job(config, prefix, exclude_node=args.exclude_node)
    state.active_job_id = job_id
    state.last_status = "submitted"
    state.config_path = str(config)
    state.history.append({"event": "submitted", "job_id": job_id, "phase": state.phase})
    state.save()
    print(json.dumps({"submitted_job_id": job_id, "phase": state.phase}, indent=2))
    return 0


def cmd_watch_once(_: argparse.Namespace) -> int:
    """Single watchdog tick: check status, auto-advance phase, resubmit on success."""
    state_before = SupervisorState.load()
    phase_before = state_before.phase
    rc = cmd_check(argparse.Namespace())
    state = SupervisorState.load()

    if rc == 3 and phase_before == "preflight":
        state.phase = "evolve20"
        state.save()
        print("SUPERVISOR_ACTION preflight_ok_advancing_to_evolve20")
        submit_rc = cmd_submit_next(argparse.Namespace(exclude_node=DEFAULT_EXCLUDE_NODE))
        if submit_rc == 0:
            print("SUPERVISOR_ACTION submitted_evolve20")
        return submit_rc

    if rc == 2:
        print("SUPERVISOR_ACTION job_failed_needs_fix")
        return 2

    if rc == 3 and phase_before == "evolve20":
        if state.completed_epochs >= state.target_epochs:
            print("SUPERVISOR_ACTION all_epochs_complete")
            return 0
        if not state.active_job_id:
            print("SUPERVISOR_ACTION evolve20_incomplete_resubmit")
            return cmd_submit_next(argparse.Namespace(exclude_node=DEFAULT_EXCLUDE_NODE))

    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervise Godel0 evolve-20 runs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status")
    p_status.set_defaults(func=cmd_status)

    p_check = sub.add_parser("check")
    p_check.set_defaults(func=cmd_check)

    p_init = sub.add_parser("init")
    p_init.add_argument("--job-id", type=int, required=True)
    p_init.add_argument("--phase", choices=["preflight", "evolve20"], default="preflight")
    p_init.add_argument("--target-epochs", type=int, default=20)
    p_init.add_argument("--config", default="")
    p_init.set_defaults(func=cmd_init)

    p_submit = sub.add_parser("submit-next")
    p_submit.add_argument("--exclude-node", default=DEFAULT_EXCLUDE_NODE)
    p_submit.set_defaults(func=cmd_submit_next)

    p_watch = sub.add_parser("watch-once")
    p_watch.set_defaults(func=cmd_watch_once)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
