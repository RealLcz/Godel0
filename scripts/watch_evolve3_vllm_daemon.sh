#!/usr/bin/env bash
# Persistent watchdog for the Gödel0 3-epoch resume run.
#
# Survives Cursor disconnect. Watches the active Slurm job; if it dies
# because vLLM failed to start (port bind, CUDA OOM, engine init, etc.),
# automatically resubmits the resume Slurm script so task_store / run_dir
# progress is reused.
#
# Start (detach from Cursor):
#   cd /mnt/vast/workspaces/Emergent_Topology/Godel0
#   nohup bash scripts/watch_evolve3_vllm_daemon.sh --job-id 216897 \
#     >> /mnt/vast/workspaces/Emergent_Topology/Godel0/godel0_evolve20_pass1/logs/evolve3_vllm_watch.out 2>&1 &
#
# Stop:
#   kill "$(cat /mnt/vast/workspaces/Emergent_Topology/Godel0/godel0_evolve20_pass1/logs/evolve3_vllm_watch.pid)"
set -uo pipefail

GODEL0_ROOT="${GODEL0_ROOT:-/mnt/vast/workspaces/Emergent_Topology/Godel0}"
WS="${GODEL0_WS:-/mnt/vast/workspaces/Emergent_Topology/Godel0/godel0_evolve20_pass1}"
# Keep daemon logs on home (workspaces previously hit Disk quota exceeded).
LOG_DIR="${WATCH_LOG_DIR:-${WS}/logs}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${WS}/godel0_evolve3_existing20.slurm}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-${GODEL0_ROOT}/runs_ansible_evolve3_existing20/ansible_evolve3_existing20_216679}"
STATE_FILE="${LOG_DIR}/evolve3_vllm_watch_state.json"
STATUS_LOG="${LOG_DIR}/evolve3_vllm_watch_status.log"
PID_FILE="${LOG_DIR}/evolve3_vllm_watch.pid"
INTERVAL_SEC="${WATCH_INTERVAL_SEC:-120}"
MAX_RESUBMITS="${WATCH_MAX_RESUBMITS:-20}"
COOLDOWN_SEC="${WATCH_COOLDOWN_SEC:-180}"
TARGET_EPOCHS="${WATCH_TARGET_EPOCHS:-3}"

SEED_JOB_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id) SEED_JOB_ID="$2"; shift 2 ;;
    --interval) INTERVAL_SEC="$2"; shift 2 ;;
    --max-resubmits) MAX_RESUBMITS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${LOG_DIR}"
echo "$$" > "${PID_FILE}"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "${STATUS_LOG}"
}

save_state() {
  local active="$1"
  local last="$2"
  local resubmits="$3"
  local reason="$4"
  python3 - "${STATE_FILE}" "${active}" "${last}" "${resubmits}" "${reason}" "${RESUME_RUN_DIR}" <<'PY'
import json, sys
from pathlib import Path
from datetime import datetime, timezone
path = Path(sys.argv[1])
data = {
    "active_job_id": sys.argv[2],
    "last_job_id": sys.argv[3],
    "resubmits": int(sys.argv[4]),
    "last_reason": sys.argv[5],
    "resume_run_dir": sys.argv[6],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

load_field() {
  local key="$1"
  python3 - "${STATE_FILE}" "${key}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
key = sys.argv[2]
if not p.is_file():
    print("")
    raise SystemExit
d = json.loads(p.read_text() or "{}")
v = d.get(key, "")
print("" if v is None else v)
PY
}

job_state() {
  local jid="$1"
  local st
  st=$(squeue -j "${jid}" -h -o "%T" 2>/dev/null | head -1 || true)
  if [[ -n "${st}" ]]; then
    echo "${st}"
    return
  fi
  sacct -j "${jid}" --format=State -n -P 2>/dev/null | head -1 | tr -d ' ' || echo "UNKNOWN"
}

job_exit() {
  local jid="$1"
  sacct -j "${jid}" --format=ExitCode -n -P 2>/dev/null | head -1 | tr -d ' ' || echo ""
}

is_terminal_success() {
  local jid="$1"
  local st exitc
  st=$(job_state "${jid}")
  exitc=$(job_exit "${jid}")
  if [[ "${st}" == "COMPLETED" && ( "${exitc}" == "0:0" || "${exitc}" == "0" ) ]]; then
    return 0
  fi
  local out="${LOG_DIR}/godel0_evolve3_${jid}.out"
  local runlog="${LOG_DIR}/godel0_evolve3_${jid}.log"
  if grep -q "Evolve-3 verification complete" "${out}" 2>/dev/null; then
    return 0
  fi
  if grep -q "Evolution complete" "${runlog}" 2>/dev/null; then
    return 0
  fi
  return 1
}

epochs_done() {
  python3 - "${RESUME_RUN_DIR}" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
archive = run / "archive.jsonl"
n = 0
if archive.is_file():
    for line in archive.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("node_id") == "root":
            continue
        if str(row.get("status", "")).lower() == "complete":
            n += 1
print(n)
PY
}

# Return 0 iff the failure looks like a vLLM / infra crash (safe to auto-retry).
is_vllm_crash() {
  local jid="$1"
  local out="${LOG_DIR}/godel0_evolve3_${jid}.out"
  local err="${LOG_DIR}/godel0_evolve3_${jid}.err"
  local vllm_err="${LOG_DIR}/vllm_evolve3_${jid}.stderr.log"
  local vllm_out="${LOG_DIR}/vllm_evolve3_${jid}.stdout.log"
  local runlog="${LOG_DIR}/godel0_evolve3_${jid}.log"
  local pattern='Address already in use|CUDA error: out of memory|Engine core initialization failed|WorkerProc initialization failed|cudaErrorMemoryAllocation|Failed to initialize|OSError: \[Errno 98\]|Disk quota exceeded|Connection refused|RemoteProtocolError|APIConnectionError|CUDA out of memory'

  # vLLM never became ready → infra failure (retry).
  if [[ -f "${out}" ]] && ! grep -q "vLLM ready" "${out}" 2>/dev/null; then
    return 0
  fi

  if grep -Eiq "${pattern}" "${out}" "${err}" "${vllm_err}" "${vllm_out}" "${runlog}" 2>/dev/null; then
    return 0
  fi

  return 1
}

maybe_resubmit() {
  local old_jid="$1"
  local reason="$2"

  if [[ "${RESUBMITS}" -ge "${MAX_RESUBMITS}" ]]; then
    log "ERROR: hit max_resubmits=${MAX_RESUBMITS}; stopping"
    return 1
  fi

  local now
  now=$(date +%s)
  if [[ "${LAST_RESUBMIT_TS}" -gt 0 && $((now - LAST_RESUBMIT_TS)) -lt "${COOLDOWN_SEC}" ]]; then
    log "Cooldowning before resubmit ($((COOLDOWN_SEC - (now - LAST_RESUBMIT_TS)))s left)"
    return 2
  fi

  if [[ ! -f "${SLURM_SCRIPT}" ]]; then
    log "ERROR: Slurm script missing: ${SLURM_SCRIPT}"
    return 1
  fi

  log "Resubmitting after job ${old_jid} failure (${reason})"
  local submit_out new_jid
  submit_out=$(sbatch "${SLURM_SCRIPT}" 2>&1) || {
    log "ERROR: sbatch failed: ${submit_out}"
    return 1
  }
  new_jid=$(echo "${submit_out}" | awk '{print $NF}')
  if [[ -z "${new_jid}" || ! "${new_jid}" =~ ^[0-9]+$ ]]; then
    log "ERROR: could not parse new job id from: ${submit_out}"
    return 1
  fi

  RESUBMITS=$((RESUBMITS + 1))
  LAST_RESUBMIT_TS=$(date +%s)
  ACTIVE_JOB_ID="${new_jid}"
  save_state "${ACTIVE_JOB_ID}" "${old_jid}" "${RESUBMITS}" "${reason}"
  log "Submitted new job ${new_jid} (resume_dir=${RESUME_RUN_DIR}, resubmits=${RESUBMITS})"
  return 0
}

handle_failure() {
  local jid="$1"
  local state="$2"

  if is_terminal_success "${jid}"; then
    log "Job ${jid} ended with success marker; exiting"
    return 10
  fi

  if [[ "${state}" == "CANCELLED" ]] && ! is_vllm_crash "${jid}"; then
    log "Job ${jid} CANCELLED without vLLM crash signature; stopping"
    return 10
  fi

  if ! is_vllm_crash "${jid}"; then
    log "Job ${jid} failed, but failure does not look like vLLM/infra; NOT auto-resubmitting"
    log "Inspect: ${LOG_DIR}/godel0_evolve3_${jid}.out / .log"
    return 10
  fi

  maybe_resubmit "${jid}" "vllm_crash:${state}"
  return $?
}

# Bootstrap state
ACTIVE_JOB_ID="${SEED_JOB_ID}"
if [[ -z "${ACTIVE_JOB_ID}" ]]; then
  ACTIVE_JOB_ID=$(load_field active_job_id)
fi
if [[ -z "${ACTIVE_JOB_ID}" ]]; then
  ACTIVE_JOB_ID=$(squeue -u "${USER}" -h -o "%i %j" 2>/dev/null | awk '$2=="godel0-e3"{print $1; exit}')
fi
if [[ -z "${ACTIVE_JOB_ID}" ]]; then
  ACTIVE_JOB_ID=$(ls -1t "${LOG_DIR}"/godel0_evolve3_*.out 2>/dev/null | head -1 | sed -E 's/.*godel0_evolve3_([0-9]+)\.out/\1/')
fi

RESUBMITS=$(load_field resubmits)
RESUBMITS="${RESUBMITS:-0}"
[[ "${RESUBMITS}" =~ ^[0-9]+$ ]] || RESUBMITS=0
LAST_RESUBMIT_TS=0

if [[ -z "${ACTIVE_JOB_ID}" ]]; then
  log "No job id found; submitting initial resume job"
  if ! maybe_resubmit "none" "initial_submit"; then
    exit 1
  fi
else
  save_state "${ACTIVE_JOB_ID}" "" "${RESUBMITS}" "watch_start"
fi

log "daemon started pid=$$ watching job=${ACTIVE_JOB_ID} interval=${INTERVAL_SEC}s max_resubmits=${MAX_RESUBMITS}"
log "resume_run_dir=${RESUME_RUN_DIR} slurm=${SLURM_SCRIPT}"

while true; do
  EPOCHS=$(epochs_done)
  if [[ "${EPOCHS}" -ge "${TARGET_EPOCHS}" ]]; then
    log "DONE: ${EPOCHS}/${TARGET_EPOCHS} successful epochs in archive; exiting"
    break
  fi

  STATE=$(job_state "${ACTIVE_JOB_ID}")
  EXITC=$(job_exit "${ACTIVE_JOB_ID}")
  log "job=${ACTIVE_JOB_ID} state=${STATE} exit=${EXITC} epochs=${EPOCHS}/${TARGET_EPOCHS} resubmits=${RESUBMITS}"

  case "${STATE}" in
    PENDING|CONFIGURING|RUNNING|COMPLETING)
      ;;
    COMPLETED)
      if is_terminal_success "${ACTIVE_JOB_ID}"; then
        log "Job ${ACTIVE_JOB_ID} completed successfully; exiting"
        break
      fi
      handle_failure "${ACTIVE_JOB_ID}" "${STATE}"
      rc=$?
      if [[ "${rc}" -eq 10 ]]; then
        break
      fi
      ;;
    FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|CANCELLED)
      handle_failure "${ACTIVE_JOB_ID}" "${STATE}"
      rc=$?
      if [[ "${rc}" -eq 10 || "${rc}" -eq 1 ]]; then
        break
      fi
      # rc 2 = cooldown; fall through to sleep
      ;;
    *)
      log "Unknown state=${STATE}; continuing to poll"
      ;;
  esac

  sleep "${INTERVAL_SEC}"
done

log "daemon exiting pid=$$"
rm -f "${PID_FILE}"
