#!/usr/bin/env bash
# Persistent daemon: monitors Godel0 evolve-20 until target_epochs reached.
# Emits AGENT_LOOP_WAKE_evolve20 lines for Cursor agent wake-ups.
set -uo pipefail

GODEL0_ROOT="${GODEL0_ROOT:-/mnt/vast/workspaces/cot_pixel_prediction/jinhe/godel0}"
INTERVAL_SEC="${WATCH_INTERVAL_SEC:-180}"
WAKE_LOG="${GODEL0_ROOT}/logs/evolve20_watch_wake.log"
STATUS_LOG="${GODEL0_ROOT}/logs/evolve20_watch_status.log"
PID_FILE="${GODEL0_ROOT}/logs/evolve20_watch.pid"

cd "${GODEL0_ROOT}"
mkdir -p logs

echo "$$" > "${PID_FILE}"

log_status() {
    echo "[$(date -Iseconds)] $*" >> "${STATUS_LOG}"
}

emit_wake() {
    local reason="$1"
    local summary="$2"
    # Escape for JSON string
    local esc
    esc=$(printf '%s' "${summary}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read())[1:-1])')
    echo "AGENT_LOOP_WAKE_evolve20 {\"reason\":\"${reason}\",\"prompt\":\"Godel0 evolve-20 watchdog: ${esc} Run: cd ${GODEL0_ROOT} && python3 scripts/supervise_evolve20.py watch-once. Fix bugs and resubmit if failed. Do NOT stop until 20 epochs complete. Re-arm daemon if not running.\"}" >> "${WAKE_LOG}"
    log_status "WAKE reason=${reason} summary=${summary}"
}

log_status "daemon started pid=$$ interval=${INTERVAL_SEC}s"

LAST_LOG_MTIME=0
LAST_STATE=""
LAST_VALIDATED=-1
LAST_EPOCHS=-1
LAST_WAKE_TS=0

while true; do
    CHANGED=0
    REASON=""

    # Slurm state
    JOB_ID=$(python3 -c "
import json
from pathlib import Path
p = Path('logs/supervisor_state.json')
if p.is_file():
    d = json.loads(p.read_text())
    print(d.get('active_job_id') or '')
" 2>/dev/null || true)

    if [[ -n "${JOB_ID}" ]]; then
        STATE=$(sacct -j "${JOB_ID}" --format=State -n -P 2>/dev/null | head -1 || squeue -j "${JOB_ID}" -h -o %T 2>/dev/null || echo UNKNOWN)
        if [[ "${STATE}" != "${LAST_STATE}" ]]; then
            LAST_STATE="${STATE}"
            CHANGED=1
            REASON="slurm_state_${STATE}"
        fi

        LOG_FILE="${GODEL0_ROOT}/logs/godel0_evolve20_${JOB_ID}.log"
        if [[ -f "${LOG_FILE}" ]]; then
            CUR=$(stat -c %Y "${LOG_FILE}" 2>/dev/null || echo 0)
            if [[ "${CUR}" != "${LAST_LOG_MTIME}" ]]; then
                LAST_LOG_MTIME="${CUR}"
                CHANGED=1
                REASON="log_updated"
            fi
        fi
    fi

    # Proposer / epoch progress from run artifacts
    VALIDATED=$(find "${GODEL0_ROOT}"/runs_*/*_"${JOB_ID}"/nodes/root/proposer/trusted_feedback -name "*.json" 2>/dev/null | wc -l || echo 0)
    if [[ "${VALIDATED}" != "${LAST_VALIDATED}" ]]; then
        LAST_VALIDATED="${VALIDATED}"
        CHANGED=1
        REASON="validated_${VALIDATED}"
    fi

    EPOCHS=$(python3 -c "
import json
from pathlib import Path
p = Path('logs/supervisor_state.json')
if not p.is_file(): print(0); exit()
d = json.loads(p.read_text())
target = d.get('target_epochs', 20)
completed = d.get('completed_epochs', 0)
# also scan archives
from pathlib import Path as P
for archive in sorted(P('.').glob('runs_*/*/archive.jsonl')):
    if '${JOB_ID}' in str(archive):
        for line in archive.read_text().splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            if row.get('status')=='complete' and row.get('node_id')!='root':
                completed = max(completed, sum(1 for l in archive.read_text().splitlines() if json.loads(l).get('status')=='complete' and json.loads(l).get('node_id')!='root'))
        break
print(completed)
" 2>/dev/null || echo 0)

    if [[ "${EPOCHS}" != "${LAST_EPOCHS}" ]]; then
        LAST_EPOCHS="${EPOCHS}"
        CHANGED=1
        REASON="epochs_${EPOCHS}"
    fi

    NOW=$(date +%s)
    ELAPSED=$(( NOW - LAST_WAKE_TS ))

    # Force heartbeat every interval even if nothing changed
    if [[ "${ELAPSED}" -ge "${INTERVAL_SEC}" ]]; then
        CHANGED=1
        REASON="heartbeat"
    fi

    if [[ "${CHANGED}" -eq 1 ]]; then
        CHECK_OUT=$(python3 "${GODEL0_ROOT}/scripts/supervise_evolve20.py" watch-once 2>&1 || true)
        SUMMARY=$(echo "${CHECK_OUT}" | python3 -c "
import json,sys
try:
    d=json.loads(sys.stdin.read())
    s=d.get('slurm',{})
    r=d.get('root',{})
    print(f\"job={d.get('active_job_id')} state={s.get('state')} epochs={d.get('completed_epochs')}/{d.get('target_epochs')} validated={r.get('validated_tasks',0)}/10 issues={d.get('log_issues',[])}\")
except Exception:
    print(sys.stdin.read()[:200])
" 2>/dev/null || echo "check_failed")
        emit_wake "${REASON}" "${SUMMARY}"
        LAST_WAKE_TS="${NOW}"
    fi

    # Done?
    TARGET=$(python3 -c "import json; print(json.load(open('logs/supervisor_state.json')).get('target_epochs',20))" 2>/dev/null || echo 20)
    if [[ "${EPOCHS}" -ge "${TARGET}" ]] && [[ "${TARGET}" -gt 0 ]]; then
        emit_wake "complete" "All ${TARGET} epochs done!"
        log_status "target reached epochs=${EPOCHS}"
        break
    fi

    sleep 60
done

log_status "daemon exiting"
