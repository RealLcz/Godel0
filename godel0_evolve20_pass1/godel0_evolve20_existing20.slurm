#!/bin/bash
#SBATCH --job-name=godel0-e20
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --exclude=cn34
#SBATCH --time=7-00:00:00
#SBATCH --output=/mnt/vast/workspaces/Emergent_Topology/Godel0/godel0_evolve20_pass1/logs/godel0_evolve20_%j.out
#SBATCH --error=/mnt/vast/workspaces/Emergent_Topology/Godel0/godel0_evolve20_pass1/logs/godel0_evolve20_%j.err

set -euo pipefail
GODEL0_ROOT="/mnt/vast/workspaces/Emergent_Topology/Godel0"
cd "${GODEL0_ROOT}"
WS="/mnt/vast/workspaces/Emergent_Topology/Godel0/godel0_evolve20_pass1"
mkdir -p "${WS}/logs" "${WS}/configs"

JOB_ID="${SLURM_JOB_ID:-local}"
VLLM_PID=""
cleanup() {
    if [ -n "${VLLM_PID}" ]; then
        kill "${VLLM_PID}" 2>/dev/null || true
        sleep 2
        kill -9 "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

eval "$(/mnt/vast/home/ym56kacy/miniconda3/bin/conda shell.bash hook)"
conda activate HGM
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

export JINHE_CACHE_ROOT="${JINHE_CACHE_ROOT:-/mnt/vast/workspaces/cot_pixel_prediction/jinhe/.cache}"
export HF_HOME="${JINHE_CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export VLLM_NO_USAGE_STATS=1
mkdir -p "${WS}/swe_tmp"
# Prefer home for TMPDIR — workspace jinhe/.cache hit Disk quota (jobs 217441+).
export TMPDIR="${WS}/swe_tmp/${JOB_ID}"
mkdir -p "${TMPDIR}" "${HF_HOME}"

if [ -d /mnt/vast/workspaces/cot_pixel_prediction/jinhe/.cache/huggingface/hub ]; then
  export HF_HUB_CACHE=/mnt/vast/workspaces/cot_pixel_prediction/jinhe/.cache/huggingface/hub
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c '.')
TP_SIZE=1
while [ $((TP_SIZE * 2)) -le "${NUM_GPUS}" ]; do
    TP_SIZE=$((TP_SIZE * 2))
done

export VLLM_MODEL_NAME="Qwen/Qwen3.6-35B-A3B"
# Prefer an explicit VLLM_PORT; otherwise bind an ephemeral free port so
# co-located jobs cannot collide on the hard-coded 8000 (job 216187).
if [ -z "${VLLM_PORT:-}" ]; then
    VLLM_PORT="$("${PYTHON_BIN}" - <<'PY'
import socket
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"
fi
export VLLM_PORT
# Self-edit trajectories peaked at 65.5k tokens and died on the 65536 cap.
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
export VLLM_HOST="${HOST_IP:-127.0.0.1}"

echo "[$(date)] Starting ${VLLM_MODEL_NAME} on ${VLLM_HOST}:${VLLM_PORT} with TP=${TP_SIZE}"
"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --model "${VLLM_MODEL_NAME}" \
    --served-model-name "${VLLM_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --enforce-eager \
    --gdn-prefill-backend triton \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --dtype auto \
    --trust-remote-code \
    > "${WS}/logs/vllm_evolve20_${JOB_ID}.stdout.log" \
    2> "${WS}/logs/vllm_evolve20_${JOB_ID}.stderr.log" &
VLLM_PID=$!

VLLM_WAIT=0
until "${PYTHON_BIN}" -c "import urllib.request; urllib.request.urlopen('http://${VLLM_HOST}:${VLLM_PORT}/v1/models', timeout=2)" >/dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        tail -80 "${WS}/logs/vllm_evolve20_${JOB_ID}.stderr.log" 2>/dev/null || true
        exit 1
    fi
    if [ "${VLLM_WAIT}" -ge 600 ]; then
        exit 1
    fi
    sleep 5
    VLLM_WAIT=$((VLLM_WAIT + 5))
done
echo "[$(date)] vLLM ready after ${VLLM_WAIT}s"

export GODEL0_MODEL="${VLLM_MODEL_NAME}"
export PYTHONPATH="${GODEL0_ROOT}/src:${GODEL0_ROOT}/initial_agent/src:${GODEL0_ROOT}"

CONFIG_PATH="${GODEL0_ROOT}/configs/evolve20_ansible_existing20.yaml"
cp -f "${CONFIG_PATH}" "${WS}/configs/evolve20_ansible_existing20.yaml"
echo "[$(date)] 20-epoch existing-tests formal; agent_repo HEAD=$(git -C "${GODEL0_ROOT}/agent_repo_ansible_evolve20_pass1" rev-parse --short HEAD)"
"${PYTHON_BIN}" -u -m godel0.cli run \
    --config "${CONFIG_PATH}" \
    --run-name "ansible_evolve20_existing20_${JOB_ID}" \
    --max-nodes 20 \
    2>&1 | tee "${WS}/logs/godel0_evolve20_${JOB_ID}.log"

echo "[$(date)] Evolve-20 existing20 complete"
