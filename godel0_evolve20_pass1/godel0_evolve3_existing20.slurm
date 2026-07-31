#!/bin/bash
#SBATCH --job-name=godel0-e3
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
# Prefer a whole node when possible, but do not require --exclusive:
# exclusive + 8 GPUs waits forever when every H100 node is already MIXED
# with small jobs (217908 sat in Reason=Resources for hours). GPU preflight
# below still aborts if leftover VRAM is present (the cn34 failure mode).
#SBATCH --exclude=cn34
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/vast/workspaces/Emergent_Topology/Godel0/godel0_evolve20_pass1/logs/godel0_evolve3_%j.out
#SBATCH --error=/mnt/vast/workspaces/Emergent_Topology/Godel0/godel0_evolve20_pass1/logs/godel0_evolve3_%j.err

set -euo pipefail
GODEL0_ROOT="/mnt/vast/workspaces/Emergent_Topology/Godel0"
cd "${GODEL0_ROOT}"
WS="/mnt/vast/workspaces/Emergent_Topology/Godel0/godel0_evolve20_pass1"
mkdir -p "${WS}/logs" "${WS}/configs" "${WS}/swe_tmp"

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

# Prefer the Godel0 workspace disk (15T free after migration). Fall back to
# the shared jinhe cache only if the local hub is missing the model.
export GODEL0_CACHE_ROOT="${GODEL0_CACHE_ROOT:-${GODEL0_ROOT}/.cache}"
export JINHE_CACHE_ROOT="${JINHE_CACHE_ROOT:-/mnt/vast/workspaces/cot_pixel_prediction/jinhe/.cache}"
export HF_HOME="${GODEL0_CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_CACHE="${HF_HOME}/hub"
export VLLM_NO_USAGE_STATS=1
# Jobs 217441/217881/217901/217906 all died in <15s with:
#   mkdir .../jinhe/.cache/swe_tmp/<job>: Disk quota exceeded
# Keep TMPDIR short: vLLM ZMQ IPC paths are capped at ~107 chars
# (sizeof sockaddr_un.sun_path). A long workspace path under
# godel0_evolve20_pass1/swe_tmp/<job>/<uuid> exceeds that and aborts
# before the engine starts (jobs 221620..222508).
export TMPDIR="/tmp/g0e3_${JOB_ID}"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${HF_HUB_CACHE}"
# Keep a workspace-side scratch for non-IPC job artifacts.
mkdir -p "${WS}/swe_tmp/${JOB_ID}"

LOCAL_MODEL="${HF_HUB_CACHE}/models--Qwen--Qwen3.6-35B-A3B"
SHARED_HUB="${JINHE_CACHE_ROOT}/huggingface/hub"
if [ ! -d "${LOCAL_MODEL}/snapshots" ] && [ -d "${SHARED_HUB}/models--Qwen--Qwen3.6-35B-A3B/snapshots" ]; then
  echo "[$(date)] Local model cache missing; falling back to ${SHARED_HUB}"
  export HF_HUB_CACHE="${SHARED_HUB}"
  export TRANSFORMERS_CACHE="${SHARED_HUB}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c '.')
TP_SIZE=1
while [ $((TP_SIZE * 2)) -le "${NUM_GPUS}" ]; do
    TP_SIZE=$((TP_SIZE * 2))
done

export VLLM_MODEL_NAME="Qwen/Qwen3.6-35B-A3B"
# Job 216187 died in 51s on cn34 because port 8000 was already taken by
# another process on a shared node. Prefer an explicit VLLM_PORT; otherwise
# bind an ephemeral free port so co-located jobs cannot collide.
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
# Open the full 256k context window for long self-edit trajectories.
# KV cache roughly doubles vs 128k, so drop util and cap concurrent seqs to
# avoid startup / runtime OOM on 8-GPU nodes.
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-262144}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
export VLLM_HOST="${HOST_IP:-127.0.0.1}"

echo "[$(date)] GPU preflight on $(hostname)"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true
# Fail fast if any GPU already holds >1 GiB (leftover process / bad node).
USED_MIB="$("${PYTHON_BIN}" - <<'PY'
import subprocess
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
vals = [int(x.strip()) for x in out.splitlines() if x.strip()]
print(max(vals) if vals else -1)
PY
)"
if [ "${USED_MIB}" -gt 1024 ]; then
    echo "ERROR: GPU memory already in use (max ${USED_MIB} MiB). Refusing to start vLLM." >&2
    exit 1
fi

echo "[$(date)] Starting ${VLLM_MODEL_NAME} on ${VLLM_HOST}:${VLLM_PORT} with TP=${TP_SIZE} max_model_len=${VLLM_MAX_MODEL_LEN} gpu_mem_util=${GPU_MEMORY_UTILIZATION} max_num_seqs=${VLLM_MAX_NUM_SEQS}"
"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --model "${VLLM_MODEL_NAME}" \
    --served-model-name "${VLLM_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --enforce-eager \
    --gdn-prefill-backend triton \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --dtype auto \
    --trust-remote-code \
    > "${WS}/logs/vllm_evolve3_${JOB_ID}.stdout.log" \
    2> "${WS}/logs/vllm_evolve3_${JOB_ID}.stderr.log" &
VLLM_PID=$!

VLLM_WAIT=0
until "${PYTHON_BIN}" -c "import urllib.request; urllib.request.urlopen('http://${VLLM_HOST}:${VLLM_PORT}/v1/models', timeout=2)" >/dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        tail -80 "${WS}/logs/vllm_evolve3_${JOB_ID}.stderr.log" 2>/dev/null || true
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

CONFIG_PATH="${GODEL0_ROOT}/configs/evolve3_ansible_existing20.yaml"
cp -f "${CONFIG_PATH}" "${WS}/configs/evolve3_ansible_existing20.yaml"
# Resume the crashed 216679 run (5/10 bootstrap tasks already in task_store)
# instead of minting a fresh run_id that would orphan that progress.
RESUME_RUN_DIR="${GODEL0_ROOT}/runs_ansible_evolve3_existing20/ansible_evolve3_existing20_216679"
echo "[$(date)] 3-epoch resume from ${RESUME_RUN_DIR}; agent_repo HEAD=$(git -C "${GODEL0_ROOT}/agent_repo_ansible_evolve20_pass1" rev-parse --short HEAD)"
"${PYTHON_BIN}" -u -m godel0.cli resume \
    --run-dir "${RESUME_RUN_DIR}" \
    2>&1 | tee "${WS}/logs/godel0_evolve3_${JOB_ID}.log"

echo "[$(date)] Evolve-3 verification complete"
