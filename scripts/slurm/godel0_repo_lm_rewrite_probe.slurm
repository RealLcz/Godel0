#!/bin/bash
#SBATCH --job-name=godel0-repo-rewrite
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=logs/godel0_repo_rewrite_%j.out
#SBATCH --error=logs/godel0_repo_rewrite_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-local}"
GODEL0_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
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
"${PYTHON_BIN}" -V

export JINHE_ROOT="/mnt/vast/home/ym56kacy/jinhe"
export JINHE_CACHE_ROOT="${JINHE_ROOT}/.cache"
export HF_HOME="${JINHE_CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export VLLM_NO_USAGE_STATS=1
export TMPDIR="${JINHE_CACHE_ROOT}/swe_tmp/${JOB_ID}"
mkdir -p "${TMPDIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c '.')
TP_SIZE=1
while [ $((TP_SIZE * 2)) -le "${NUM_GPUS}" ]; do
    TP_SIZE=$((TP_SIZE * 2))
done

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ -d "${CUDA_HOME}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export CPATH="${CUDA_HOME}/include:${CPATH:-}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

export VLLM_MODEL_NAME="Qwen/Qwen3.6-35B-A3B"
export SERVED_MODEL_NAME="${VLLM_MODEL_NAME}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
export VLLM_HOST="${HOST_IP:-127.0.0.1}"

echo "[$(date)] Starting ${VLLM_MODEL_NAME} on ${VLLM_HOST}:${VLLM_PORT} with TP=${TP_SIZE}"
"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --model "${VLLM_MODEL_NAME}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --enforce-eager \
    --gdn-prefill-backend triton \
    --dtype auto \
    --trust-remote-code \
    > "logs/vllm_repo_rewrite_${JOB_ID}.stdout.log" \
    2> "logs/vllm_repo_rewrite_${JOB_ID}.stderr.log" &
VLLM_PID=$!

VLLM_WAIT=0
until "${PYTHON_BIN}" -c "import urllib.request; urllib.request.urlopen('http://${VLLM_HOST}:${VLLM_PORT}/v1/models', timeout=2)" >/dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        tail -80 "logs/vllm_repo_rewrite_${JOB_ID}.stderr.log" 2>/dev/null || true
        exit 1
    fi
    if [ "${VLLM_WAIT}" -ge 600 ]; then
        tail -80 "logs/vllm_repo_rewrite_${JOB_ID}.stderr.log" 2>/dev/null || true
        exit 1
    fi
    sleep 5
    VLLM_WAIT=$((VLLM_WAIT + 5))
done
echo "[$(date)] vLLM ready after ${VLLM_WAIT}s"

unset QWEN_API_BASE_URL
export VLLM_HOST VLLM_PORT
export PYTHONPATH="${GODEL0_ROOT}/src:${GODEL0_ROOT}/initial_agent/src:${GODEL0_ROOT}"
OUTPUT_DIR="${GODEL0_ROOT}/output_repo_lm_rewrite_${JOB_ID}"

"${PYTHON_BIN}" -u "${GODEL0_ROOT}/scripts/run_repo_lm_rewrite_probe.py" \
    --godel0-root "${GODEL0_ROOT}" \
    --repo-pool "${GODEL0_ROOT}/repo_pool" \
    --repo-id ansible \
    --domain yaml_pipeline \
    --model "${VLLM_MODEL_NAME}" \
    --vllm-host "${VLLM_HOST}" \
    --vllm-port "${VLLM_PORT}" \
    --attempts 1 \
    --rewrite-ratio 0.60 \
    --temperature 0 \
    --max-output-tokens 32768 \
    --request-timeout 1800 \
    --agent-timeout 1200 \
    --test-timeout 300 \
    --run-solver \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "logs/godel0_repo_rewrite_${JOB_ID}.log"

echo "[$(date)] Repository LM Rewrite probe complete: ${OUTPUT_DIR}"
