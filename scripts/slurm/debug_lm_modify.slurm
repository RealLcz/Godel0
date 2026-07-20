#!/bin/bash
#SBATCH --job-name=godel0-debug
#SBATCH --partition=all
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=00:30:00
#SBATCH --output=logs/godel0_debug_%j.out
#SBATCH --error=logs/godel0_debug_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-local}"
VLLM_PID=""
cleanup() {
    [ -n "${VLLM_PID}" ] && kill "${VLLM_PID}" 2>/dev/null && sleep 2 && kill -9 "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

eval "$(/mnt/vast/home/ym56kacy/miniconda3/bin/conda shell.bash hook)"
conda activate HGM
export PYTHON_BIN="${CONDA_PREFIX}/bin/python"

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
while [ $((TP_SIZE * 2)) -le "${NUM_GPUS}" ]; do TP_SIZE=$((TP_SIZE * 2)); done
export TENSOR_PARALLEL_SIZE="${TP_SIZE}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/include:${CPATH:-}"

export VLLM_MODEL_NAME="Qwen/Qwen3.6-35B-A3B"
export SERVED_MODEL_NAME="Qwen/Qwen3.6-35B-A3B"
export VLLM_PORT="${VLLM_PORT:-8000}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
export VLLM_HOST="${HOST_IP:-127.0.0.1}"

echo "[$(date)] Starting vLLM..."
"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port "${VLLM_PORT}" \
    --model "${VLLM_MODEL_NAME}" --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --enforce-eager --gdn-prefill-backend triton \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --dtype auto --trust-remote-code \
    > "logs/vllm_debug_${JOB_ID}.stdout.log" 2> "logs/vllm_debug_${JOB_ID}.stderr.log" &
VLLM_PID=$!

echo "[$(date)] Waiting for vLLM..."
VLLM_WAIT=0
until "${PYTHON_BIN}" -c "import urllib.request; urllib.request.urlopen('http://${VLLM_HOST}:${VLLM_PORT}/v1/models', timeout=2)" >/dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM died"
        tail -30 "logs/vllm_debug_${JOB_ID}.stderr.log" 2>/dev/null || true
        exit 1
    fi
    VLLM_WAIT=$((VLLM_WAIT + 5))
    if [ "${VLLM_WAIT}" -ge 600 ]; then echo "ERROR: vLLM timeout"; exit 1; fi
    sleep 5
done
echo "[$(date)] vLLM ready (~${VLLM_WAIT}s)"

export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}/src:${SLURM_SUBMIT_DIR:-$PWD}/initial_agent/src:${SLURM_SUBMIT_DIR:-$PWD}"
export VLLM_HOST VLLM_PORT

echo "[$(date)] Running LM Modify debug test..."
"${PYTHON_BIN}" -u scripts/test_lm_modify_debug.py 2>&1 | tee "logs/lm_modify_debug_${JOB_ID}.log"

echo "[$(date)] Done."
