#!/bin/bash
#SBATCH --job-name=godel0-proposer
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=logs/godel0_proposer_%j.out
#SBATCH --error=logs/godel0_proposer_%j.err
#
# Gödel0 Proposer experiment with Qwen3.6-35B-A3B on Ansible repo.
#
# Submit:
#   sbatch scripts/slurm/godel0_proposer.slurm
#
# This script:
#   1. Activates the HGM conda environment (has vLLM 0.19.1)
#   2. Starts a vLLM OpenAI-compatible server with Qwen3.6-35B-A3B
#   3. Waits for vLLM to be ready
#   4. Runs the Gödel0 proposer on the Ansible repo
#   5. Validates generated candidates (F2P check)
#   6. Prints a quality report

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-local}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
GODEL0_ROOT="${REPO_ROOT}"

# --- Cleanup trap ---
VLLM_PID=""
cleanup() {
    echo "[cleanup] Killing vLLM (pid=${VLLM_PID})..."
    [ -n "${VLLM_PID}" ] && kill "${VLLM_PID}" 2>/dev/null && sleep 2 && kill -9 "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- Conda environment ---
echo "[$(date)] Activating HGM conda environment..."
eval "$(/mnt/vast/home/ym56kacy/miniconda3/bin/conda shell.bash hook)"
conda activate HGM
export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
"${PYTHON_BIN}" -V

# --- Cache environment ---
export JINHE_ROOT="/mnt/vast/home/ym56kacy/jinhe"
export JINHE_CACHE_ROOT="${JINHE_ROOT}/.cache"
export HF_HOME="${JINHE_CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export VLLM_NO_USAGE_STATS=1
export TMPDIR="${JINHE_CACHE_ROOT}/swe_tmp/${JOB_ID}"
mkdir -p "${TMPDIR}"

# --- GPU / Tensor Parallel setup ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c '.')
TP_SIZE=1
while [ $((TP_SIZE * 2)) -le "${NUM_GPUS}" ]; do
    TP_SIZE=$((TP_SIZE * 2))
done
export TENSOR_PARALLEL_SIZE="${TP_SIZE}"
echo "[$(date)] GPUs: ${NUM_GPUS}, Tensor parallel: ${TENSOR_PARALLEL_SIZE}"

# Set CUDA_HOME so FlashInfer/nvcc can find cuda_runtime.h
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -d "${CUDA_HOME}" ]; then
    # Try conda's CUDA
    CUDA_HOME_CANDIDATES=(/mnt/vast/home/ym56kacy/miniconda3/envs/HGM)
    for c in "${CUDA_HOME_CANDIDATES[@]}"; do
        if [ -d "${c}/include" ] && [ -f "${c}/include/cuda_runtime.h" ]; then
            export CUDA_HOME="${c}"
            break
        fi
    done
fi
if [ -n "${CUDA_HOME}" ] && [ -d "${CUDA_HOME}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export CPATH="${CUDA_HOME}/include:${CPATH:-}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
echo "[$(date)] CUDA_HOME=${CUDA_HOME}"

# --- vLLM server ---
export VLLM_MODEL_NAME="Qwen/Qwen3.6-35B-A3B"
export SERVED_MODEL_NAME="Qwen/Qwen3.6-35B-A3B"
export VLLM_PORT="${VLLM_PORT:-8000}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
if [ -z "${HOST_IP}" ]; then
    HOST_IP="$(hostname -i 2>/dev/null | awk '{print $1}' || true)"
fi
export VLLM_HOST="${HOST_IP:-127.0.0.1}"

echo "[$(date)] Starting vLLM server: ${VLLM_MODEL_NAME} on ${VLLM_HOST}:${VLLM_PORT}"

"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --model "${VLLM_MODEL_NAME}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --enforce-eager \
    --gdn-prefill-backend triton \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --dtype auto \
    --trust-remote-code \
    > "logs/vllm_${JOB_ID}.stdout.log" 2> "logs/vllm_${JOB_ID}.stderr.log" &
VLLM_PID=$!

# --- Wait for vLLM ---
echo "[$(date)] Waiting for vLLM to be ready..."
VLLM_WAIT=0
VLLM_MAX_WAIT=600
until "${PYTHON_BIN}" -c "import urllib.request; urllib.request.urlopen('http://${VLLM_HOST}:${VLLM_PORT}/v1/models', timeout=2)" >/dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[ERROR] vLLM process died. Check logs/vllm_${JOB_ID}.stderr.log"
        tail -30 "logs/vllm_${JOB_ID}.stderr.log" 2>/dev/null || true
        exit 1
    fi
    VLLM_WAIT=$((VLLM_WAIT + 5))
    if [ "${VLLM_WAIT}" -ge "${VLLM_MAX_WAIT}" ]; then
        echo "[ERROR] vLLM not ready within ${VLLM_MAX_WAIT}s"
        tail -30 "logs/vllm_${JOB_ID}.stderr.log" 2>/dev/null || true
        exit 1
    fi
    sleep 5
done
echo "[$(date)] vLLM ready (~${VLLM_WAIT}s). Served model:"
curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" | "${PYTHON_BIN}" -m json.tool 2>/dev/null || true

# --- Quick LLM sanity check ---
echo "[$(date)] Quick LLM sanity check..."
curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.6-35B-A3B","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":10,"temperature":0}' \
    | "${PYTHON_BIN}" -m json.tool 2>/dev/null || true
echo ""

# --- Run Gödel0 Proposer ---
echo "[$(date)] Running Gödel0 Proposer on Ansible repo..."
export VLLM_HOST VLLM_PORT
export GODEL0_MODEL="Qwen/Qwen3.6-35B-A3B"
export PYTHONPATH="${GODEL0_ROOT}/src:${GODEL0_ROOT}/initial_agent/src:${GODEL0_ROOT}"

"${PYTHON_BIN}" -u "${GODEL0_ROOT}/scripts/run_proposer_experiment.py" \
    --godel0-root "${GODEL0_ROOT}" \
    --repo-pool "${GODEL0_ROOT}/repo_pool" \
    --repo-id ansible \
    --model "${GODEL0_MODEL}" \
    --vllm-host "${VLLM_HOST}" \
    --vllm-port "${VLLM_PORT}" \
    --batch-size 3 \
    --max-candidates 15 \
    --output-dir "${GODEL0_ROOT}/output_proposer_${JOB_ID}" \
    2>&1 | tee "logs/godel0_proposer_${JOB_ID}.log"

echo "[$(date)] Experiment complete."
echo "[$(date)] Outputs: ${GODEL0_ROOT}/output_proposer_${JOB_ID}"
