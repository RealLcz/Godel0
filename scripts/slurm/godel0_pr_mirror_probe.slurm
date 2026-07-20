#!/bin/bash
#SBATCH --job-name=godel0-prmirror
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=logs/godel0_pr_mirror_%j.out
#SBATCH --error=logs/godel0_pr_mirror_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-local}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
GODEL0_ROOT="${REPO_ROOT}"

VLLM_PID=""
cleanup() {
    echo "[cleanup] Killing vLLM (pid=${VLLM_PID})..."
    [ -n "${VLLM_PID}" ] && kill "${VLLM_PID}" 2>/dev/null && sleep 2 && kill -9 "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[$(date)] Activating HGM conda environment..."
eval "$(/mnt/vast/home/ym56kacy/miniconda3/bin/conda shell.bash hook)"
conda activate HGM
export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
"${PYTHON_BIN}" -V

export JINHE_ROOT="/mnt/vast/home/ym56kacy/jinhe"
export JINHE_CACHE_ROOT="${JINHE_ROOT}/.cache"
export HF_HOME="${JINHE_CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export VLLM_NO_USAGE_STATS=1
export TMPDIR="${JINHE_CACHE_ROOT}/swe_tmp/${JOB_ID}"
mkdir -p "${TMPDIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c '.')
TP_SIZE=1
while [ $((TP_SIZE * 2)) -le "${NUM_GPUS}" ]; do
    TP_SIZE=$((TP_SIZE * 2))
done
export TENSOR_PARALLEL_SIZE="${TP_SIZE}"
echo "[$(date)] GPUs: ${NUM_GPUS}, Tensor parallel: ${TENSOR_PARALLEL_SIZE}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -d "${CUDA_HOME}" ]; then
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

export VLLM_MODEL_NAME="Qwen/Qwen3.6-35B-A3B"
export SERVED_MODEL_NAME="Qwen/Qwen3.6-35B-A3B"
export VLLM_PORT="${VLLM_PORT:-8000}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"

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
    > "logs/vllm_pr_mirror_${JOB_ID}.stdout.log" 2> "logs/vllm_pr_mirror_${JOB_ID}.stderr.log" &
VLLM_PID=$!

echo "[$(date)] Waiting for vLLM to be ready..."
VLLM_WAIT=0
VLLM_MAX_WAIT=600
until "${PYTHON_BIN}" -c "import urllib.request; urllib.request.urlopen('http://${VLLM_HOST}:${VLLM_PORT}/v1/models', timeout=2)" >/dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[ERROR] vLLM process died. Check logs/vllm_pr_mirror_${JOB_ID}.stderr.log"
        tail -30 "logs/vllm_pr_mirror_${JOB_ID}.stderr.log" 2>/dev/null || true
        exit 1
    fi
    VLLM_WAIT=$((VLLM_WAIT + 5))
    if [ "${VLLM_WAIT}" -ge "${VLLM_MAX_WAIT}" ]; then
        echo "[ERROR] vLLM not ready within ${VLLM_MAX_WAIT}s"
        tail -30 "logs/vllm_pr_mirror_${JOB_ID}.stderr.log" 2>/dev/null || true
        exit 1
    fi
    sleep 5
done
echo "[$(date)] vLLM ready (~${VLLM_WAIT}s)."

export VLLM_HOST VLLM_PORT
export GODEL0_MODEL="Qwen/Qwen3.6-35B-A3B"
export PYTHONPATH="${GODEL0_ROOT}/src:${GODEL0_ROOT}/initial_agent/src:${GODEL0_ROOT}"

echo "[$(date)] Running PR Mirror probe on Ansible repo..."
"${PYTHON_BIN}" -u "${GODEL0_ROOT}/scripts/run_pr_mirror_probe.py" \
    --godel0-root "${GODEL0_ROOT}" \
    --repo-pool "${GODEL0_ROOT}/repo_pool" \
    --repo-id ansible \
    --model "${GODEL0_MODEL}" \
    --vllm-host "${VLLM_HOST}" \
    --vllm-port "${VLLM_PORT}" \
    --max-targets 3 \
    --test-python "${PYTHON_BIN}" \
    --output-dir "${GODEL0_ROOT}/output_pr_mirror_probe_${JOB_ID}" \
    2>&1 | tee "logs/godel0_pr_mirror_${JOB_ID}.log"

echo "[$(date)] PR Mirror probe complete."
echo "[$(date)] Outputs: ${GODEL0_ROOT}/output_pr_mirror_probe_${JOB_ID}"
