#!/bin/bash
#SBATCH --job-name=godel0-swe-best
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/godel0_swe_best_%j.out
#SBATCH --error=logs/godel0_swe_best_%j.err

set -euo pipefail
GODEL0_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_DIR="${GODEL0_RUN_DIR:?Set GODEL0_RUN_DIR to the completed evolution run directory}"
AGENT_REPO="${GODEL0_AGENT_REPO:?Set GODEL0_AGENT_REPO to the evolution agent Git repository}"
MENDELGM_ROOT="${MENDELGM_ROOT:-${GODEL0_ROOT}/../MendelGM}"
EXPORT_DIR="${GODEL0_SWEBENCH_AGENT_DIR:-${RUN_DIR}/swebench_best_agent}"
RESULT_DIR="${GODEL0_SWEBENCH_OUTPUT_DIR:-${RUN_DIR}/swebench_evaluation}"

mkdir -p "${GODEL0_ROOT}/logs"
eval "$(/mnt/vast/home/ym56kacy/miniconda3/bin/conda shell.bash hook)"
conda activate HGM
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
"${PYTHON_BIN}" "${GODEL0_ROOT}/scripts/export_best_node.py" \
    --run-dir "${RUN_DIR}" \
    --agent-repo "${AGENT_REPO}" \
    --output-dir "${EXPORT_DIR}"

cd "${MENDELGM_ROOT}"
env -u SLURM_SUBMIT_DIR \
    INITIAL_AGENT_PATH="${EXPORT_DIR}" \
    HGM_OUTPUT_DIR="${RESULT_DIR}" \
    EVAL_MODEL_NAME="godel0_best_Qwen3.6-35B-A3B" \
    bash swe_scripts/eval_initial_agent.slurm
