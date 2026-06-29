#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --job-name=surf_pca_gpu
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_surf_pca_gpu_%j.out
#SBATCH --error=slurm_surf_pca_gpu_%j.err

set -euo pipefail

WORKDIR=/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/surrogate-models
ENV_PATH=/proj/berzelius-2026-62/users/x_ribec/software/surf-surrogate-env

echo "============================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURMD_NODENAME}"
echo "Started:   $(date)"
echo "Workdir:   ${WORKDIR}"
echo "============================================"

# ---- GPU check ----
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ---- conda environment ----
module load Mambaforge/23.3.1-1-hpc1-bdist
source /software/sse/manual/Mambaforge/23.3.1-1/hpc1-bdist/etc/profile.d/conda.sh

if [ ! -d "${ENV_PATH}" ]; then
    echo "Creating conda environment at ${ENV_PATH} ..."
    mamba create -y -p "${ENV_PATH}" -c conda-forge \
        python=3.10 numpy pandas scipy scikit-learn \
        py-xgboost-gpu optuna rdkit
    echo "Environment created."
else
    echo "Using existing environment: ${ENV_PATH}"
fi

conda activate "${ENV_PATH}"

python -c "import xgboost; print('xgboost version:', xgboost.__version__)"
python -c "from rdkit import Chem; print('rdkit: ok')"
python -c "import optuna; print('optuna:', optuna.__version__)"

# ---- run ----
cd "${WORKDIR}"
echo ""
echo "Running surrogate_pca_models_gpu.py ..."
echo ""

python -u surrogate_pca_models_gpu.py

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
