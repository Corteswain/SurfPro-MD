#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --job-name=surf_pca_par
#SBATCH --partition=berzelius-cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_surf_pca_parallel_%j.out
#SBATCH --error=slurm_surf_pca_parallel_%j.err

set -euo pipefail

WORKDIR=/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/surrogate-models
ENV_PATH=/proj/berzelius-2026-62/users/x_ribec/software/surf-surrogate-env

echo "============================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURMD_NODENAME}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Started:   $(date)"
echo "Workdir:   ${WORKDIR}"
echo "============================================"

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

python -c "import xgboost; print('xgboost:', xgboost.__version__)"
python -c "from rdkit import Chem; print('rdkit: ok')"
python -c "import optuna; print('optuna:', optuna.__version__)"

# ---- run ----
cd "${WORKDIR}"
echo ""
echo "Running surrogate_pca_models_parallel.py ..."
echo ""

python -u surrogate_pca_models_parallel.py --global-only

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
