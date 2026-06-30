#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --job-name=morgan_job
#SBATCH --partition=berzelius-cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/morgan-models/%x_%j.out
#SBATCH --error=/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/morgan-models/%x_%j.err

# Environment variables expected (set via sbatch --export=...):
#   MORGAN_RADIUS  — fingerprint radius (2, 3, or 4)
#   MORGAN_NBITS   — fingerprint size   (512, 1024, or 2048)
#   MORGAN_PCA     — "1" to enable PCA preprocessing, "0" for raw

set -euo pipefail

WORKDIR=/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/morgan-models
ENV_PATH=/proj/berzelius-2026-62/users/x_ribec/software/surf-surrogate-env

echo "============================================"
echo "Job:      ${SLURM_JOB_ID}  (${SLURM_JOB_NAME})"
echo "Node:     ${SLURMD_NODENAME}"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "Started:  $(date)"
echo "Radius:   ${MORGAN_RADIUS}"
echo "Nbits:    ${MORGAN_NBITS}"
echo "PCA:      ${MORGAN_PCA}"
echo "============================================"

module load Mambaforge/23.3.1-1-hpc1-bdist
source /software/sse/manual/Mambaforge/23.3.1-1/hpc1-bdist/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"

python -c "import xgboost; print('xgboost:', xgboost.__version__)"
python -c "from rdkit import Chem; print('rdkit: ok')"

PCA_FLAG=""
if [ "${MORGAN_PCA}" = "1" ]; then
    PCA_FLAG="--pca"
fi

cd "${WORKDIR}"
python -u surrogate_morgan.py \
    --radius "${MORGAN_RADIUS}" \
    --nbits  "${MORGAN_NBITS}"  \
    ${PCA_FLAG}

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
