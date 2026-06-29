#!/bin/bash
# Submit all 18 Morgan fingerprint ablation jobs in parallel.
#
# 9 raw jobs:  radius ∈ {2,3,4} × nbits ∈ {512,1024,2048}
# 9 PCA jobs:  same grid, with --pca flag
#
# Usage: bash submit_morgan_jobs.sh [--dry-run]

SLURM_SCRIPT="$(dirname "$0")/run_morgan.slurm"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "=== DRY RUN — commands will be printed but not executed ==="
fi

submit() {
    local radius="$1" nbits="$2" pca="$3"
    local pca_label; pca_label=$([ "$pca" = "1" ] && echo "pca" || echo "raw")
    local job_name="morgan_r${radius}_n${nbits}_${pca_label}"

    local cmd=(
        sbatch
        --job-name="${job_name}"
        --export="MORGAN_RADIUS=${radius},MORGAN_NBITS=${nbits},MORGAN_PCA=${pca}"
        "${SLURM_SCRIPT}"
    )

    if [ "$DRY_RUN" = "1" ]; then
        echo "  ${cmd[*]}"
    else
        "${cmd[@]}"
    fi
}

echo "Submitting 18 jobs (9 raw + 9 PCA) …"
echo ""
echo "--- Raw fingerprints ---"
for RADIUS in 2 3 4; do
    for NBITS in 512 1024 2048; do
        submit "$RADIUS" "$NBITS" 0
    done
done

echo ""
echo "--- PCA fingerprints ---"
for RADIUS in 2 3 4; do
    for NBITS in 512 1024 2048; do
        submit "$RADIUS" "$NBITS" 1
    done
done

echo ""
echo "Done. Monitor with: squeue -u \$USER"
