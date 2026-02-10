#!/bin/bash
#PBS -A UCSC0009
#PBS -N o_cifar100_sv
#PBS -q main
#PBS -l select=1:ncpus=8:ngpus=1:mem=32GB
#PBS -l walltime=11:59:00
#PBS -j oe
# qsub -J 0-14 test.sh
# qsub -v PBS_ARRAY_INDEX=11 test.sh

module load conda
conda activate py2d_env

START_TIME=$(date +%s)
echo "Job started at $(date)"

# Widths 1–15
WIDEN_FACTORS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20)
WIDEN_FACTOR=${WIDEN_FACTORS[$PBS_ARRAY_INDEX]}

echo "=================================================="
echo "Computing SVs for WRN widen_factor=${WIDEN_FACTOR}"
echo "Array Index: $PBS_ARRAY_INDEX"
echo "Job ID: $PBS_JOBID"
echo "=================================================="

python -u /glade/derecho/scratch/tsatoperry/GAD/CIFAR100/test.py \
  --weights /glade/derecho/scratch/tsatoperry/GAD/CIFAR100/models/n_500/depth28/weights \
  --output-dir /glade/derecho/scratch/tsatoperry/GAD/CIFAR100/models/n_500/depth28 \
  --widen-factor $WIDEN_FACTOR \
  --samples 500

END_TIME=$(date +%s)
echo "Job ended at $(date)"

DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "=================================================="
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=================================================="
