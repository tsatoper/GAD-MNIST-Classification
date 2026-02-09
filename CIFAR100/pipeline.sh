#!/bin/bash
#PBS -A UCSC0009
#PBS -N o_cifar100
#PBS -q main           
#PBS -l select=1:ncpus=8:ngpus=1:mem=32GB
#PBS -l walltime=11:59:00
#PBS -j oe
# qsub -J 0-11 pipeline.sh
# qsub -v PBS_ARRAY_INDEX=0 pipeline.sh


module load conda  
conda activate py2d_env

START_TIME=$(date +%s)
echo "Job started at $(date)"

WIDEN_FACTORS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
WIDEN_FACTOR=${WIDEN_FACTORS[$PBS_ARRAY_INDEX]}

echo "=================================================="
echo "Running WideResNet with widen_factor=${WIDEN_FACTOR}"
echo "Array Index: $PBS_ARRAY_INDEX"
echo "Job ID: $PBS_JOBID"
echo "=================================================="

python -u /glade/derecho/scratch/tsatoperry/GAD/CIFAR100/pipeline.py \
    --array-idx $PBS_ARRAY_INDEX \
    --job-num $PBS_JOBID \
    --output-dir /glade/derecho/scratch/tsatoperry/GAD/CIFAR100/models/n_5000 \
    --depth 28 \
    --widen-factor $WIDEN_FACTOR \
    --samples 5000 \
    --use-mixed-precision


# k = 1, 2, 4, 8
# depth: 28, maybe 40 or 16
# use 0.3 dropout for k=> 10

END_TIME=$(date +%s)
echo "Job ended at $(date)"

# Calculate and print duration
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "=================================================="
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=================================================="