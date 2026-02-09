#!/bin/bash
#PBS -A UCSC0009
#PBS -N eval_multistep
#PBS -q main           
#PBS -l select=1:ncpus=1:ngpus=1:mem=16GB
#PBS -l walltime=00:30:00
#PBS -j oe

module load conda  
conda activate py2d_env

START_TIME=$(date +%s)

# Configuration
WEIGHTS_DIR="deep/long"
EPOCH=100
VAL_DATA_PATH="/glade/derecho/scratch/tsatoperry/GAD/KS_1d/training_data/val_KS_1024.npy"
NUM_ROLLOUT_STEPS=100

echo "=================================================="
echo "Multi-Step Rollout Evaluation"
echo "=================================================="
echo "Weights directory: ${WEIGHTS_DIR}"
echo "Epoch: ${EPOCH}"
echo "Validation data: ${VAL_DATA_PATH}"
echo "Rollout steps: ${NUM_ROLLOUT_STEPS}"
echo "Job started at $(date)"
echo "=================================================="

python -u /glade/derecho/scratch/tsatoperry/GAD/KS_1d/evaluate_multistep.py \
    --weights-dir $WEIGHTS_DIR \
    --epoch $EPOCH \
    --val-data-path $VAL_DATA_PATH \
    --num-rollout-steps $NUM_ROLLOUT_STEPS

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "=================================================="
echo "Evaluation complete"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=================================================="