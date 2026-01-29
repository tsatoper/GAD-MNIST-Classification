#!/bin/bash
#PBS -A UCSC0009
#PBS -N o_ar_mlp
#PBS -q main           
#PBS -l select=1:ncpus=4:ngpus=1:mem=32GB
#PBS -l walltime=11:59:00
#PBS -j oe
# qsub -J 0-13 main.sh
# qsub -v PBS_ARRAY_INDEX=0 main.sh

module load conda  
conda activate py2d_env

# Array of hidden dimension pairs to test
# Format: "hidden1,hidden2"
HIDDEN_DIMS=(
    "64,64"
    "8192,8192"
    "128,128"
    "256,256"
    "512,512"
    "1024,1024"
    "2048,2048"
    "128,64"
    "256,128"
    "512,256"
    "1024,512"
    "64,128"
    "128,256"
    "256,512"
    "512,1024"
)

# Parse hidden dimensions for this job
IFS=',' read -r HIDDEN1 HIDDEN2 <<< "${HIDDEN_DIMS[$PBS_ARRAY_INDEX]}"

START_TIME=$(date +%s)
echo "Job started at $(date)"
echo "Job index: $PBS_ARRAY_INDEX"
echo "Hidden1: $HIDDEN1, Hidden2: $HIDDEN2"

python -u /glade/derecho/scratch/tsatoperry/GAD/KS_1d/main.py \
    --job-idx $PBS_ARRAY_INDEX \
    --output-dir /glade/derecho/scratch/tsatoperry/GAD/KS_1d/models/default \
    --train-data-path /glade/derecho/scratch/tsatoperry/GAD/KS_1d/training_data/train_KS_1024.npy \
    --val-data-path /glade/derecho/scratch/tsatoperry/GAD/KS_1d/training_data/val_KS_1024.npy \
    --hidden1-dim $HIDDEN1 \
    --hidden2-dim $HIDDEN2

END_TIME=$(date +%s)
echo "Job ended at $(date)"

# Calculate and print duration
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "=================================================="
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Hidden dimensions: $HIDDEN1 x $HIDDEN2"
echo "=================================================="