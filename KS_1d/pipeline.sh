#!/bin/bash
#PBS -A UCSC0009
#PBS -N o_ar_mlp
#PBS -q main           
#PBS -l select=1:ncpus=4:ngpus=1:mem=32GB
#PBS -l walltime=11:59:00
#PBS -j oe
# qsub -J 10-14 pipeline.sh
# qsub -v PBS_ARRAY_INDEX=10 pipeline.sh 

module load conda  
conda activate py2d_env

# Array of hidden dimensions to test
HIDDEN_DIMS=("0" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024" "2048" "4096" "8192")

# Get hidden dimension for this job
HIDDEN_DIM="${HIDDEN_DIMS[$PBS_ARRAY_INDEX]}"

START_TIME=$(date +%s)
echo "Job started at $(date)"
echo "Job index: $PBS_ARRAY_INDEX"
echo "Hidden dim: $HIDDEN_DIM"

#AR_MLP_deep
#AR_MLP_one_layer
python -u /glade/derecho/scratch/tsatoperry/GAD/KS_1d/pipeline.py \
    --job-idx $PBS_ARRAY_INDEX \
    --model AR_MLP_deep \
    --output-dir long \
    --train-data-path /glade/derecho/scratch/tsatoperry/GAD/KS_1d/training_data/train_KS_1024.npy \
    --val-data-path /glade/derecho/scratch/tsatoperry/GAD/KS_1d/training_data/val_KS_1024.npy \
    --hidden-dim $HIDDEN_DIM \
    --epochs 100
END_TIME=$(date +%s)
echo "Job ended at $(date)"

# Calculate and print duration
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "=================================================="
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Hidden dimension: $HIDDEN_DIM"
echo "=================================================="