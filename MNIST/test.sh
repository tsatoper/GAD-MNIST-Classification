#!/bin/bash
#PBS -A UCSC0009
#PBS -N o_mnist
#PBS -q main           
#PBS -l select=1:ncpus=1:mem=5GB
#PBS -l walltime=00:59:00
#PBS -j oe

module load conda  
conda activate py2d_env

START_TIME=$(date +%s)
echo "Job started at $(date)"

python -u test.py \
    --weights /glade/derecho/scratch/tsatoperry/GAD/MNIST/models/ddtrueN2/weights \
    --output-dir /glade/derecho/scratch/tsatoperry/GAD/MNIST/models/ddtrueN2/singular_values2 \
    --epoch 2000 \
    --batch-size 2048


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
