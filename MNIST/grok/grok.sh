#!/bin/bash
#PBS -A UCSC0009
#PBS -N o_mnist
#PBS -q main           
#PBS -l select=1:ncpus=4:ngpus=1:mem=2GB
#PBS -l walltime=11:59:00
#PBS -j oe

module load conda  
conda activate py2d_env

START_TIME=$(date +%s)
echo "Job started at $(date)"

python -u /glade/derecho/scratch/tsatoperry/GAD/MNIST/grok/grok.py

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
