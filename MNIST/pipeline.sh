#!/bin/bash
#PBS -A UCSC0009
#PBS -N output
#PBS -q main           
#PBS -l select=1:ncpus=1:ngpus=1:mem=5GB
#PBS -l walltime=11:59:00
#PBS -j oe
# qsub -J 0-20 pipeline.sh
# qsub -v PBS_ARRAY_INDEX=0 pipeline.sh


module load conda  
conda activate py2d_env

START_TIME=$(date +%s)
echo "Job started at $(date)"

python -u /glade/derecho/scratch/tsatoperry/GAD/MNIST/pipeline.py \
    --job-idx $PBS_ARRAY_INDEX \
    --output-dir /glade/derecho/scratch/tsatoperry/GAD/MNIST/models/dd \
    --loss-fn mse

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
