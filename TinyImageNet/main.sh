#!/bin/bash
#PBS -A UCSC0009
#PBS -N output
#PBS -q main           
#PBS -l select=1:ncpus=4:ngpus=1:mem=32GB
#PBS -l walltime=11:59:00
#PBS -j oe
# qsub -J 0-10 main.sh
# qsub -v PBS_ARRAY_INDEX=0 main.sh

module load conda  
conda activate py2d_env

# Array of widths to test
WIDTHS=(1 2 4 8 16 32 64 128 256 512 1024 ) #2048 4096 8192)

START_TIME=$(date +%s)
echo "Job started at $(date)"

python -u /glade/derecho/scratch/tsatoperry/GAD/TinyImageNet/main.py \
    --job-idx $PBS_ARRAY_INDEX \
    --output-dir /glade/derecho/scratch/tsatoperry/GAD/TinyImageNet/models/lr1e-4 \
    --data-dir /glade/derecho/scratch/tsatoperry/GAD/TinyImageNet/.tinyimagenet/tiny-imagenet-200 \
    --train-suffix N3 \
    --width ${WIDTHS[$PBS_ARRAY_INDEX]} \
    --learning-rate 1e-4


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