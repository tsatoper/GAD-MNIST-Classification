#!/bin/bash
#PBS -A UCSC0009
#PBS -N o_mnist_N1
#PBS -q main           
#PBS -l select=1:ncpus=4:ngpus=1:mem=8GB
#PBS -l walltime=11:59:00
#PBS -j oe
# qsub -J 0-14 pipeline.sh
# qsub -v PBS_ARRAY_INDEX=0 pipeline.sh


module load conda  
conda activate py2d_env

START_TIME=$(date +%s)
echo "Job started at $(date)"

python -u /glade/derecho/scratch/tsatoperry/GAD/MNIST/pipeline.py \
    --array-idx $PBS_ARRAY_INDEX \
    --job-num $PBS_JOBID \
    --output-dir /glade/derecho/scratch/tsatoperry/GAD/MNIST/models/N1_1e-4 \
    --learning-rate 0.0001 \
    --n-samples 1000 \
    --gamma 1.0 \
    --epochs 10000


# N1 = 1000
# N2 = 10.000
# N2 = 60.000

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
