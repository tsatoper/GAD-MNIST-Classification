#!/bin/bash
#PBS -A UCSC0009
#PBS -N o_mnist
#PBS -q main           
#PBS -l select=1:ncpus=4:ngpus=1:mem=2GB
#PBS -l walltime=11:59:00
#PBS -j oe
# qsub -J 0-19 pipeline.sh
# qsub -v PBS_ARRAY_INDEX=5 pipeline.sh


module load conda  
conda activate py2d_env

START_TIME=$(date +%s)
echo "Job started at $(date)"

python -u /glade/derecho/scratch/tsatoperry/GAD/MNIST/pipeline.py \
    --array-idx $PBS_ARRAY_INDEX \
    --job-num $PBS_JOBID \
    --output-dir /glade/derecho/scratch/tsatoperry/GAD/MNIST/models/testing \
    --learning-rate 0.0001 \
    --n-samples 4000 \
    --gamma 1.0 \
    --epochs 2000 \
    --noise 0.0 \
    --weight-decay 0.00001 


# N1 = 1000
# N2 = 6000 
# N3 = 10.000
# N4 = 60.000

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
