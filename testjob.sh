#!/bin/bash
#PBS -A UCSC0009
#PBS -N output
#PBS -q main           
#PBS -l select=1:ncpus=1:mem=4GB
#PBS -l walltime=00:05:00
#PBS -j oe
# -J 0-30


module load conda  
conda activate py2d_env

echo "Job started at $(date)"
python -u /glade/derecho/scratch/tsatoperry/GAD/test.py \
    --job-idx $PBS_ARRAY_INDEX \
    
echo "Job ended at $(date)"

