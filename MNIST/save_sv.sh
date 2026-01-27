#!/bin/bash
#PBS -A UCSC0009
#PBS -N output
#PBS -q main           
#PBS -l select=1:ncpus=1:mem=100GB
#PBS -l walltime=11:59:00
#PBS -j oe
# qsub -J 0-19 save_sv.sh
# qsub -v PBS_ARRAY_INDEX=0 save_sv.sh



module load conda  
conda activate py2d_env

echo "Job started at $(date)"
python -u /glade/derecho/scratch/tsatoperry/GAD/MNIST/save_sv.py \
    --job-idx $PBS_ARRAY_INDEX \
    --max-samples 10000 \
    --model-dir '/glade/derecho/scratch/tsatoperry/GAD/MNIST/models/omni'

    
echo "Job ended at $(date)"

