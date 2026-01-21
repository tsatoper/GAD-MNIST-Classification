#!/bin/bash
#PBS -A UCSC0009
#PBS -N output
#PBS -q main           
#PBS -l select=1:ncpus=1:mem=5GB
#PBS -l walltime=00:09:00
#PBS -j oe
# qsub -J 0-40 save_sv.sh
# qsub -v PBS_ARRAY_INDEX=0 save_sv.sh



module load conda  
conda activate py2d_env

echo "Job started at $(date)"
python -u /glade/derecho/scratch/tsatoperry/GAD/save_sv.py \
    --job-idx $PBS_ARRAY_INDEX \
    
echo "Job ended at $(date)"

