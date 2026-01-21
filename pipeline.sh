#!/bin/bash
#PBS -A UCSC0009
#PBS -N output
#PBS -q main           
#PBS -l select=1:ncpus=1:ngpus=1:mem=5GB
#PBS -l walltime=11:59:00
#PBS -j oe
# qsub -J 0-40 pipeline.sh
# qsub -v PBS_ARRAY_INDEX=0 pipeline.sh



module load conda  
conda activate py2d_env

echo "Job started at $(date)"
python -u /glade/derecho/scratch/tsatoperry/GAD/pipeline.py \
    --job-idx $PBS_ARRAY_INDEX \
    --output-dir /glade/derecho/scratch/tsatoperry/GAD/models/omni3 \
    --loss-fn mse
    
echo "Job ended at $(date)"

