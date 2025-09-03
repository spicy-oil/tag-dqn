#!/bin/bash
#PBS -l select=1:ncpus=8:mem=128gb
#PBS -l walltime=12:00:00

cd $TMPDIR
mkdir -p $FOLDER_NAME

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate tag-dqn

echo "Running with seed $SEED"

/usr/bin/time -v python -u mcts.py > $PBS_O_WORKDIR/$FOLDER_NAME/log_$SEED.txt 2>&1
