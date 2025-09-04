#!/bin/bash
#PBS -l select=1:ncpus=8:mem=128gb
#PBS -l walltime=12:00:00

# We must copy files over to the local disk on the compute node!
cd $TMPDIR
mkdir -p data/envs
cp -r $PBS_O_WORKDIR/data/envs/* ./data/envs/
cp $PBS_O_WORKDIR/mcts_job.py .
cp $PBS_O_WORKDIR/$FOLDER_NAME/config.yaml .
mkdir -p $FOLDER_NAME

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate tag-dqn

echo "Running with seed $SEED"

# Track log
/usr/bin/time -v python -u mcts_job.py \
    --config "config.yaml" \
    --seed "$SEED" \
    --output_dir "$FOLDER_NAME" \
    > "$PBS_O_WORKDIR/$FOLDER_NAME/log_$SEED.txt" 2>&1


cp -rn ./$FOLDER_NAME/* $PBS_O_WORKDIR/$FOLDER_NAME/

# Average with results from all seeds and hyparams
cd $PBS_O_WORKDIR/$FOLDER_NAME/
cd ..
cp $PBS_O_WORKDIR/mcts_summary.py .
python mcts_summary.py
rm mcts_summary.py