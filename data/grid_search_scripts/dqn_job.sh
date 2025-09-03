#!/bin/bash
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=24:00:00

# Job starts in temp dir, make the output directory
cd $TMPDIR
mkdir -p $FOLDER_NAME

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate tag-dqn

echo "Running with seed $SEED"

# Track log
/usr/bin/time -v python -u tag_dqn_job.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --output_dir "$FOLDER_NAME" \
    > "$PBS_O_WORKDIR/$FOLDER_NAME/log_$SEED.txt" 2>&1


cp -r ./dqn_results/* $PBS_O_WORKDIR/$FOLDER_NAME/

# Average with results from other seeds
cp $PBS_O_WORKDIR/dqn_summary.py $PBS_O_WORKDIR/$FOLDER_NAME/dqn_summary.py
cd $PBS_O_WORKDIR/$FOLDER_NAME
python dqn_summary.py
rm dqn_summary.py