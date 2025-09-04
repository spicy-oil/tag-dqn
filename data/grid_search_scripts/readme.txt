Example scripts we use for grid search and multi-seed runs on the Imperial College HPC (2025). 

To access conda on IC HPC run

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"

then use conda commands as usual

Example usage:
cp everything here to source directory (which contains data/)
on the HPC bash:
python3 dqn_launch_gs.py
or
python3 mcts_launch_gs.py