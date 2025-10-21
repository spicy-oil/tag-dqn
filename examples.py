#%%
import tag_dqn
import numpy as np
import torch
import random

config_dir = './data/envs/'
#%% Reward Learning -------------------------------------------------------------------------------

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Obtain expert demonstrations from stochastically reversing MDP
# Make sure MDP initial states have cycles on known subgraph (e.g. won't work for the nd2 example case)
# Both are lists of expert state transitions, can shuffle and join as required
co2_demos = tag_dqn.get_demos(config_dir+'co2/config.yaml')  # Train dataset
nd3_demos = tag_dqn.get_demos(config_dir+'nd3/config.yaml')  # Validation dataset

# Rw_fn parameters from lowest validation loss epoch
reward_fn = tag_dqn.train_rw(32, co2_demos, nd3_demos)  # (N_epochs, train_demos, val_demos)
tag_dqn.eval_rw(reward_fn, nd3_demos)  # This prints results

# If saving:
# torch.save(reward_fn.state_dict(), 'reward.pth')
# Then move to installation_dir/tag_dqn/pkg_data and it will be loaded in dqn_env.py or specify location in the functions below
#%% Reinforcement Learning -----------------------------------------------------------------------

# kwarg reward_params can be None (default if unspecified), which uses the reward function trained above, same as config_dir+'reward.pth'
# kwarg output_dir is optional, otherwise default ./
tag_dqn.run_greedy(config_dir+'co2/config.yaml') 
tag_dqn.run_mcts(config_dir+'co2/config.yaml', seed=42)
tag_dqn.run_tag_dqn(config_dir+'co2/config.yaml', seed=42)

# Grid search and multi-seed run examples in data/grid_search_scripts

#%% After RL: Prune and Swap -----------------------------------------------------------------------

# known_levels_out.csv and known_lines_out.csv are generated after tag_dqn runs in the results directory
# these can be new inputs but pruning and relabelling wrong levels is expected

# Prune example: remove levels with indices 5 and 12
tag_dqn.prune(config_dir+'/co2/known_levels_out.csv',  # File format would be the same as input
              config_dir+'/co2/known_lines_out.csv',   # File format would be the same as input
              prune_list=[5, 12])
# Outputs known_levels_pruned.csv and known_lines_pruned.csv in the current directory

# Relabel example: level index 8 to 15, and 20 to 25
tag_dqn.relabel(config_dir+'/co2/known_levels_out.csv', # File format would be the same as input
                config_dir+'/co2/known_lines_out.csv',  # File format would be the same as input
                label_from=[8, 20], label_to=[15, 25])
# Outputs known_levels_relab.csv and known_lines_relab.csv in the current directory