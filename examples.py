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
# Then move to installation_dir/tag_dqn/dqn and it will be loaded in dqn_env.py
#%% Reinforcement Learning -----------------------------------------------------------------------

# reward_params can be None, which uses the reward function trained above, same as config_dir+'reward.pth'
# Last optional argument could be output directory, otherwise default ./
tag_dqn.run_greedy(config_dir+'nd2_k/config.yaml', reward_params=config_dir+'reward.pth') 
tag_dqn.run_mcts(config_dir+'nd2_k/config.yaml', seed=42, reward_params=config_dir+'reward.pth')
tag_dqn.run_tag_dqn(config_dir+'nd2_k/config.yaml', seed=42, reward_params=config_dir+'reward.pth')

# Grid search and multi-seed run examples in data/grid_search_scripts