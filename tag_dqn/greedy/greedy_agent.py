'''
Greedy search agent for the DQN environment
'''

import torch
import os
import yaml
import numpy as np
import pandas as pd

from copy import deepcopy
from ..dqn import dqn_data_proc, dqn_agent, dqn_env
from contextlib import redirect_stdout

def two_step_greedy_term_analysis(q_net, env):
    '''
    A1 and A2 are separate, just like for RL agent, requires non-zero reward for A1
    '''
    env.reset()
    env._print_decisions = False

    done = False
    total_reward = 0
    rewards = []

    q_net.eval()  # no training, also removes noise contributions in noisy nets
    env.epsilon = 0  # no exploring
    with torch.no_grad():
        while not done: # For each step in the episode
            print(f'Step {env.timestep.long().item() + 1}/{env.ep_length}')
            S_0 = deepcopy(env.state)
            a_type = S_0['action_type'] + 1
            max_reward = -1e5
            max_reward_state = None  # after all possible two steps (action 1 + 2) from S_0
            # Action
            Q_0, A_0 = q_net(S_0)
            # Loop over each Q_value to scan over action space
            for i in range(len(Q_0)):
                Qs = Q_0.clone()  
                Qs = Qs * 0  # Set all Q values to zero
                Qs[i] = 1  # Set the current Q value to 1 (the action to take)
                # Select action and get next state
                S_1, action_index, reward, done = deepcopy(env.step(Qs))  # action_index = i, reward = 0
                if reward > max_reward:
                    max_reward_state = deepcopy(S_1)
                    max_reward = reward.item()
                # Return to prev state
                env.state = deepcopy(S_0)
                env.state_to_attr()
            rewards.append(max_reward)
            total_reward += max_reward
            print(f'|A{a_type}| = {len(Q_0)}')
            print(f'Reward: {max_reward:.4f}')
            # Update env
            env.state = deepcopy(max_reward_state) 
            env.state_to_attr()
            if max_reward == 0: # if no-op
                env.timestep += 1
                env.state['time_step'] = env.timestep

    print(f'Total reward: {total_reward:.4f}')

def run_greedy(config_file, reward_params=None, output_dir='greedy_results'):
    '''
    Initialise the RL agent and environment
    '''
    # Create results directory at where run is called if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_dir+'/log.txt', 'w') as f:
        tee = dqn_data_proc.Tee(os.sys.stdout, f)
        with redirect_stdout(tee):  

            # Initialise data ----------------------------------------------------------------------------------
            # All 1D numpy arrays, wn and E in units of 1000 cm-1
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            params = config['params']

            preproc_in = dqn_data_proc.get_preproc_input(config_file, float_levs=params['float_levs'])

            # Get graph and linelist
            init_graph, linelist, E_scale = dqn_data_proc.preproc(preproc_in, plot=False)  # kwarg
            print('Graph data shapes:', init_graph)
            print('Line list shape:', linelist.shape)

            diff_scale = params['wn_range'] / E_scale  # Scale for the difference between observed and calculated energies

            # Initialise Q networks ----------------------------------------------------------------------------

            # Hidden sizes
            gat_n_layers = 1  # number of GAT layers
            hidden_size = 1  # attention hidden size
            heads = 1  # attention heads, GAT output size is hidden_size * heads
            mlp_hidden_size = 1  # MLP hidden size, MLP input size is hidden_size * heads

            # Noisy Networks for exploration?
            noisy = False
            sigma_init = 1  # this constant value times (1 / mlp_hidden_size) ** 0.5 fills the sigma matrix

            z = torch.tensor([1]) # one atom at Q=1 for regular Q value learning
            
            # Duelling?
            duel = params['duel']

            # Action elimination network in agent? (not found to help and likely does not work anymore)
            aen = False
            
            # Online Q net
            q_net = dqn_agent.Agent(hidden_size, heads, gat_n_layers, mlp_hidden_size, 
                                    noisy, sigma_init, z, diff_scale, aen, duel)

            # Initialise environment ---------------------------------------------------------------------------
            epsilon_start = 1  # Exploration rate
            data = (init_graph, linelist, E_scale)
            env = dqn_env.Env(data, preproc_in.lev_name, preproc_in.J, 
                              preproc_in.fixed_lev_indices, preproc_in.fixed_lev_values, 
                              params['ep_length'], epsilon_start, z, params['wn_range'], 
                              params['tol'], params['int_tol'], 
                              A2_max = params['A2_max'],
                              reward_params=reward_params, NN_ham=False)

            # Load all known levels for evaluation, if applicable
            if config['all_known_levels'] is not None:
                all_known_levs = np.loadtxt(config['all_known_levels'])
            else:
                all_known_levs = np.array([0.0])

            if config['all_known_levels_and_labels'] is not None:
                all_known_levels_and_labels = pd.read_csv(config['all_known_levels_and_labels'])
            else:
                all_known_levels_and_labels = None
    
            print('Running greedy term analysis...')
            two_step_greedy_term_analysis(q_net, env)  # run greedy algorithm
            lev_names, levs = env._get_known_levs()
            dqn_data_proc.comp(lev_names, levs, preproc_in.known_lev_values, 
                                      all_known_levs, all_known_levels_and_labels, True)
            
            # # Get classified line list pd dataframe, check for dodgy levels and add rejected level indices to prune list
            # result_graph = env.state['graph']
            # classified_linelist = dqn_data_proc.get_pd_table(result_graph, env.E_scale, env.lev_name, env.J, known_only=True) 
            # classified_linelist.to_csv(output_dir+'/classified_linelist.csv', index=False)

    