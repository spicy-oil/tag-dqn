'''
Run MCTS
'''

import torch
import pandas as pd
import random
import numpy as np
import os
import yaml

from . import mcts_agent
from ..dqn import dqn_env
from ..dqn import dqn_data_proc
from contextlib import redirect_stdout

def run_mcts(config_file, seed, reward_params=None, output_dir='./mcts_results'):
    '''
    Initialise the RL agent, env, and trainer, then train and evaluate
    '''
    with open(output_dir+'/log_'+str(seed)+'.txt', 'w') as f:
        tee = dqn_data_proc.Tee(os.sys.stdout, f)
        with redirect_stdout(tee): 
            # Create results directory at where run is called if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            seed = seed

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Initialise data ----------------------------------------------------------------------------------
            # All 1D numpy arrays, wn and E in units of 1000 cm-1
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            params = config['params']

            (
            wn_obs, wn_obs_unc, I_obs, snr_obs,  # Line list
            wn_calc, gA_calc, upper_lev_id,  lower_lev_id, lev_id, E_calc, J, P, lev_name,  # Calcs
            known_lev_indices, known_lev_values, known_lines,  # Term analysis state
            fixed_lev_indices, fixed_lev_values,  # Usually known_lev_indices, known_lev_values from the above line
            min_snr, spec_range, wn_range, tol, int_tol, A2_max, ep_length  # Env parameters
            ) = dqn_data_proc.env_input(config_file, float_levs=params['float_levs'])

            init_graph, linelist, E_scale = dqn_data_proc.preproc(
                        wn_obs, wn_obs_unc, I_obs, snr_obs,  # Line list
                        wn_calc, gA_calc, upper_lev_id, lower_lev_id,  # Transition probabilities
                        lev_id, E_calc, J, P,  # Energy levels
                        known_lev_indices, known_lev_values, known_lines,  # Known levels and lines
                        fixed_lev_indices, fixed_lev_values,  # Fixed levels
                        min_snr, spec_range,  # Filter parameters
                        wn_range,  # Search range is used to calculate line densities
                        plot=False)  # kwarg
            
            print('Graph data shapes:', init_graph)
            print('Line list shape:', linelist.shape)

            # Initialise environment ---------------------------------------------------------------------------
            epsilon_start = 1  # Exploration rate
            data = (init_graph, linelist, E_scale)
            z = torch.tensor([0])  # 1 support atom no DL
            env = dqn_env.Env(data, lev_name, J, fixed_lev_indices, fixed_lev_values, 
                                ep_length, epsilon_start, z, wn_range, tol, int_tol,
                                A2_max = A2_max,
                                reward_params=reward_params, NN_ham=False)

            #%% MCTS ------------------------------------------------------------------------------------------

            mag = mcts_agent.MCTSAgent(env, params=params)

            #print(f'{len(env.A2_cache)} unique A2s and {env.A2_cache_saves} A2 computations were saved')

            #%% Eval
            # Load all known levels for evaluation, if applicable
            if config['all_known_levels'] is not None:
                all_known_levs = np.loadtxt(config['all_known_levels'])
            else:
                all_known_levs = np.array([0.0])

            if config['all_known_levels_and_labels'] is not None:
                all_known_levels_and_labels = pd.read_csv(config['all_known_levels_and_labels'])
            else:
                all_known_levels_and_labels = None

            print('Starting MCTS')
            mag.search()  # run MCTS

            print('Final state:')
            env.state = mag.root_node.state
            env.state_to_attr()
            lev_names, levs = env._get_known_levs()
            if config['all_known_levels_and_labels'] is None:
                N_correct, N_found = dqn_data_proc.comp(lev_names, levs, known_lev_values, 
                                            all_known_levs, all_known_levels_and_labels, True)
                print(f'{N_correct} correct levs out of {N_found}')
            else:
                N_correct, N_found, N_correct_id = dqn_data_proc.comp(lev_names, levs, known_lev_values, 
                                            all_known_levs, all_known_levels_and_labels, True)
                print(f'{N_correct} correct levs out of {N_found}')

            print('Largest reward trajectory final state:')
            env.state = mag.largest_traj_R_state
            env.state_to_attr()
            lev_names, levs = env._get_known_levs()
            if config['all_known_levels_and_labels'] is None:
                N_correct_ls, N_found_ls = dqn_data_proc.comp(lev_names, levs, known_lev_values, 
                                            all_known_levs, all_known_levels_and_labels, True)
            else:
                N_correct_ls, N_found_ls, N_correct_id_ls = dqn_data_proc.comp(lev_names, levs, known_lev_values, 
                                            all_known_levs, all_known_levels_and_labels, True)
            print(f'{N_correct_ls} correct levs out of {N_found_ls}')
            tmp = ''
            if config['all_known_levels_and_labels'] is not None:
                tmp = f' and {N_correct_id_ls} correct IDs'
            print(f'Total stepping reward {mag.total_stepped_rewards:.4f}, {N_correct} correct levs out of {N_found}')
            print(f'Max trajectory reward {mag.largest_traj_R:.4f}, {N_correct_ls} correct levs out of {N_found_ls}' + tmp)


