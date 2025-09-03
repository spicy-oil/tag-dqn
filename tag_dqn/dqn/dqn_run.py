#%%
import torch
torch.set_printoptions(precision=9) # to see more d.p.
#torch.set_default_dtype(torch.float64)
import torch.optim as optim

from . import dqn_agent
from . import dqn_env
from . import dqn_exp_buffer
from . import dqn_trainer
from . import dqn_data_proc
from contextlib import redirect_stdout

import os
import random
import yaml
import numpy as np
import pandas as pd

def run_tag_dqn(config_file, seed, reward_params=None, output_dir='./dqn_results'):
    '''
    Initialise the RL agent, env, and trainer, then train and evaluate
    '''
    # Create results directory at where run is called if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+'/learning_curves', exist_ok=True)
    os.makedirs(output_dir+'/exploration_curves', exist_ok=True)
    os.makedirs(output_dir+'/loss_curves', exist_ok=True)
    os.makedirs(output_dir+'/alignment_curves', exist_ok=True)
    os.makedirs(output_dir+'/action_spaces', exist_ok=True)
    os.makedirs(output_dir+'/boltzmann_plots', exist_ok=True)
    os.makedirs(output_dir+'/classified_linelist', exist_ok=True)

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

    with open(output_dir+'/log_'+str(seed)+'.txt', 'w') as f:
        tee = dqn_data_proc.Tee(os.sys.stdout, f)
        with redirect_stdout(tee): 

            # Get graph and linelist
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

            diff_scale = wn_range / E_scale  # Scale for the difference between observed and calculated energies

            # Set rollout parameters ---------------------------------------------------------------------------
            episodes = params['episodes']  # total number of episodes

            # Initialise prioritised exp replay buffer ---------------------------------------------------------
            buffer_capacity = params['buffer_capacity']  # max number of state transitions in the buffer
            batch_size = params['batch_size']  # number of transitions sampled per gradient descent
            beta_start = params['per_alpha']
            alpha = params['per_alpha']  # prioritisation exponent
            n_step = params['n_step']  # n-step Q-learning
            gamma = params['gamma']  # discount factor
            replay_buffer = dqn_exp_buffer.PrioritisedReplayBuffer(buffer_capacity, batch_size, alpha, beta_start, n_step, gamma)

            # Initialise Q networks ----------------------------------------------------------------------------

            # Hidden sizes
            gat_n_layers = params['gat_n_layers']  # number of GAT layers
            hidden_size = params['gat_hidden_size']  # attention hidden size
            heads = params['gat_heads']  # attention heads, GAT output size is hidden_size * heads
            mlp_hidden_size = params['mlp_hidden_size']  # MLP hidden size, MLP input size is hidden_size * heads

            # Noisy Networks for exploration?
            noisy = params['noisy'] 
            sigma_init = params['sigma_0']  # this constant value times (1 / mlp_hidden_size) ** 0.5 fills the sigma matrix

            # Distributional RL? (not found to help and likely does not work anymore)
            dist = False
            if dist:
                # Geometric series sum for gamma^2 (because expected reward at every other step)
                max_reward_per_step = 3  # max reward (no more than 10^5 total SNR of lines found for the level)
                ub = int(np.ceil(max_reward_per_step * (1 - (gamma ** 2) ** (ep_length / 2)) / (1 - gamma ** 2)))
                lb = 0
                z = torch.linspace(lb, ub, 51) # atoms axis
            else:
                z = torch.tensor([1]) # one atom at Q=1 for regular Q value learning
            
            # Duelling?
            duel = params['duel']

            # Action elimination network in agent? (not found to help and likely does not work anymore)
            aen = False
            
            # Online Q net
            q_net = dqn_agent.Agent(hidden_size, heads, gat_n_layers, mlp_hidden_size, 
                                    noisy, sigma_init, z, diff_scale, aen, duel)

            # Target Q net
            q_net_t = dqn_agent.Agent(hidden_size, heads, gat_n_layers, mlp_hidden_size, 
                                    noisy, sigma_init, z, diff_scale, aen, duel) 
            q_net_t.load_state_dict(q_net.state_dict()) # same initial weights as the Q-network
            q_net_t.eval() # do not train target q net

            # Largest reward Q net
            q_net_largest = dqn_agent.Agent(hidden_size, heads, gat_n_layers, mlp_hidden_size, 
                                            noisy, sigma_init, z, diff_scale, aen, duel)
            q_net_largest.load_state_dict(q_net.state_dict()) # same initial weights as the Q-network
            q_net_largest.eval() # do not train largest reward q net

            total_params = sum(p.numel() for p in q_net.parameters())
            print(f"Total parameters of agent: {total_params}")

            # Initialise environment ---------------------------------------------------------------------------
            epsilon_start = 1  # Exploration rate
            data = (init_graph, linelist, E_scale)
            env = dqn_env.Env(data, lev_name, J, fixed_lev_indices, fixed_lev_values, 
                            ep_length, epsilon_start, z, wn_range, tol, int_tol, 
                            A2_max = A2_max,
                            reward_params=reward_params, NN_ham=False)

            # Initialise optimizer ----------------------------------------------------------------------------
            optimizer = optim.Adam(q_net.parameters(), lr=params['adam_lr'])

            # Initialise trainer -------------------------------------------------------------------------------
            tau = params['tau']  # number of episodes to run before updating the target network
            double = params['double']  # Whether to use double DQN
            min_epsilon = params['min_epsilon']
            patience = params['patience']  # number of episodes to wait before stopping training if no improvement in the largest reward
            steps_per_train = params['steps_per_train']  # number of steps to take before training the Q network

            n_grad = np.log(0.01) / np.log(1-tau)
            n_grad_per_ep = ep_length / steps_per_train
            n_ep = n_grad / n_grad_per_ep
            print(f'Total steps (gradient descents): {int(episodes * ep_length)} ({int(episodes * ep_length / steps_per_train)})')
            print(f'Episodes (steps) capacity of the replay buffer: ~{int(buffer_capacity / ep_length)} ({buffer_capacity})')
            print(f'Episodes (steps) to reach 0.01 of the target network: ~{int(n_ep)} ({int(n_grad * steps_per_train)})')

            # Load all known levels for evaluation, if applicable
            if config['all_known_levels'] is not None:
                all_known_levs = np.loadtxt(config['all_known_levels'])
            else:
                all_known_levs = np.array([0.0])

            if config['all_known_levels_and_labels'] is not None:
                all_known_levels_and_labels = pd.read_csv(config['all_known_levels_and_labels'])
            else:
                all_known_levels_and_labels = None

            trainer = dqn_trainer.Trainer(q_net, q_net_t, q_net_largest, env, replay_buffer, optimizer, z, 
                                        batch_size, steps_per_train, tau, double, noisy, min_epsilon, 
                                        patience, known_lev_values, all_known_levs, all_known_levels_and_labels)

            #Train ------------------------------------------------------------------------------------------
            trainer.train_q_net(episodes, tr_start_ep=params['tr_start_ep'])

            # Eval ------------------------------------------------------------------------------------------

            #print(f'{len(env.A2_cache)} unique A2s and {env.A2_cache_saves} A2 computations were saved')
            trainer.plot_training(seed, output_dir=output_dir)

            vars = trainer.term_analysis_check(prnt=True)

            # Record results

            with open(output_dir+'/info_'+str(seed)+'.txt', 'w') as f:
                f.write('Parameters:\n')
                for key, value in params.items():
                    f.write(f'{key}: {value}\n')
                f.write('\n')
                f.write('Results:\n')
                f.write('Largest ep reward during training\n')
                f.write(f'{vars[0]:.2f}\n')
                f.write('Ep reward from deployment of the final agent\n')
                f.write(f'{vars[1]:.2f}\n')
                f.write('N lev match with human / N lev total\n')
                f.write(f'{vars[2]} / {vars[3]}\n')
                f.write('Ep reward from deployment of the agent that achieved largest ep reward during training\n')
                f.write(f'{vars[4]:.2f}\n')
                f.write('N lev match with human / N lev total\n')
                f.write(f'{vars[5]} / {vars[6]}\n')
                f.write('N lev match with human / N lev total for the final state with the largest ep reward during training\n')
                f.write(f'{vars[7]} / {vars[8]}\n')

            with open(output_dir+'/info_'+str(seed)+'.csv', 'w') as f:
                f.write(','.join(map(str, vars)))
            