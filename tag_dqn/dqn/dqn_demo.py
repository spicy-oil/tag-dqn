'''
Obtain expert MDP state transitions (demos) by reversing MDP
'''

import torch
import yaml

from . import dqn_env
from . import dqn_data_proc
from .dqn_reward import mark_prob_known_lines, mark_ll_known_lines, collect_demos
from .lopt import lopt_known_subgraph
# %% Data proc

def get_demos(config_file):
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

    spec_range = [0, 1e5]  # use all lines for demos

    # Get graph and linelist
    init_graph, linelist, E_scale = dqn_data_proc.preproc(
                wn_obs, wn_obs_unc, I_obs, snr_obs,  # Line list
                wn_calc, gA_calc, upper_lev_id, lower_lev_id,  # Transition probabilities
                lev_id, E_calc, J, P,  # Energy levels
                known_lev_indices, known_lev_values, known_lines,  # Known levels and lines
                fixed_lev_indices, fixed_lev_values,  # Fixed levels
                min_snr, spec_range,  # Filter parameters
                wn_range,  # Search range is used to calculate line densities
                remove_known_lines=False,  # keep known lines in the line list because we are reversing
                plot=False)  # kwarg
    
    # Let known lines have linelist_mapping, so that when removed, they appear in LEVHAM
    linelist_mapping = mark_ll_known_lines(init_graph, linelist)

    # Initialise environment mainly for A2 space computations
    epsilon_start = 1  # Exploration rate
    data = (init_graph, linelist, E_scale)
    env = dqn_env.Env(data, lev_name, J, fixed_lev_indices, fixed_lev_values, 
                      888, epsilon_start, torch.tensor([1]), wn_range, tol, int_tol,
                      A2_max=99999)  # no need for A2 capping here
    env.linelist_mapping = linelist_mapping
    
    # Add noise to I_calc because when these levels were found, I_calc was not as accurate as it is now
    I_calc = 10 ** ((init_graph.edge_attr[:, 3]) * 5)
    I_calc = torch.abs(I_calc + torch.randn_like(I_calc) * 0.1 * I_calc)  # add 10% noise
    I_calc = torch.log10(I_calc) / 5  # scale back
    init_graph.edge_attr[:, 3] = I_calc

    # Collect demos by reversing MDP
    demos = collect_demos(init_graph, env, E_scale, wn_range, tol)
    return demos
