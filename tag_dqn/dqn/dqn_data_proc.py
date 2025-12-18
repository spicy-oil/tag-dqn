#%%
'''
Utility functions for DQN data processing
'''

import pandas as pd
import numpy as np
import torch
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re

import yaml

from torch_geometric.data import Data
from scipy.interpolate import interp1d
from .lopt import lopt_known_subgraph

def moving_average(arr, window_size=64):
    arr = np.pad(arr, (window_size // 2, window_size // 2), mode='mean', stat_length=window_size)
    arr = np.convolve(arr, np.ones(window_size) / window_size, mode='same')
    return arr[window_size // 2: - window_size // 2]

def preproc_temp(E_u, known_lines, known_line_gAs):
    '''
    ['L1', 'L2', 'wn', 'wn_unc', 'I_obs'] for dataframe known_lines
    '''
    x = E_u
    y = np.log(known_lines.I_obs.values / known_line_gAs) 
    m, c = huber_fit(x, y)
    T = - (6.63e-34 * 3e8 * 1e5) / (m * 1.38e-23) 
    return T, m, c

def wn_unc_to_graph(wn_unc):
    '''wn_unc from 1e3 cm-1 unit to graph & linelist units'''
    # FTS wn uncs
    # highest is 1 cm-1 for problem lines
    # highest non-problem lines are ~ 0.05 cm-1
    # lowest is a few 0.0001 cm-1
    wn_unc = wn_unc * 1e7  # from kK to 0.1 mK
    wn_unc = np.log10(wn_unc)  # [0, 4]
    # Most uncs >0.001 cm-1 (>1) and < 0.1 cm-1 (3)
    wn_unc = wn_unc / 4  # [0, 1]
    wn_unc = - wn_unc  # make negative so that higher means better
    return wn_unc

def wn_unc_from_graph(wn_unc):
    '''wn_unc from graph & linelist units to 1e3 cm-1'''
    wn_unc = -wn_unc
    wn_unc = wn_unc * 4
    wn_unc = 10 ** wn_unc
    wn_unc = wn_unc * 1e-7
    return wn_unc

def line_den_to_graph(rho):
    '''line_den from lines per 1e3 cm-1 to graph units'''
    rho = rho  # avoid log(0), at least one line within 1000 cm-1 with S/N between 0 and 1 order of magnitude above
    # highest line den is a few per cm-1, so a few thousand per 1000 cm-1
    rho = np.log10(rho) / 3  # [0, ~1]
    rho = -rho  # negative so that higher means better for reward
    return rho

def line_den_from_graph(rho):
    '''line_den from graph units to lines per 1e3 cm-1'''
    rho = -rho * 3
    rho = 10 ** rho
    rho = rho
    return rho

def I_to_graph(I):
    '''I from relative intensity to graph units'''
    I = np.log10(I) / 5 
    return I

def I_from_graph(I):
    '''I from graph units to relative intensity'''
    I = 10 ** (I * 5)
    return I

def preproc(wn_obs, wn_obs_unc, I_obs, snr_obs,  # Line list (all 1D np arrays)
            wn_calc, gA_calc, upper_lev_id, lower_lev_id,  # Transition probabilities (all 1D np arrays)
            lev_id, E_calc, J, P, # Energy levels (all 1D np arrays)
            known_lev_indices, known_lev_values, known_lines,  # Known levels and lines (1D np arrays but known_lines is pd.DataFrame)
            fixed_lev_indices, fixed_lev_values,  # Fixed levels (1D np arrays)
            min_snr=2, spec_range=[12, 55],  # Filter parameters
            wn_range=0.250,  # Search range is used to calculate line densities
            remove_known_lines=True,  # If remove known lines from line list
            plot=False):
    '''
    Require wn and E units in kK (1000 cm-1)
    all IDs start from 0 and have no gaps
    I_obs as relative photon flux, same scale as known_lines.I_obs
    min_snr is used to remove all predicted lines below the min expected snr
    spec_range is used to remove all predicted lines outside the range from the graph
    '''
    # Fit snr_obs / I_obs ratio ------------------------------------------------
    lowess = sm.nonparametric.lowess(snr_obs / I_obs, wn_obs, frac=.02)
    observability = interp1d(lowess[:, 0], lowess[:, 1], kind='linear', bounds_error=False, fill_value=(lowess[0, 1], lowess[-1, 1]))

    if plot:
        plt.figure()
        plt.title('Relative snr_obs / I_obs ratio')
        plt.scatter(wn_obs, snr_obs / I_obs)
        plt.plot(wn_obs, observability(wn_obs), 'r')
        plt.ylabel('snr_obs / I_obs')
        plt.xlabel('wn (kK)')
        plt.tight_layout()

    # Calculate expected SNR ------------------------------------------------
    ul_idx = known_lines.L2.values  # integer np array of upper level indices
    ll_idx = known_lines.L1.values  # integer np array of lower level indices
    E_u = []
    known_line_gAs = []
    for i in range(len(ul_idx)):
        idx = known_lev_indices == ul_idx[i]
        E_u.append(known_lev_values[idx])
        idx = (upper_lev_id == ul_idx[i]) & (lower_lev_id == ll_idx[i])
        if idx.any() == False: # if observed line not in calc (maybe gA too small)
            gA = np.array([1e5])
        else:
            gA = gA_calc[idx]
        known_line_gAs.append(gA)
    known_line_gAs = np.array(known_line_gAs).flatten()
    E_u = np.array(E_u).flatten()
    T, m, c = preproc_temp(E_u, known_lines, known_line_gAs)
    if plot:
        plt.figure()
        plt.title(f'Boltzman plot for known lines T={T:.0f} K')
        plt.scatter(E_u, np.log(known_lines.I_obs.values / known_line_gAs))
        x = np.linspace(E_u.min(), E_u.max(), 100)
        plt.plot(x, c + m * x, 'r')
        plt.ylabel(r'log_e(I_obs/gA_calc)')
        plt.xlabel('E2 (kK)')
        plt.tight_layout()
    I_calc = gA_calc * np.exp(c - (6.63e-34 * 3e8 * E_calc[upper_lev_id] * 1e5) / (1.38e-23 * T))  # photon flux
    snr_calc = observability(wn_calc) * I_calc

    # Filter predicted lines ------------------------------------------------
    observable_idx = snr_calc > min_snr  # Remove predicted lines below expected SNR of min_snr 

    # Remove predicted lines outside spec_range
    in_wn_range = (wn_calc > spec_range[0]) & (wn_calc < spec_range[1])
    observable_idx = observable_idx & in_wn_range
    
    snr_calc = np.clip(snr_calc, 1, None)  # minimum expected snr of 1 when min_snr < 1, because of logging later
    if plot:
        in_wn_obs_range = (wn_obs > spec_range[0]) & (wn_obs < spec_range[1])
        N = len(wn_calc[observable_idx])
        plt.figure()
        plt.title(f'Expected snr for the {N} calculated lines expected to be observable (red)')
        plt.vlines(wn_obs[in_wn_obs_range], 0, snr_obs[in_wn_obs_range], 'k', label='snr obs')
        plt.vlines(wn_calc[observable_idx], 0, snr_calc[observable_idx], 'r', label = 'snr calc')
        plt.legend(loc='upper right')
        plt.ylabel('S/N')
        plt.xlabel('wn (kK)')
        plt.tight_layout()

        plt.figure()
        plt.title(f'Expected intensity for the {N} calculated lines expected to be observable (red)')
        plt.vlines(wn_obs[in_wn_obs_range], 0, I_obs[in_wn_obs_range], color='k', label='I_obs')
        plt.vlines(wn_calc[observable_idx], 0 , I_calc[observable_idx], color='r', label='I_calc')
        plt.legend(loc='upper right')
        plt.ylabel('Relative Intensity')
        plt.xlabel('wn (kK)')
        plt.tight_layout()
        plt.show()

    # Log10 the intensities, snrs, and gA_calc ------------------------------
    I_obs = I_to_graph(I_obs)
    snr_obs = I_to_graph(snr_obs)
    I_calc = I_to_graph(I_calc)
    snr_calc = I_to_graph(snr_calc) 
    gA_calc = np.log10(gA_calc) / 10
    known_lines_I_obs = I_to_graph(known_lines.I_obs.values)
    known_snr_obs = I_to_graph(known_lines.snr.values)

    # Get expected line density for calculated lines ------------------------
    line_den = [] # N lines per 1 kK
    for i in range(len(wn_calc)):
        # Filter wn_obs within search range
        in_wn_range = (wn_obs > wn_calc[i] - wn_range) & (wn_obs < wn_calc[i] + wn_range)
        # Filter snr_obs within 1 mag above snr_calc
        # Because if both ways lines near noise will have lower density vs mid-snr lines!
        snr_calc_i = np.clip(snr_calc[i], 1 / 5, None)  # let snr_calc be 10 for snr_calc below 10
        in_snr_range = (snr_obs > snr_calc[i]) & (snr_obs < snr_calc_i + 1 / 5) & (snr_obs > snr_calc_i - 1 / 5) # within +-1 order of magnitude
        rho = len(wn_obs[in_wn_range & in_snr_range]) / (2 * wn_range) + 1  # +1 from the line itself
        line_den.append(rho)
    line_den = np.array(line_den)
    line_den = line_den_to_graph(line_den)

    # To torch tensors ------------------------------------------------------
    in_wn_range = (wn_obs > spec_range[0]) & (wn_obs < spec_range[1])
    wn_obs = torch.tensor(wn_obs[in_wn_range], dtype=torch.float64)
    wn_obs_unc = torch.tensor(wn_obs_unc[in_wn_range], dtype=torch.float64)
    I_obs = torch.tensor(I_obs[in_wn_range], dtype=torch.float64)
    snr_obs = torch.tensor(snr_obs[in_wn_range], dtype=torch.float64)

    wn_calc = torch.tensor(wn_calc, dtype=torch.float64)
    I_calc = torch.tensor(I_calc, dtype=torch.float64)
    gA_calc = torch.tensor(gA_calc, dtype=torch.float64)
    snr_calc = torch.tensor(snr_calc, dtype=torch.float64)

    # Scale wn & E ----------------------------------------------------------
    E_scale = np.float64(E_calc.max())
    wn_calc = wn_calc / E_scale
    wn_obs = wn_obs / E_scale
    wn_obs_unc = wn_unc_to_graph(wn_obs_unc)
    E_calc = E_calc / E_scale

    # Construct graph edges ------------------------------------------------
    # [source_nodes, target_nodes]
    edge_index = torch.tensor(np.array([upper_lev_id, lower_lev_id]), dtype=torch.long)
    edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1) # For message passing both ways (not just from upper levels to lower levels)

    # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
    # One hot encoding for observed and unobserved
    edge_attr_emission = torch.tensor(np.array([wn_calc, torch.zeros_like(wn_calc), torch.zeros_like(wn_calc), 
                                                I_calc, torch.zeros_like(wn_calc), gA_calc,
                                                snr_calc, torch.zeros_like(wn_calc), line_den,
                                                torch.zeros_like(wn_calc), torch.ones_like(wn_calc)]), 
                                                dtype=torch.float64).T
    
    edge_attr_absorption = torch.tensor(np.array([wn_calc, torch.zeros_like(wn_calc), torch.zeros_like(wn_calc),
                                                  I_calc, torch.zeros_like(wn_calc), gA_calc,
                                                  snr_calc, torch.zeros_like(wn_calc), line_den,
                                                  torch.zeros_like(wn_calc), torch.ones_like(wn_calc)]), 
                                                  dtype=torch.float64).T
    # Undirected for message passing both ways
    edge_attr_undirected = torch.cat([edge_attr_emission, edge_attr_absorption], dim=0)

    # Construct graph nodes ------------------------------------------------
    # One hot encoding for known and unknown levels
    x_known = np.zeros_like(E_calc)
    x_unknown = np.ones_like(E_calc)
    # One hot encoding for selected and unselected levels
    x_sel = np.zeros_like(E_calc)
    x_unsel = np.ones_like(E_calc)
    # # One hot encoding for even and odd levels
    # even = P == 0
    # odd = P == 1
    # [E_calc, E_obs, known, unknown, selected, unselected]
    x = torch.tensor(np.array([E_calc, np.zeros_like(E_calc), x_known, x_unknown, x_sel, x_unsel]).T, dtype=torch.float64)

    # Generate graph and line list instances -------------------------
    init_graph = Data(x=x, edge_index=edge_index_undirected, edge_attr=edge_attr_undirected)
    init_linelist = torch.tensor(np.array([wn_obs, wn_obs_unc, I_obs, snr_obs]).T, dtype=torch.float64)

    # Remove known lines from init_linelist
    if remove_known_lines:
        for w in known_lines.wn.values:
            idx = torch.abs(init_linelist[:, 0] * E_scale - w).argmin()
            init_linelist = torch.cat((init_linelist[:idx], init_linelist[idx+1:]), 0)

    # Add known and fixed levels to graph
    init_graph, known_line_index = assign_knowns(init_graph, E_scale, 
                                known_lev_indices, known_lev_values, 
                                known_lines.L2.values, known_lines.L1.values, 
                                known_lines.wn.values, known_lines.wn_unc.values, known_lines_I_obs, known_snr_obs,
                                fixed_lev_indices, fixed_lev_values)
    
    # Filter unobservable predicted lines
    observable_idx = np.concatenate((observable_idx, observable_idx)) | ((init_graph.edge_attr[:, -1]).numpy() == 0)  # keep known lines
    init_graph.edge_attr = init_graph.edge_attr[observable_idx]
    init_graph.edge_index = init_graph.edge_index[:, observable_idx]

    # Check these values!
    E, wn_obs_unc, d_wn, known_level_indices = lopt_known_subgraph(init_graph, E_scale, 
                                                fixed_lev_indices, fixed_lev_values / E_scale, test=True) # d_wn is obs-ritz
    print('Any initial graph (wn_obs - wn_ritz) > 1.2 * wn_unc? ', (1.2 * wn_obs_unc < d_wn).any())  # Hopefully false because if optimisation is off at the start none of this would work

    return init_graph, init_linelist, E_scale

#%%
def assign_knowns(graph, E_scale,
                  known_lev_indices, known_lev_values, 
                  known_source_index, known_target_index, 
                  known_wn_obs, known_wn_obs_unc, known_I_obs, known_snr_obs,
                  fixed_lev_indices, fixed_lev_values):
    '''
    Add known quantities to the graph of purely calcs, all E and wn in kK (1000 cm-1)
    '''
    x = graph.x
    edge_attr = graph.edge_attr
    edge_index = graph.edge_index

    # Let known levels be known
    # [E_calc, E_obs, known, unknown, selected, unselected]
    for j, i in enumerate(known_lev_indices):
        x[i][[1, 2, 3]] = torch.tensor([known_lev_values[j] / E_scale, 1, 0], dtype=torch.float64)
        
    # Let known lines be known
    known_line_index = []  # for first half of graph.edge_attr
    # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
    for i, source in enumerate(known_source_index):
        target = known_target_index[i]
        temp = torch.where((edge_index[0] == source) & (edge_index[1] == target))[0]
        if temp.size(0) == 0:
            print('----')
            print(f'Warning - known line from {known_lev_values[known_lev_indices == source.item()][0]:.7f} to {known_lev_values[known_lev_indices == target.item()][0]:.7f} not found in graph, likely that its expected S/N is too low and was filtered out')
            print(f'This line was not added to the graph')
            continue
        edge_idx = temp[0]
        known_line_index.append(edge_idx)
        edge_attr[edge_idx][[1, 2, 4, 7, 9, 10]] = torch.tensor([known_wn_obs[i] / E_scale, 
                                                            wn_unc_to_graph(known_wn_obs_unc[i]),
                                                            known_I_obs[i],
                                                            known_snr_obs[i],
                                                            1, 0], dtype=torch.float64)
        # Other way because undirected
        edge_idx = torch.where((edge_index[0] == target) & (edge_index[1] == source))[0][0]
        edge_attr[edge_idx][[1, 2, 4, 7, 9, 10]] = torch.tensor([known_wn_obs[i] / E_scale, 
                                                            wn_unc_to_graph(known_wn_obs_unc[i]),
                                                            known_I_obs[i], 
                                                            known_snr_obs[i],
                                                            1, 0], dtype=torch.float64)

    # Let fixed levels be known
    for j, i in enumerate(fixed_lev_indices):
        x[i][[1, 2, 3]] = torch.tensor([fixed_lev_values[j] / E_scale, 1, 0], dtype=torch.float64)
    
    return graph, known_line_index

def remove_known_lines(linelist, E_scale, known_wn_obs):
    for w in known_wn_obs:
        line_idx = torch.abs(linelist[:, 0] - w / E_scale).argmin()
        linelist = torch.cat((linelist[:line_idx], linelist[line_idx+1:]), dim=0) 
    return linelist

def get_knowns(graph, E_scale, prune=[]):
    '''
    Prune are indices of levels and their associated lines to remove (back to being unknown)
    '''
    known_lev_indices = torch.where(graph.x[:, 2] > 0)[0]
    known_lev_values = (graph.x[known_lev_indices, 1] * E_scale)
    known_lev_indices = known_lev_indices
    keep_index = ~torch.isin(known_lev_indices, torch.tensor(prune))
    known_lev_indices = known_lev_indices[keep_index].numpy()
    known_lev_values = known_lev_values[keep_index].numpy()

    E_calc = graph.x[:, 0].numpy()
    E_obs = graph.x[:, 1].numpy()
    E = E_obs * (E_obs != 0) + E_calc * (E_obs == 0)  # E_obs is zero for known levels, so use E_calc instead
    E = E * E_scale
    E[0] = 0

    edge_index = graph.edge_index#[:, graph.edge_attr[:, -1] == 0]
    edge_attr = graph.edge_attr#[graph.edge_attr[:, -1] == 0]

    edge_index = edge_index[:, :edge_index.shape[1] // 2]
    edge_attr = edge_attr[:edge_attr.shape[0] // 2]

    keep_index = ~torch.isin(edge_index, torch.tensor(prune))
    keep_index = keep_index[0] & keep_index[1]
    edge_index = edge_index[:, keep_index]
    edge_attr = edge_attr[keep_index]

    all_lines = []
    known_source_index = []
    known_target_index = []
    lev_start_index = []
    lev_id = []
    counter = 0
    for lid in known_lev_indices:
        ulev = torch.isin(edge_index[0], lid)
        llev = torch.isin(edge_index[1], lid)
        lines = ulev | llev
        indices = edge_index[:, lines]
        lid2 = indices[0]
        lid1 = indices[1]
        lines = edge_attr[lines]
        sort_index = torch.argsort(lines[:, 3], descending=True)  # sort by I_calc
        lines = lines[sort_index]
        lid1 = lid1[sort_index]
        lid2 = lid2[sort_index]
        all_lines.append(lines)
        known_source_index.append(lid2)
        known_target_index.append(lid1)
        counter += len(lines)
        lev_start_index.append(counter)
        lev_id.append(lid)

    all_lines = torch.cat(all_lines, dim=0)
    known_source_index = torch.cat(known_source_index, dim=0)
    known_target_index = torch.cat(known_target_index, dim=0)

    known_source_index = known_source_index.numpy()
    known_target_index = known_target_index.numpy()

    # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
    known_wn_obs = (all_lines[:, 1] * E_scale).numpy()  # kK
    known_wn_obs_unc = (wn_unc_from_graph(all_lines[:, 2])).numpy()  # kK
    known_I_calc = 10 ** (all_lines[:, 3].numpy() * 5) 
    known_I_obs = 10 ** (all_lines[:, 4].numpy() * 5)
    known_snr_obs = 10 ** (all_lines[:, 7].numpy() * 5)
    known_gA = 10 ** (all_lines[:, 5].numpy() * 10)  # gA_calc

    return (known_lev_indices, 
            known_lev_values, 
            known_source_index, 
            known_target_index, 
            known_wn_obs, 
            known_wn_obs_unc, 
            known_I_calc,
            known_I_obs,
            known_snr_obs,
            known_gA,
            lev_start_index[:-1],
            lev_id,
            E)

def get_pd_table(graph, E_scale, lev_name, J, prune=[], known_only=False):
    '''
    Create human readable table of classified line list
    '''
    (known_lev_indices, 
    known_lev_values, 
    known_source_index, 
    known_target_index, 
    known_wn_obs, 
    known_wn_obs_unc, 
    known_I_calc,
    known_I_obs,
    known_snr_obs,
    known_gA,
    lev_start_index,
    lev_id,
    E) = get_knowns(graph, E_scale, prune)
    # mapping = np.zeros(known_lev_indices.max() + 1)
    # mapping[known_lev_indices] = np.arange(len(known_lev_indices))
    pd.set_option('display.float_format', '{:.4f}'.format)
    df = pd.DataFrame()
    df['lid1'] = known_target_index
    df['lid2'] = known_source_index
    df['E1'] = E[known_target_index] * 1e3 # cm-1
    df['E2'] = E[known_source_index] * 1e3 # cm-1
    df['J1'] = J[known_target_index]
    df['J2'] = J[known_source_index]
    df['snr'] = known_snr_obs.astype(int)
    df['wn_ritz'] = df.E2 - df.E1
    df['wn_obs'] = known_wn_obs * 1e3 # cm-1
    df['wn_obs_unc'] = known_wn_obs_unc * 1e3 # cm-1
    df['obs-ritz'] = df['wn_obs'] - df['wn_ritz']
    df['I_calc'] = known_I_calc
    df['I_obs'] = known_I_obs
    df['gA'] = known_gA
    df['L1'] = lev_name[known_target_index]
    df['L2'] = lev_name[known_source_index]

    float_cols = ['E1', 'E2', 'wn_ritz', 'wn_obs', 'wn_obs_unc', 'obs-ritz']
    for col in float_cols:
        df[col] = df[col].map('{:.4f}'.format)

    exp_cols = ['I_calc', 'I_obs', 'gA']
    for col in exp_cols:
        df[col] = df[col].map('{:.1e}'.format)

    # Convert all columns to string
    df = df.astype(str)

    mask = df['snr'] == '1'
    df.loc[mask, ['snr', 'I_obs', 'wn_obs', 'wn_obs_unc', 'obs-ritz']] = '-'

    insert_indices = [i for i in lev_start_index]
    blank_row = pd.DataFrame({col: [''] for col in df.columns})
    for i in reversed(insert_indices):  # reverse so index positions don't shift
        df = pd.concat([df.iloc[:i], blank_row, df.iloc[i:]], ignore_index=True)
        df = pd.concat([df.iloc[:i], blank_row, df.iloc[i:]], ignore_index=True)
    if known_only:
        mask = df['snr'] != '-'
        df = df[mask]
    return df

def huber_fit(x, y, t=1.05):
    # hg = HuberRegressor(epsilon=1.05, fit_intercept=True) # this is more immune to outliers than MSE as loss function
    # hg.fit(x.reshape(-1, 1), y)  # needs .reshape(-1, 1) on x
    # m = hg.coef_[0] 
    # c = hg.intercept_
    # print(m, c)
    x = x.reshape(-1, 1)
    x_with_const = sm.add_constant(x)  # adds intercept term
    model = sm.RLM(y, x_with_const, M=sm.robust.norms.HuberT(t=t))
    results = model.fit()
    c = results.params[0]  # intercept
    m = results.params[1]  # slope
    #print(m, c)
    return m, c

def fit_pop(graph, E_scale, plot=False):
    '''
    Fit straight line using huber loss for log10(I_obs/gA_calc) against upper E
    '''
    known_line_index = graph.edge_attr[:, -1] == 0
    # Since edges are undirected, use first half and the source node is always upper level
    known_line_index = known_line_index[:known_line_index.shape[0] // 2]
    known_line_index = torch.where(known_line_index)[0]
    known_line_edge_attr = graph.edge_attr[known_line_index]
    known_line_upper_level_index = graph.edge_index[0][known_line_index]

    # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, line_den, known, unknown] 
    known_line_I_obs = known_line_edge_attr[:, 4].numpy() * 5  # log10(I_obs)
    known_line_gA_calc = known_line_edge_attr[:, 5].numpy() * 10  # log10(gA_calc)

    # [E_calc, E_obs, known, unknown]
    known_line_upper_level = graph.x[:, 1][known_line_upper_level_index].numpy()  # 1000 cm-1 / E_scale

    x = known_line_upper_level
    y = known_line_I_obs - known_line_gA_calc 
    m, c = huber_fit(x, y)
    if plot:
        plt.figure()
        plt.scatter(x * E_scale, y, marker='+', color='k')
        plt.title(f'Boltzman plot for known lines T={- (6.63e-34 * 3e8 * 1e5 * 0.4343) / ( (m / E_scale) * 1.38e-23):.0f} K')
        plt.plot(x * E_scale, m * x + c, 'r')
        plt.ylabel('log10(I_obs/gA_calc)')
        plt.xlabel('E_u (1000 cm-1)')
        plt.tight_layout()
        #plt.show()
    return m, c, x, y
# %%

def get_log_gf(gA, wn):
    gf = gA / (0.66702 * wn ** 2)
    return np.log10(gf)

def get_gA(loggf, wn):
    gf = 10 ** loggf
    gA = gf * (0.66702 * wn ** 2)
    return gA

def comp(final_known_lev_names, final_known_levs, init_known_levs, all_known_levs, all_known_levs_and_labels, prnt=True):
    tol = 5e-5  # cm-1 ~LEVHAM TOL
    init_known_count = init_known_levs.size
    final_known_count = len(final_known_levs)
    N_found = final_known_count - init_known_count
    init_known_levs = init_known_levs

    if all_known_levs_and_labels is not None:
        known_levs, known_lev_ids = all_known_levs_and_labels['known_levs'].values, all_known_levs_and_labels['known_lev_ids'].values
        if len(all_known_levs) == 1:  # If no known levels provided, just ground np.array([0.0])
            all_known_levs = known_levs  # Use the known levels from ll_known_levels_and_labels.csv

    count = 0
    E_and_id_count = 0
    for i, lev in enumerate(final_known_levs):
        lev = lev
        flag = False
        if all_known_levs_and_labels is not None:
            match = re.search(r'id (\d+)', final_known_lev_names[i])
            lev_id = int(match.group(1))
            comp_value = known_levs[known_lev_ids == lev_id]
            if comp_value.size == 0:
                #print(f'Level id {lev_id} not found in all_known_levels_and_labels.csv, skipping comparison of E and id for this level')
                comp_value = np.array([np.nan])
            diff = abs(comp_value - lev)
            if diff < tol:
                E_and_id_count += 1
                flag = True
                if prnt:
                    print(f'{lev*1e3:.4f} cm-1 agree with human {comp_value[0]*1e3:.4f} cm-1 and human label !!! ({final_known_lev_names[i]})' )
        diff = abs(lev - all_known_levs) 
        if (diff < tol).any(): 
            count += 1
            known_diff = abs(lev - init_known_levs)
            if (known_diff < tol).any():
                initial_val = init_known_levs[known_diff.argmin()]
                if prnt:
                    print(f'{lev*1e3:.4f} cm-1 from initial state at {initial_val*1e3:.4f} cm-1 ({final_known_lev_names[i]})')
            else:
                if prnt and not flag:
                    print(f'{lev*1e3:.4f} cm-1 agree with human {all_known_levs[diff.argmin()]*1e3:.4f} cm-1 !!! ({final_known_lev_names[i]})' )
        else:
            if prnt:
                print(f'{lev*1e3:.4f} cm-1 not found by humans... ({final_known_lev_names[i]})')
    if prnt:
        if all_known_levs_and_labels is not None:
            print(f'{E_and_id_count-init_known_count} out of {N_found} new known levels agree in terms of E AND label')
        print(f'{count-init_known_count} out of {N_found} new known levels agree in terms of E')

    if all_known_levs_and_labels is not None:
        return count-init_known_count, N_found, E_and_id_count-init_known_count
    else:
        return count-init_known_count, N_found

def env_input(env_config, float_levs=False):
    '''
    Get environment input data from config file.
    all_float: if True, only ground level will be fixed at 0 cm-1 for all level optimisations.
    '''
    with open(env_config, 'r') as f:
        config = yaml.safe_load(f)

    # Get DataFrames
    line_list = pd.read_csv(config['line_list'])
    theo_levels = pd.read_csv(config['theo_levels'])
    theo_lines = pd.read_csv(config['theo_lines'])
    known_levels = pd.read_csv(config['known_levels'])
    known_lines = pd.read_csv(config['known_lines'])

    # Convert to numpy arrays
    wn_obs, wn_obs_unc, I_obs, snr_obs = (line_list['wn_obs'].values, line_list['wn_obs_unc'].values, 
                                          line_list['I_obs'].values, line_list['snr_obs'].values)
    lev_id, E_calc, J, P, lev_name = (theo_levels['lev_id'].values, theo_levels['E_calc'].values, 
                                      theo_levels['J'].values, theo_levels['P'].values, theo_levels['lev_name'].values)
    wn_calc, gA_calc, upper_lev_id, lower_lev_id = (theo_lines['wn_calc'].values, theo_lines['gA_calc'].values, 
                                                    theo_lines['upper_lev_id'].values, theo_lines['lower_lev_id'].values)
    known_lev_indices, known_lev_values = known_levels['known_lev_indices'].values, known_levels['known_lev_values'].values
    
    # Fix known levels
    if float_levs:
        fixed_lev_indices = np.array([0])
        fixed_lev_values = np.array([0.0])
    else:
        fixed_lev_indices = known_lev_indices
        fixed_lev_values = known_lev_values

    # Get environment parameters
    env_params = config['params']
    min_snr = env_params['min_snr']
    spec_range = env_params['spec_range']
    wn_range = env_params['wn_range']
    tol = env_params['tol']
    int_tol = env_params['int_tol']
    A2_max = env_params['A2_max']
    ep_length = env_params['ep_length']

    return (wn_obs, wn_obs_unc, I_obs, snr_obs,
            wn_calc, gA_calc, upper_lev_id,  lower_lev_id, lev_id, E_calc, J, P, lev_name,
            known_lev_indices, known_lev_values, known_lines,
            fixed_lev_indices, fixed_lev_values,
            min_snr, spec_range, wn_range, tol, int_tol, A2_max, ep_length)

def graph_to_known_csv(graph, E_scale, output_dir='./', out_suffix=''):
    '''
    Get known levels and lines from graph and output to csv files that can be used as input
    '''
    # [E_calc, E_obs, known, unknown, selected, unselected]
    known_lev_indices = torch.where(graph.x[:, 2] > 0)[0]
    known_lev_values = graph.x[known_lev_indices, 1] * E_scale
    known_lev_indices = known_lev_indices.numpy()
    known_lev_values = known_lev_values.numpy()
    df = pd.DataFrame()
    df['known_lev_indices'] = known_lev_indices
    df['known_lev_values'] = known_lev_values
    df.to_csv(output_dir+'known_levels_out'+out_suffix+'.csv', index=False)
    
    # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
    known_lin_indices = torch.where(graph.edge_attr[:, -1] == 0)[0]
    N = known_lin_indices.shape[0] // 2  # because undirected edges
    known_lin_indices = known_lin_indices[:N]
    L1 = graph.edge_index[1][known_lin_indices]
    L2 = graph.edge_index[0][known_lin_indices]
    wn = graph.edge_attr[known_lin_indices, 1] * E_scale
    wn_unc = wn_unc_from_graph(graph.edge_attr[known_lin_indices, 2])
    I_obs = I_from_graph(graph.edge_attr[known_lin_indices, 4])
    snr = I_from_graph(graph.edge_attr[known_lin_indices, 7])
    df = pd.DataFrame()
    df['L1'] = L1.numpy()
    df['L2'] = L2.numpy()
    df['wn'] = wn.numpy()
    df['wn_unc'] = wn_unc.numpy()
    df['I_obs'] = I_obs.numpy()
    df['snr'] = snr.numpy().astype(int)
    df.to_csv(output_dir+'known_lines_out'+out_suffix+'.csv', index=False)

def prune(known_lev_csv, known_lines_csv, prune_list=[], out_suffix=''):
    '''
    prune_list: list of level indices to remove from known levels and their associated lines
    '''
    known_levels = pd.read_csv(known_lev_csv)
    known_lines = pd.read_csv(known_lines_csv)

    for pid in prune_list:
        # Remove from known levels
        mask = known_levels['known_lev_indices'] != pid
        known_levels = known_levels[mask]

        # Remove associated lines from known lines
        mask = (known_lines['L1'] != pid) & (known_lines['L2'] != pid)
        known_lines = known_lines[mask]

    known_levels.to_csv('known_levels_pruned'+out_suffix+'.csv', index=False)
    known_lines.to_csv('known_lines_pruned'+out_suffix+'.csv', index=False)

def relabel(known_levels_csv, known_lines_csv, label_from=[], label_to=[], out_suffix=''):
    '''
    label_from: list of level indices to be swap from
    label_to: list of corresponding level indices to swap into
    '''
    known_levels = pd.read_csv(known_levels_csv)
    known_lines = pd.read_csv(known_lines_csv)

    mapping = {i: j for i, j in zip(label_from, label_to)}
    known_levels['known_lev_indices'] = known_levels['known_lev_indices'].replace(mapping)
    known_lines['L1'] = known_lines['L1'].replace(mapping)
    known_lines['L2'] = known_lines['L2'].replace(mapping)

    known_levels.to_csv('known_levels_relab'+out_suffix+'.csv', index=False)
    known_lines.to_csv('known_lines_relab'+out_suffix+'.csv', index=False)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()
# %%
