'''
Weighted least-squares optimisation of level energies using known line wavenumbers, part of state transitions for the environment
Not as sophisticated as [A. Kramida. The program LOPT for least-squares optimization of energy levels. Computer Physics Communications, 182:419–434, 2011]
'''
#%%
import torch
import numpy as np
np.set_printoptions(suppress=True, precision=6) # stop standard form so we can see d.p.

from . import dqn_data_proc
from torch_geometric.utils import subgraph

def lopt(wn_obs, wn_obs_unc, N_lev, edges, 
         fixed_lev_indices = np.array([0]), 
         fixed_lev_values = np.array([1e-6]), 
         print_results=False):
    '''
    Able to get Ritz WN within ~1 mK of actual LOPT output for 144 Nd III levels (VIS-UV lines)
    Please use cm-1 for wn and unc
    Edges are edge indices with strictly no gaps (0 to N-1) for N levels
    By default ground level E is fixed at zero (1e-4 cm-1)
    
    Examples:
        # Four levels with transitions
        wn_obs = np.array([18883.6687, 20348.7431, 17745.8752, 19210.9460]), 
        wn_obs_unc = np.array([0.0018, 0.0021, 0.0028, 0.0024]),
        edges = np.array([[5, 0], [6, 0], [5, 1], [6, 1]]),

        # Ground term fixed
        fixed_lev_indices = np.array([0, 1, 2, 3, 4]),
        fixed_lev_values = np.array([0.0, 1137.7937, 2387.5440, 3714.5476, 5093.2573])  # makesure on same scale!
    '''
    # Get free lev indices
    all_indices = np.arange(N_lev)
    mask = np.ones(N_lev, dtype=bool)
    mask[fixed_lev_indices] = False
    free_lev_indices = all_indices[mask]
    if len(free_lev_indices) == 0:
        return fixed_lev_values, np.array([]), np.array([])
    
    # Prepare matrix eqn -------------------------------
    
    N = N_lev  # Number of nodes (no. fixed + no. to be optimised)
    M = len(edges)

    # Construct the incidence matrix A
    A = np.zeros((M, N)) # extra rows for fixed level transitions to ground

    # Add observed edges (direct matrix assignment)
    rows = np.arange(M)
    A[rows, edges[:, 0]] = 1 # upper levels
    A[rows, edges[:, 1]] = -1 # lower levels

    # Solve free separately from fixed ------------------

    # Remove fixed lev values in wn_obs
    A_free = A[:, free_lev_indices]
    A_fixed = A[:, fixed_lev_indices]
    rhs = wn_obs - A_fixed @ fixed_lev_values  # this determines free levs offset relative to fixed levs

    # Solve A_free @ E_free = rhs -----------------------

    # Weight the incidence matrix A_free and wn_obs
    weights = 1 / wn_obs_unc**2 
    W = np.sqrt(weights)

    A_weighted = A_free * W[:, np.newaxis]  # Weight A
    rhs_weighted  = rhs * W  # Weight wn_obs

    # Scale for stability
    scale = np.max(np.abs(rhs_weighted))

    # Solve the weighted least squares problem
    try:
        E_free_scaled, *_ = np.linalg.lstsq(A_weighted, rhs_weighted / scale, rcond=None)
    except np.linalg.LinAlgError:
        print('lstsq failed to converge in LOPT, treating this A2 as invalid')
        # I never got this error
        return None, None, None
    E_free = E_free_scaled * scale

    # Get results ---------------------------------------

    # Reconstruct full E vector
    E = np.zeros(N)
    E[fixed_lev_indices] = fixed_lev_values
    E[free_lev_indices] = E_free

    wn_ritz = A @ E
    d_wn = np.abs(wn_obs - wn_ritz)
    
    if print_results:
        print(f'Optimised E values: {np.round(E, 4)}')
        print(f'Obs WN unc: {np.round(wn_obs_unc, 4)}')
        print(f'obs - Ritz: {np.round(d_wn, 4)}')

    return E, wn_ritz, d_wn # ignore transitions for fixed levs


# %% For agent and enviroment

def change_state(new_graph, linelist, E_from_clevs, level_to_find, clev_indices, cwns, cwn_uncs, cIobss, cIcalcs, csnr_obs):
    '''
    Change graph and line list states using a LEVHAM candidate
        level_to_find - dim1 size1 tensor
        clev_indices, cwns, cwn_uncs, cIobss all shape [N_connecting_levels]
    '''
    level_to_find = level_to_find[0] # make a torch long (integer)

    E_tentative = torch.mean(E_from_clevs)
    # let level_to_find be now known and unselected on the new graph
    # [E_calc, E_obs, J, even, odd, known, unknown, selected, unselected]
    node_feats = dqn_data_proc.FeatureIndexer.node_feature_indices(['E_obs', 'known', 'unknown', 'selected', 'unselected'])
    new_graph.x[level_to_find, torch.tensor(node_feats)] = torch.tensor([E_tentative, 1., 0., 0., 1.], dtype=torch.float64)
    # new_graph.x[level_to_find, torch.tensor([1,2,3,4,5])] = torch.tensor([E_tentative, 1., 0., 0., 1.], dtype=torch.float64)

    # Graph edge adjustments------------------------------------------------------------
    wn_obs_indices = torch.empty(0, dtype=torch.long)
    edge_indices = torch.empty(0, dtype=torch.long)
    edge_feats_names = ['wn_obs', 'wn_obs_unc', 'I_calc', 'I_obs', 'snr_obs', 'known', 'unknown']
    edge_feats = dqn_data_proc.FeatureIndexer.edge_feature_indices(edge_feats_names)
    #edge_feats = [1, 2, 3, 4, 7, 9, 10]
    for i, clev in enumerate(clev_indices):
        wn_obs = cwns[i]
        wn_unc = cwn_uncs[i]
        I_obs = cIobss[i]
        I_calc = cIcalcs[i]
        snr_obs = csnr_obs[i]
        
        # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
        # Alter new_graph edges using classified lines [wn, wn_unc, Icalc, Iobs, snr_obs, known, unknown], others stay the same
        edge_attr_change = torch.tensor([wn_obs, wn_unc, I_calc, I_obs, snr_obs, 1, 0], dtype=torch.float64)
        edge_idx = torch.where((new_graph.edge_index[0] == level_to_find) & (new_graph.edge_index[1] == clev))[0][0]
        new_graph.edge_attr[edge_idx][edge_feats] = edge_attr_change
        # Undirected graph so alter also the other direction
        shift = new_graph.edge_attr.size(0) // 2
        if edge_idx < shift:
            undir_edge_idx = edge_idx + shift
        else:
            undir_edge_idx = edge_idx - shift
        #undir_edge_idx = torch.where((new_graph.edge_index[0] == clev) & (new_graph.edge_index[1] == level_to_find))[0][0]
        new_graph.edge_attr[undir_edge_idx][edge_feats] = edge_attr_change
        edge_indices = torch.cat([edge_indices, edge_idx.unsqueeze(0)], dim=0)  # emission edge index


        wn_obs_index = torch.argmin(torch.abs(wn_obs - linelist[:, 0]))  # index of observed line in the line list
        wn_obs_indices = torch.cat([wn_obs_indices, wn_obs_index.unsqueeze(0)], dim=0)

    return new_graph, wn_obs_indices, edge_indices

def assess_cand(new_graph, E_scale, fixed_lev_indices, fixed_lev_values, threshold=1.2):
    '''
    Assess candidate using LOPT, returns new graph with optimised energies
    '''

    # Run LOPT mini to check known level system consistency-------------------------------------------
    E, wn_obs_unc, d_wn, known_level_indices = lopt_known_subgraph(new_graph, E_scale, 
                                                                   fixed_lev_indices, fixed_lev_values) # d_wn is obs-ritz
    if E is None:  # failed lstsq
        return None

    # LOPT A2 MASKING DISABLED FOR NOW
    # # actual LOPT is 1.2 I think, when it raises a !
    # if len(d_wn) > 0 and (d_wn > threshold * wn_obs_unc).any():  # If any |wn_obs - wn_ritz| is greater than threshold * wn_obs_unc
    #     if d_wn.max() > 1:
    #         print('LOPT might have exploded, d_wn max:', d_wn.max().item(), ' cm-1')
    #     return None
    
    # Passed LOPT check
    # Update graph E_obs
    E = E * 1e-3 / E_scale # change units from cm-1 to kK / scale
    new_graph.x[known_level_indices, 1] = torch.tensor(E, dtype=torch.float64)

    # Accept changes to graph
    return new_graph
    
def lopt_known_subgraph(new_graph, E_scale, fixed_lev_indices, fixed_lev_values, test=False):
    '''Returns subgraph of the known levels retaining indices of the full graph'''
    # Identify known levels (nodes with feature[2] = 1)
    # [E_calc, E_obs, J, even, odd, known, unknown, selected, unselected]
    known_levels = new_graph.x[:, 2] == 1  # Boolean mask for known levels
    known_indices = torch.nonzero(known_levels).flatten()  # Indices of known nodes

    # Get the subgraph of known indices
    edge_index, edge_attr = subgraph(known_indices, 
                            new_graph.edge_index, 
                            edge_attr=new_graph.edge_attr, 
                            relabel_nodes=False)
    
    # Get edges that are classified lines
    classified_lines = edge_attr[:, -1] == 0
    edge_attr = edge_attr[classified_lines]
    edge_index = edge_index[:, classified_lines]

    # Since edges are undirected, use first half and the source node is always upper level (from Kramida's lid2)
    edge_index = edge_index[:, :edge_index.shape[1] // 2]
    edge_attr = edge_attr[:edge_attr.shape[0] // 2]

    # Node mapping between sub and full graphs
    mapping = torch.zeros(known_indices.max() + 1, dtype=torch.long)  # zeros of size max full graph index + 1
    mapping[known_indices] = torch.arange(len(known_indices))  # full graph index positions are mapped to subgraph indices
    edge_index = mapping[edge_index]  # map edge indices (pairs) to subgraph indices
    N_lev = edge_index.max().item() + 1  # Number of levels in the subgraph (max index + 1)
    edges = edge_index.T.numpy()  # [N_known_lines, 2], col 1 is upper level, col 2 is lower level
    fixed_lev_indices = mapping[fixed_lev_indices].numpy()  # map fixed lev index to sub graph index

    # Get observed wavenumbers of known levels
    wn_obs = edge_attr[:, 1].numpy() * E_scale * 1e3 # scale to cm-1, 1D np array
    wn_obs_unc = dqn_data_proc.wn_unc_from_graph(edge_attr[:, 2].numpy()) * 1e3 # scale to cm-1, 1D np array

    # Make sure fixed_lev_values are on the same scale as wn_obs!!!
    fixed_lev_values = fixed_lev_values * E_scale * 1e3 # scale to cm-1, 1D np array

    if test:  # if testing initial graph (fixed) level system consistency 
        # all floating except for ground
        fixed_lev_indices = np.array([0])
        fixed_lev_values = np.array([0.])
    else:
        # Remove edges that are with fixed levels， helps a lot for MCTS
        floating_lev_edges = ~np.isin(edges, fixed_lev_indices).all(axis=1)  # edges that are not purely between fixed levels
        edges = edges[floating_lev_edges]  # only keep edges that are not purely between fixed levels
        wn_obs = wn_obs[floating_lev_edges]
        wn_obs_unc = wn_obs_unc[floating_lev_edges]

    # LOPT mini (uncertainties weighted lsq fit)
    E, wn_ritz, d_wn = lopt(wn_obs, wn_obs_unc, N_lev, edges, fixed_lev_indices, fixed_lev_values)
    # E, wn_ritz, d_wn, self.fixed_lev_values in cm-1

    return E, wn_obs_unc, d_wn, known_indices 