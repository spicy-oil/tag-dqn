'''
LEVHAM (originally Fortran programm from Lund Univ., there is no citation for this) implemented for A2
'''

import torch
import torch.nn as nn
from torch_geometric.utils import k_hop_subgraph
from . import dqn_data_proc
from torch_geometric.data import Data
from . import lopt

def get_levham_inputs(lev_selection, graph):
    '''
    Get connecting known level indices, energies, and lines to find lev_selection in LEVHAM
    '''
    # Full graph indices of levels connected to the selected level by one edge
    lev_selection_subgraph_node_indices, _, _, edge_mask = k_hop_subgraph(lev_selection, 1, 
                                                                graph.edge_index, # from known + 1 subgraph only
                                                                relabel_nodes=False)
    # Full graph indices of known levels connected to the selected level by one edge
    # [E_calc, E_obs, J, even, odd, known, unknown, selected, unselected]
    known_level_indices = torch.where(graph.x[:, 2] == 1)[0]
    known_clev_mask = torch.isin(lev_selection_subgraph_node_indices, known_level_indices)
    lev_selection_subgraph_node_indices = lev_selection_subgraph_node_indices[known_clev_mask]
    neighbourhood_edge_index = graph.edge_index[0][edge_mask]
    neighbourhood_edge_attr = graph.edge_attr[edge_mask]
    
    # Remove index of the selected level
    clev_indices = lev_selection_subgraph_node_indices[lev_selection_subgraph_node_indices != lev_selection]

    # Get known clev Es and lines
    clev_energies = graph.x[clev_indices, 1] # use E_obs for LEVHAM
    clev_lines = torch.stack([neighbourhood_edge_attr[neighbourhood_edge_index == i][0] for i in clev_indices])
    
    return clev_indices, clev_energies, clev_lines

def LEVHAM(graph, linelist, E_scale, lev_selection, linelist_mapping,
           levham_lev_indices, levham_lev_energies, levham_lev_lines, 
           fixed_lev_indices, fixed_lev_values,
           wn_range, tol, int_tol,
           learning=False, nnham=None):
    '''
    LEVHAM algorithm, wn_range and tol in kK but will be scaled by self.E_scale
    LOPT is run for each candidate, only those pass LOPT are output as valid actions
        graph - full graph state
        linelist - line list [N_lines, 3] tensor
        lev_selection - index of the level to find, dim=1 size=1 torch tensor
        levham_lev_indices - full graph indices of connecting levels input to LEVHAM, dim=1 tensor
        levham_lev_energies - the energies of the connecting levels, dim=1 tensor
        levham_lev_lines - graph edge attributes of the connecting levels, N_line dim=10 tensors
    '''
    # Search using unclassified lines
    ll_idx = torch.arange(0, linelist.size(0))  # this is just the index for linelist
    classified_lines_ll_idx = linelist_mapping[linelist_mapping != -1]  # linelist index of already classified lines in linelist
    unclassified_lines_ll_idx = ~torch.isin(ll_idx, classified_lines_ll_idx)  # boolean linelist index for unclassified lines
    unclassified_lines = linelist[unclassified_lines_ll_idx]  # [N_lines, 3] tensor of unclassified lines
    
    pred_E = graph.x[lev_selection][0][0] # already scaled
    wn_obs = unclassified_lines[:, 0] # already scaled
    wn_obs_unc = unclassified_lines[:, 1] # already scaled
    I_obs = unclassified_lines[:, 2]  # log10(I_obs) / 5
    snr_obs = unclassified_lines[:, 3]  # log10(SNR_obs) / 5

    cand_wn = torch.empty(0, dtype=torch.float64)
    cand_wn_unc = torch.empty(0, dtype=torch.float64)
    cand_Iobs = torch.empty(0, dtype=torch.float64)
    cand_Icalc = torch.empty(0, dtype=torch.float64)
    cand_snr_obs = torch.empty(0, dtype=torch.float64) 
    cand_clev = torch.empty(0, dtype=torch.float64)
    cand_clev_indices = torch.empty(0, dtype=torch.long)
    lev_candidates = torch.empty(0, dtype=torch.float64)
    
    cline_prob_not_missing = torch.empty(0, dtype=torch.float64)  # probability of missing line
    # WN and intensity filtering ---------------------------------------------------------------------------------------
    m, c, _, _ = dqn_data_proc.fit_pop(graph, E_scale, plot=False)  # E_scale only for plotting, so x axis is kK / E_Scale
    for i, lev_id in enumerate(levham_lev_indices): # probably fine, only a few levels
        # For each connecting level, filter possible lines    
        # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
        line = levham_lev_lines[i]

        upper_level = torch.maximum(pred_E, levham_lev_energies[i])
        gA_calc = line[5] * 10 # log10(gA_calc)
        I_calc = (gA_calc + m * upper_level + c) / 5  # log10(I_calc) / 5

        # I_calc = line[3]  # log10(I_calc) / 5

        valid_wn = torch.abs(line[0] - wn_obs) < (wn_range / E_scale) # within wn search range
        valid_intensity = torch.abs(I_calc - I_obs) * 5 < (int_tol)  # within ~1 order of magnitude
        valid_lines = valid_wn & valid_intensity

        # Prob not missing in line list
        snr_calc = line[6]
        #snr_in_range = snr_obs[valid_wn]
        #N_maskable = (snr_in_range > (snr_calc + 1 / 5)).sum() # number of lines that can mask this line
        #mask_density = N_maskable / (2 * wn_range)  # lines per 1000 cm-1
        #prob_not_missing = torch.tensor([1 - (2 *  tol * mask_density + (10 ** (snr_calc * 5) < 5))])
        prob_not_missing = torch.tensor([1 - (10 ** (snr_calc * 5) < 5).float()])  # if SNR < 5, then line is not missing, so prob = 1
        cline_prob_not_missing = torch.cat([cline_prob_not_missing, prob_not_missing])

        cand_wn_i = wn_obs[valid_lines]
        cand_wn = torch.cat([cand_wn, cand_wn_i])

        cand_wn_unc_i = wn_obs_unc[valid_lines]
        cand_wn_unc = torch.cat([cand_wn_unc, cand_wn_unc_i])

        cand_Iobs_i = I_obs[valid_lines]
        cand_Iobs = torch.cat([cand_Iobs, cand_Iobs_i])

        cand_Icalc = torch.cat([cand_Icalc, I_calc.expand(cand_Iobs_i.size(0))])

        cand_snr_obs_i = snr_obs[valid_lines]
        cand_snr_obs = torch.cat([cand_snr_obs, cand_snr_obs_i])

        sign = torch.sign(pred_E - levham_lev_energies[i]) # + if upper level, - if lower level
        lev_candidates_i = levham_lev_energies[i] + sign * cand_wn_i
        lev_candidates = torch.cat([lev_candidates, lev_candidates_i])

        cand_clev_i = torch.ones_like(cand_wn_i) * levham_lev_energies[i]
        cand_clev = torch.cat([cand_clev, cand_clev_i])

        cand_clev_indices_i = torch.zeros_like(cand_wn_i, dtype=torch.long) + lev_id
        cand_clev_indices = torch.cat([cand_clev_indices, cand_clev_indices_i])


    # sort by candidate energy
    sort_indices = torch.argsort(lev_candidates)
    lev_candidates = lev_candidates.index_select(0, sort_indices)
    cand_wn = torch.abs(cand_wn.index_select(0, sort_indices)) # transitions go both ways so take abs value
    cand_wn_unc = cand_wn_unc.index_select(0, sort_indices)
    cand_Iobs = cand_Iobs.index_select(0, sort_indices)
    cand_Icalc = cand_Icalc.index_select(0, sort_indices)
    cand_snr_obs = cand_snr_obs.index_select(0, sort_indices)
    cand_clev = cand_clev.index_select(0, sort_indices)
    cand_clev_indices = cand_clev_indices.index_select(0, sort_indices)

    # Obtain candidate levels --------------------------------------------------------------------------------------
    diff = torch.diff(lev_candidates) # shape [lev_candidates - 1]

    # Find indices where the gap is greater than tol (these indicate group boundaries)
    group_end_indices = torch.where(diff >= (tol / E_scale))[0] + 1  # Shift index by 1 since `diff` is one element shorter
    # Split indices into groups
    all_indices = torch.arange(lev_candidates.shape[0])
    split_sizes = torch.diff(
        torch.cat([
            torch.tensor([0]), 
            group_end_indices, 
            torch.tensor([lev_candidates.shape[0]])
        ])
    ).tolist()
    groups = torch.split(all_indices, split_sizes)

    # Get candidates with at least n_lines_to_look_for
    if learning:
        n_lines_to_look_for = 2 # minimum number of lines to known levels for a valid candidate
    else:
        if len(levham_lev_indices) > 2:
            probs, _ = torch.topk(cline_prob_not_missing, k=3, largest=True)
            if torch.prod(probs) > 0.99:
                #print(f'N_repeat = 3 for {lev_selection.item()}')
                n_lines_to_look_for = 3
                if len(levham_lev_indices) > 3:
                    probs, _ = torch.topk(cline_prob_not_missing, k=4, largest=True)
                    if torch.prod(probs) > 0.99:
                        n_lines_to_look_for = 4
                        #print('N_repeat = 4!!')
                    else:
                        n_lines_to_look_for = 3
            else:
                n_lines_to_look_for = 2
        else:
            n_lines_to_look_for = 2 # minimum number of lines to known levels for a valid candidate
    # n_lines_to_look_for = 2 # minimum number of lines to known levels for a valid candidate
    cand_indices_groups = [g.tolist() for g in groups if len(g) >= n_lines_to_look_for]

    # Get candidate graphs (candidate level + candidate lines) and their values ----------------------------------------------
    cand_graphs = []
    cand_energies = []
    #cand_E_obs = []
    #cand_known_unknown = []
    cand_wn_obs_indices = [] 
    cand_edge_indices = []
    #cand_edge_attr = []
    cands_prelopt = []  # each is [lev, cwn, cwn_unc, cIobs, cIcalc]
    for i, g in enumerate(cand_indices_groups):
        # extract candidate data
        lev = lev_candidates[g]
        clev = cand_clev[g] # [N_clev]
        cwn = cand_wn[g] # [N_clev]
        cwn_unc = cand_wn_unc[g] # [N_clev]
        cIobs = cand_Iobs[g] # [N_clev]
        cIcalc = cand_Icalc[g] # [N_clev]
        csnr_obs = cand_snr_obs[g] # [N_clev]
        cindices = cand_clev_indices[g] # [N_clev], full graph clev indices

        # Handle when more than one line connected to a clev... (lines closer than TOL or multiply identified lines usually)
        clev_sorted = torch.unique(cindices)
        if clev_sorted.size() != cindices.size():
            clean_index = []
            for c in clev_sorted:
                # take first instance of any repeated lev
                clean_index.append(torch.isin(cindices, c).long().argmax()) 
            clean_index = torch.stack(clean_index)
            lev = lev[clean_index] # [N_clev]
            clev = clev[clean_index] # [N_clev], was unsqueezed earlier
            cwn = cwn[clean_index] # [N_clev]
            cwn_unc = cwn_unc[clean_index] # [N_clev]
            cIobs = cIobs[clean_index] # [N_clev]
            csnr_obs = csnr_obs[clean_index] # [N_clev]
            cIcalc = cIcalc[clean_index]
            cindices = cindices[clean_index] # [N_clev], full graph clev indices
        if lev.size(0) < n_lines_to_look_for: # skip this loop if just one level with multiply identified line
            continue

        # NN decision on blends or problem lines
        if nnham is not None:
            cwn_unc = NN_levham_mask(nnham, lev, E_scale, cwn_unc, cIobs, cIcalc)

        # Create dummy graph and line list states to be assessed by LOPT ----------------------------------------------------
        new_graph = Data(x=graph.x.clone(), 
                            edge_index=graph.edge_index.clone(), 
                            edge_attr=graph.edge_attr.clone()
                        ) # ensure independent copy
        linelist_clone = linelist.clone() # ensure deep independent copy
        # Apply changes this candidate would have on graph and linelist
        new_graph, wn_obs_indices, edge_indices = lopt.change_state(new_graph, linelist_clone, lev,
                                        lev_selection, cindices, cwn, cwn_unc, cIobs, cIcalc, csnr_obs)
        # Check if this candidate passes LOPT
        # new_graph = lopt.assess_cand(new_graph, E_scale, fixed_lev_indices, fixed_lev_values, threshold=1.2)

        # Add candidate for DQN only if LOPT is good
        if new_graph is not None:
            if learning:
                # always learning using known lines
                # problem line cwn_unc is 1 cm-1, so will need the actual unc, which can be found from cwn + raw line list
                cands_prelopt.append([lev, cwn, cwn_unc, cIobs, cIcalc])
            cand_graphs.append(new_graph)
            # Track changes in graph and line list
            lev = new_graph.x[lev_selection][0][1]
            cand_energies.append(float(round(lev.item() * E_scale * 1e3, 4)))  # a number for debugging mainly
            cand_wn_obs_indices.append(wn_obs_indices)  # indices of lines to used from line list
            cand_edge_indices.append(edge_indices)

    if learning:
        return (cand_graphs, # list of candidate graphs (full size)
            cand_energies, # list of candidate energies in cm-1
            cand_wn_obs_indices, # N_cand lists of indices to remove from line list
            cand_edge_indices, # N_cand lists of edge indices on full graph that were modified for each cand_graph
            cands_prelopt)  # N_pre_lopt_cand of [lev, cwn, cwn_unc, cIobs, cIcalc]
    else:
        return (cand_graphs, # list of candidate graphs (full size)
                cand_energies, # list of candidate energies in cm-1
                cand_wn_obs_indices, # N_cand lists of indices to remove from line list
                cand_edge_indices) # N_cand lists of edge indices on full graph that were modified for each cand_graph

def nnham_in(E_obs, E_scale, cwn_unc_ll, cIobs, cIcalc):
    # Get the smallest absolute deviation from neighbour, 1 neigh if highest/lowest E_obs, 2 neigh otherwise
    E_obs_min_diff = min_diff(E_obs) * E_scale + 1e-6  # kK, 1 mK minimum so we don't log zero
    E_obs_min_diff = dqn_data_proc.wn_unc_to_graph(E_obs_min_diff)  # to wn unc scale for GNN

    # WN unc
    #cwn_unc_ll = torch.tensor(cwn_unc_ll).clone()  # use wn unc scaled for GNN

    # Intensities 
    cIobs = cIobs / cIobs.sum()
    cIcalc = cIcalc / cIcalc.sum()
    I_diff = torch.abs(cIobs - cIcalc)  # now using scaled values, more diff means more likely wrong intensity
    I_calc_order = torch.clamp(torch.argsort(cIcalc) - len(E_obs) + 4, 1, 3)
    I_diff = I_diff * I_calc_order  # so the strongest two line I_diff are more important

    x = torch.stack([E_obs_min_diff, cwn_unc_ll, E_obs_min_diff / cwn_unc_ll,
                      cIobs, cIcalc, I_diff]).T.float()

    return x

class NNHam(nn.Module):
    def __init__(self):
        super(NNHam, self).__init__()
        # Use log-params for positive values a = exp(log_a)
        # self.l1 = nn.Linear(in_features=5, out_features=8)
        # self.relu = nn.ReLU()
        # self.l2 = nn.Linear(16, 1)

        self.l1 = nn.Linear(in_features=6, out_features=16)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(16, 1)
        #self.l3 = nn.Linear(6, 1)

    def forward(self, x):
        # x (N_E_obs, 5)
        # N_E_obs = len(x)
        # x = self.l1(x)  # (N_E_obs, 8)
        # x = self.relu(x)
        # x_agg = x.mean(dim=0).expand(N_E_obs, 8)  # (N_E_obs, 8)
        # x_context = torch.cat([x, x_agg], dim=1)  # (N_E_obs, 16)
        # x = self.l2(x_context)  # (N_E_obs, 1)
        # return x.flatten()  # (N_E_obs, )

        N_E_obs = len(x)
        x_context = self.l1(x)
        x_context = self.relu(x_context)
        x_context = self.l2(x_context)
        #x_context = x_context.mean(dim=0).expand(N_E_obs, 1)  # (N_E_obs, 4)
        #x = torch.cat([x, x_context], dim=1)
        #x = self.l3(x)  # (N_E_obs, 1)
        return x_context.squeeze()  # (N_E_obs, )
    
def min_diff(E_obs):
    E_obs_prev = torch.roll(E_obs, 1)
    E_obs_next = torch.roll(E_obs, -1)
    E_obs_prev[0] = 1e6
    E_obs_next[-1] = 1e6
    diff_prev = torch.abs(E_obs - E_obs_prev)
    diff_next = torch.abs(E_obs - E_obs_next)
    return torch.minimum(diff_prev, diff_next)

def NN_levham_mask(nnham, lev, E_scale, cwn_unc, cIobs, cIcalc):
    x = nnham_in(lev, E_scale, cwn_unc, cIobs, cIcalc)
    prob = torch.sigmoid(nnham(x))
    mask = prob > 0.2  # thresh
    cwn_unc[mask] = dqn_data_proc.wn_unc_to_graph(1e-3)  # kK to graph
    return cwn_unc

# Get classified lines for each demo
def get_ham_data(demos, E_scale, linelist, tol):
    '''linelist is linelist with bad unc (not 1 cm-1 to weight zero in lopt)'''
    X = []
    Y = []
    tol = tol / E_scale
    neg_count = 0
    pos_count = 0
    for demo in demos:
        # demo (s, |A|, a, s'), s is the graph
        # A_info = cand_graphs, cand_energies, cand_wn_obs_indices, cand_edge_indices, cand_prelopt
        A_idx = demo[2]  # expert action idx
        A_space = demo[1]  # action space
        # cl_idx = A_space[3][A_idx]  # cand_edge_indices of expert action
        # E_opt = A_space[1][A_idx]  # optimised energy of the level found of expert action
        E_obs, cwn, cwn_unc, cIobs, cIcalc = A_space[4][A_idx]  # E_obs, cwn, cwn_unc, cIobs, cIcalc

        # Locate actual unc because from demo cwn_unc of problem lines are at 1 cm-1
        cwn_unc_ll = []  # linelist wn_unc, before a human marks it as high wn unc problem line
        cwn_unc_label = []  # whether a human has marked it as a problem line
        for i, cw in enumerate(cwn):
            ll_unc = linelist[:, 1][abs(linelist[:, 0] - cw).argmin()]
            cwn_unc_ll.append(ll_unc)
            if cwn_unc[i] < -0.99:  # if 1 cm-1
                cwn_unc_label.append(1)
                pos_count += 1
            else:
                cwn_unc_label.append(0)
                neg_count += 1
        cwn_unc_ll = torch.tensor(cwn_unc_ll).float()
        y = torch.tensor(cwn_unc_label).float()
        
        # Now design x for each E_obs
        x = nnham_in(E_obs, E_scale, cwn_unc_ll, cIobs, cIcalc)

        X.append(x)
        Y.append(y)

    pos_weight = neg_count / pos_count
    return X, Y, pos_weight