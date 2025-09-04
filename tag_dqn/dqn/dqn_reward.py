'''
Reward function and reward learning
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.transforms.largest_connected_components as lcc 
import numpy as np
import random

from . import dqn_data_proc
from .levham import min_diff
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from tqdm import tqdm

def reward_classified_lines(graph, cand_edge_indices, E_scale, wn_range, tol, learn_reward=False):
    '''
    Old manually defined reward calculation for classified lines,
    performance poorer than the trained NN reward
    '''
    # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
    attrs = graph.edge_attr[cand_edge_indices]  # edge_attr of the edges to consider reward from
    # Line snr reward (base)
    snr_calc = attrs[:, 6] * 5  # so that snr_calc is in log10 units
    tot_exp_snr = torch.log10(torch.sum(10 ** snr_calc))
    # Line density discount
    line_den = dqn_data_proc.line_den_from_graph(attrs[:, 8])  # lines per 1000 cm-1
    rhos, _ = torch.topk(line_den, k=2, largest=False)  # get smallest two line densities, lines per kK
    x0 = torch.clip(torch.prod(rhos) * 4 * wn_range * tol, 0, 10)
    #N_cand_discount = 0.9 ** x0  # LEVHAM search params in kK units
    # Line intensity discount
    line_intensities = attrs[:, [3, 4]]
    line_intensities[:, 0] = line_intensities[:, 0] / line_intensities[:, 0].max()  # normalise I_calc
    line_intensities[:, 1] = line_intensities[:, 1] / line_intensities[:, 1].max()  # normalise I_obs
    x1 = torch.abs(torch.diff(line_intensities, dim=1)).mean()  # should be no more than 1 because filtered within +- 1 intensity
    #rel_int_discount = torch.exp( - mean_diff)  # between 1 and approx. e^(-1) when intensity tol is 1 in LEVHAM
    # LEVHAM WN Tolerance discount
    wn_obs = attrs[:, 1]
    wn_obs_unc = attrs[:, 2]
    line_lev_Es = torch.empty(0, dtype=torch.float64)
    for i, index in enumerate(cand_edge_indices):
        lev_id, clev_id = graph.edge_index[:, index]  # index is from lopt.change_state(), lev_id is always source
        lev_E_obs = graph.x[lev_id, 1]
        clev_E_obs = graph.x[clev_id, 1]
        if lev_E_obs > clev_E_obs: # if finding upper level
            line_lev_Es = torch.cat((line_lev_Es, (clev_E_obs + wn_obs[i]).unsqueeze(0)))
        else: # if finding lower level
            line_lev_Es = torch.cat((line_lev_Es, (clev_E_obs - wn_obs[i]).unsqueeze(0)))
        
    max_diff = (line_lev_Es.max() - line_lev_Es.min()) * 1e6 * E_scale # mK
    min_unc = dqn_data_proc.wn_unc_from_graph(wn_obs_unc.min()) * 1e6  # mK
    # Lower reward if max_diff ~ tol
    # Higher reward if max_diff ~ min_unc
    x2 = torch.clip(max_diff - min_unc, 0, None) / (tol * 1e6) # [0, ~1]
    #wn_tol_discount = torch.exp(-max_diff) # between 1 and approx e^(-1)
    if learn_reward:  # only for comparison with NNRw
        x = [x0, x1, x2]
        return (
                torch.tensor(x, dtype=torch.float),  # old reward inputs
                0.9 ** x0 * torch.exp(-x1) * torch.exp(-x2)  # old reward discount
        )
    else:
        return tot_exp_snr * 0.9 ** x0 * torch.exp(-x1) * torch.exp(-x2)  # old reward  (discount * base)

def NN_reward_input(graph, cand_edge_indices, E_scale):
    '''Making input features that are as generalisable as possible across different term analyses/graph states'''
    # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
    # nodes, counts = torch.unique(graph.edge_index[:, cand_edge_indices], return_counts=True)
    # level_found = nodes[counts.argmax()]  # full graph idx
    # boo = torch.isin(graph.edge_index[0], level_found)  # just need src because undirected graph
    # edge_attr = graph.edge_attr[boo]
    edge_attr = graph.edge_attr[cand_edge_indices]  # edge_attr of the known edges to consider reward from
    known_idx = edge_attr[:, -2] == 1
    N_lines = len(edge_attr)

    # Scale for the difference between observed and calculated wn
    # Don't want this because o_c tend to be bigger for unknown levels! So generalisation would be poor
    # diff_scale = wn_range / E_scale  
    # o_c = (edge_attr[:, 1] - edge_attr[:, 0]) / diff_scale  # (wn_obs - wn_calc) / scale
    # o_c[~known_idx] = 0.0  # unknown lines stay 0 

    # LEVHAM WN Tolerance discount, this needs to be done with double floats
    known_attr = edge_attr[known_idx]
    wn_obs = known_attr[:, 1]
    wn_obs_unc = known_attr[:, 2]
    line_lev_Es = torch.empty(0, dtype=torch.float64)
    for i, index in enumerate(cand_edge_indices):
        lev_id, clev_id = graph.edge_index[:, index]  # index is from lopt.change_state(), lev_id is always source
        lev_E_obs = graph.x[lev_id, 1]
        clev_E_obs = graph.x[clev_id, 1]
        if lev_E_obs > clev_E_obs: # if finding upper level
            line_lev_Es = torch.cat((line_lev_Es, (clev_E_obs + wn_obs[i]).unsqueeze(0)))
        else: # if finding lower level
            line_lev_Es = torch.cat((line_lev_Es, (clev_E_obs - wn_obs[i]).unsqueeze(0)))
        
    wn_diff = min_diff(line_lev_Es) * E_scale + 1e-6  # kK, 1 mK minimum so we don't log zero
    wn_diff = dqn_data_proc.wn_unc_to_graph(wn_diff)  # to GNN wn unc scale
    wn_unc_agreement = wn_obs_unc - torch.abs(wn_diff)  # larger means better

    I_obs = edge_attr[:, 4]
    I_calc = edge_attr[:, 3]

    # Normalise against largest intensity
    I_obs = I_obs / I_obs.max()
    I_calc = I_calc / I_calc.max()

    I_diff = -torch.abs(I_obs - I_calc)  # now using scaled values, negative abs so smaller diff is better
    # I_calc_order = torch.clamp(torch.argsort(edge_attr[:, 3]) - N_lines + 4, 1, 3)
    # I_diff = I_diff * I_calc_order  # so the strongest two line I_diff are more important

    # Keep line_den, known, unknown one-hot encoding
    line_den = edge_attr[:, -3]  # Use graph units

    #snr_obs = torch.clamp(edge_attr[:, 7], 0, 0.2)  # Use graph units, clamp to 10 snr

    # Number of lines
    # N_lines = torch.tensor(N_lines).expand(N_lines) / 5

    x = torch.stack([wn_obs_unc, wn_diff, wn_unc_agreement, I_obs, I_calc, I_diff, line_den]).T
    return x.float()  # (N_known_lines, 7)

def collect_demos(init_graph, env, E_scale, wn_range, tol):
    '''Collect demonstrations'''
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    graph = init_graph.clone()
    graph, env = remove_branch_nodes(graph, env)  # so that all nodes are in loops, because MDP exludes single line identifications
    demos = []

    for _ in tqdm(range(200)):
        A = A_rev(graph, env.linelist_mapping)  # if removed, there should be no branches and no disconnected graphs
        if len(A) == 0:  # no more valid removals, reverse episode ends
            print('No more valid removals')
            break
        A_prob = np.array([A_rev_rw(graph, i) for i in A])
        A_prob = torch.softmax(torch.tensor(A_prob) * -2, dim=0)  # lower rewards have higher prob of being the prev action
        A_choice = np.random.choice(A, p=A_prob)  # the node that was the previous action
        graph, A1, A1_index, A2, A2_index, next_graph = rev_step(env, graph, A_choice, E_scale)
        if len(A2[0]) > 2:  # if more than one candidate, learn
            demos.append((graph.clone(), A2, A2_index, next_graph.clone(), (E_scale, wn_range, tol)))
    return demos

def mark_prob_known_lines(known_lines, linelist, E_scale):
    '''Let problem known lines have high unc in line list'''
    for i, row in known_lines.iterrows():
        if row.wn_unc * 1e3 > 0.99:
            wn = row.wn / E_scale
            idx = torch.argmin(torch.abs(linelist[:, 0] - wn))  # linelist idx
            linelist[idx][1] = dqn_data_proc.wn_unc_to_graph(1e-3)  # kK to graph units 
    return linelist

def mark_ll_known_lines(init_graph, linelist):
    '''Let known lines have linelist_mapping, so that when removed, they appear in LEVHAM'''
    linelist_mapping = torch.zeros(init_graph.edge_attr.size(0), dtype=torch.long) - 1 # mapping from graph edge_attr to linelist
    # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
    for i, e in enumerate(init_graph.edge_attr):
        if e[-2] == 1:
            wn = e[1]
            idx = torch.argmin(torch.abs(linelist[:, 0] - wn))  # linelist idx
            linelist_mapping[i] = idx
    return linelist_mapping

# Remove non-looping nodes
def get_known_subgraph(graph, relabel=False):
    # Identify known levels (nodes with feature[2] = 1)
    # [E_calc, E_obs, J, even, odd, known, unknown, selected, unselected]
    known_x = graph.x[:, 2] == 1  # Boolean mask for known levels
    known_x_index = torch.nonzero(known_x).flatten()  # Indices of known nodes

    # Get the subgraph of known indices
    known_edge_index, known_edge_attr = subgraph(known_x_index, 
                            graph.edge_index, 
                            edge_attr=graph.edge_attr, 
                            relabel_nodes=relabel)

    # Get edges that are classified lines
    classified_lines = known_edge_attr[:, -1] == 0
    known_edge_attr = known_edge_attr[classified_lines]
    known_edge_index = known_edge_index[:, classified_lines]

    return known_x, known_x_index, known_edge_index, known_edge_attr

def remove_branch_nodes(graph, env):
    '''
    Make nodes on branches and their edges unknown
    '''
    known_x, known_x_index, known_edge_index, known_edge_attr = get_known_subgraph(graph)
    node_rem = []
    edge_rem = []
    for i in known_x_index:
        branch, edge_index = check_branch(graph, i, known_edge_index)
        if branch:  # because undirected
            print(i)
            node_rem.append(i)
            edge_rem = edge_rem + edge_index  # is list

    for i in node_rem:
        graph = remove_node(graph, i)
        fix_rem = np.where(env.fixed_lev_indices == i)[0][0]
        env.fixed_lev_indices = np.delete(env.fixed_lev_indices, fix_rem)
        env.fixed_lev_values = np.delete(env.fixed_lev_values, fix_rem)

    
    for i in edge_rem:
        graph, env.linelist_mapping = remove_edge(graph, env.linelist_mapping, i)
    
    return graph, env

def remove_node(graph, i):
    '''Change node[i] so that it becomes unknown'''
    # [E_calc, E_obs, known, unknown, selected, unselected]
    graph.x[i][[1,2,3]] = torch.tensor([0,0,1], dtype=torch.float64)
    return graph

def remove_edge(graph, linelist_mapping, i):
    '''Change edge[i] so that it becomes unknown'''
    # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
    graph.edge_attr[i][[1,2,4,7,9,10]] = torch.tensor([0,0,0,0,0,1], dtype=torch.float64)
    linelist_mapping[i] = -1
    return graph, linelist_mapping

def check_branch(graph, i, known_edge_index):
    '''
    Check if node i has only one edge on known graph, or has no edges, 
    if so returns the known graph edge_attr indices of edge and undirected edge,
    else return all known graph edge_attr indices with this node
    '''
    involved = (known_edge_index[0] == i) | (known_edge_index[1] == i)

    src = known_edge_index[0][involved]
    trg = known_edge_index[1][involved]
    edge_pairs = graph.edge_index.t()  # (num_edges, 2)
    target_pairs = torch.stack([src, trg], dim=1)  # (num_target_edges, 2)
    match = (edge_pairs[:, None] == target_pairs).all(-1).any(-1)  # (num_edges,)
    known_edge_index_i = torch.where(match)[0].tolist()

    #known_edge_index_i = [torch.where((graph.edge_index[0] == s) & (graph.edge_index[1] == t))[0][0].item()
                          #for s, t in zip(src, trg)]

    if len(known_edge_index_i) <= 2:
        return True, known_edge_index_i
    else:
        return False, known_edge_index_i
    
def remove_node_and_edges(graph, linelist_mapping, i, known_edge_index):
    # remove node i and its edges
    _, known_edge_index_i = check_branch(graph, i, known_edge_index)  # known_edge_index_i is full graph edge_attr index
    graph = remove_node(graph, i)
    for j in known_edge_index_i:
        graph, linelist_mapping = remove_edge(graph, linelist_mapping, j)
    return graph, linelist_mapping

def A_rev(graph, linelist_mapping):
    '''Compute reverse action space'''
    # To get largest connected component of a torch_geometric.data.Data graph
    # Strong becuase we work with undirected
    get_lcc = lcc.LargestConnectedComponents(num_components=1, connection='strong')  # Get only the first largest
    _, known_x_index, known_edge_index, _ = get_known_subgraph(graph)
    known_x_index = known_x_index.numpy()
    A = []  # valid node indices to remove (those that could have been the action pair that led to graph state)
    for i in known_x_index:
        if i == 0:
            continue
        old_graph = graph.clone()  # make a copy to make changes on
        old_linelist_mapping = linelist_mapping.clone()
        # remove node i and its edges
        old_graph, old_linelist_mapping = remove_node_and_edges(old_graph, old_linelist_mapping, i, known_edge_index)
        # Now check if there are branches
        old_known_x, old_x_index, old_edge_index, old_edge_attr = get_known_subgraph(old_graph)
        branch_flag = False
        for j in old_x_index:
            branch, _ = check_branch(old_graph, j, old_edge_index)
            if branch:
                branch_flag = True
                break
        if branch_flag:
            #print(f'There would be at least one branch or an isolated node from removing node {i}')
            continue  # i not added to A
        # Now check if there are isolated ndoes or graphs
        # Must relabel indices for the torch_geometric functions to work!!!
        old_known_x, old_x_index, old_edge_index, old_edge_attr = get_known_subgraph(old_graph, relabel=True)
        # if isolated.contains_isolated_nodes(old_edge_index):
        #     print(f'There would be at least an isolated node from removing node {i}')
        #     continue  # i not added to A
        old_graph = Data(torch.unique(old_edge_index), old_edge_index)
        if old_graph.x.shape != get_lcc(old_graph).x.shape:
            print(f'Largest connected component would not be the entire graph if removing {i}!')
            continue 

        # All OK, append
        A.append(i.item())
    return A

def A_rev_rw(graph, i):
    '''Reward for selecting a level as in A1, but this is used for probability of previous state transition'''
    # Reward for selecting the level
    cline_index = torch.where(i == graph.edge_index[0])  # undirected so compare with only source is fine
    clev_index = graph.edge_index[1][cline_index]
    known_clev_boo = graph.x[clev_index][:, 2] == 1
    expected_lines = graph.edge_attr[cline_index[0][known_clev_boo]]
    snr_calc = expected_lines[:, 6] * 5  # so that snr_calc is in log10 units
    tot_exp_snr = torch.log10(torch.sum(10 ** snr_calc))
    return tot_exp_snr.item()

def rev_step(env, graph, i, E_scale):
    '''Reverse env step and return demonstration'''
    next_graph = graph.clone()
    E_obs = graph.x[i][1].numpy() * 1e3 * E_scale  # cm-1 to mach with cand_energies
    print(f'Removing {i} with {E_obs:.4f}')
    _, _, known_edge_index, _ = get_known_subgraph(graph)
    env.graph = graph.clone()
    env.graph, env.linelist_mapping = remove_node_and_edges(graph, env.linelist_mapping, i, known_edge_index)
    fix_rem = np.where(env.fixed_lev_indices == i)[0][0]
    env.fixed_lev_indices = np.delete(env.fixed_lev_indices, fix_rem)
    env.fixed_lev_values = np.delete(env.fixed_lev_values, fix_rem)
    env.level_to_find = torch.tensor([i])
    A1 = env.A1_space(graph)  # valid node indices
    A1_index = torch.where(A1 == i)[0][0]
    A2 = env.A2_space_compute(learning=True)  # cand_graphs, cand_energies, cand_wn_obs_indices, cand_edge_indices
    A2_index = int(np.argmin(np.abs(np.array(A2[1]) - E_obs)))
    #print((env.linelist_mapping > -1).sum())
    return graph, A1, A1_index, A2, A2_index, next_graph

#%% Reward functions
class ParamReward(nn.Module):
    '''Not sure what the loss function would be'''
    def __init__(self):
        super().__init__()
        # Use log-params for positive values a = exp(log_a)
        self.log_a = nn.Parameter(torch.tensor(0.0))
        self.log_b = nn.Parameter(torch.tensor(0.0))
        self.log_c = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # x is a tensor like [x0, x1, x2]
        log_coeffs = torch.stack([self.log_a, -self.log_b, -self.log_c])
        log_reward = (x * log_coeffs).sum(dim=-1)  # supports batch or per-action input
        return log_reward

class NNReward(nn.Module):
    def __init__(self):
        super(NNReward, self).__init__()

        H = 8
        self.mlp1 = nn.Sequential(
                    nn.Linear(in_features=7, out_features=H),
                    nn.ReLU(),
                    nn.Linear(in_features=H, out_features=4),
        )
        self.mlp2 = nn.Sequential(
                    nn.Linear(4, 1)
                                  )

    def forward(self, x):
        # x (N_clev, 8)
        x = self.mlp1(x)  # (N_clev, H)
        x = x.sum(dim=0)  # (1, H)
        x = self.mlp2(x)  # (1, 1)
        return x.squeeze()  # (, )

def binary_focal_loss_with_logits(log_p, labels, pos_weight=None, gamma=2.0):

    probs = torch.sigmoid(log_p)
    labels = labels.float()
    
    # Compute pt
    pt = torch.where(labels == 1, probs, 1 - probs)
    log_pt = torch.log(pt + 1e-8)

    # Compute the focal loss
    focal_term = (1 - pt) ** gamma
    loss = -focal_term * log_pt

    # Apply alpha
    alpha = 1 - 1 / (pos_weight + 1)  # class imbalance
    if alpha is not None:
        alpha_t = torch.where(labels == 1, alpha, 1 - alpha)
        loss = alpha_t * loss

    return loss.mean()

def demo_loss(demo, reward_fn):
    expert_action = demo[2]
    E_scale, wn_range, tol = demo[4]
    x = []
    for i in range(len(demo[1][1])):  # don't care about no-op
        action_graph = demo[1][0][i]
        action_edge_indices = demo[1][3][i]
        #x.append(reward_classified_lines(action_graph, action_edge_indices, E_scale, wn_range, tol, learn_reward=True)[0])
        x.append(NN_reward_input(action_graph, action_edge_indices, E_scale))
    #x = torch.stack(x)  # (N_cand, N_lines, 6)
    log_p = []
    for line in x:
        #print(line)
        log_p.append(reward_fn(line))
    log_p = torch.stack(log_p)  # (N_cand, )

    labels = torch.zeros(len(log_p))
    labels[expert_action] = 1.0

    pos_weight = len(log_p) - 1  # number of non-expert actions

    loss = F.binary_cross_entropy_with_logits(log_p, labels, pos_weight=torch.tensor(pos_weight))
    #loss = binary_focal_loss_with_logits(log_p, labels, pos_weight=torch.tensor(pos_weight))
    return loss

#%% Training

def train_rw(num_epochs, train_data, test_data):
    seed = 50

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    reward_fn = NNReward() 
    # input_fn = reward_classified_lines

    total_params = sum(p.numel() for p in reward_fn.parameters())
    print(f"Total parameters of reward: {total_params}")

    optimizer = optim.Adam(reward_fn.parameters(), lr=0.001)


    # demos is a list of tuples (s, |a|, a, s')
    #train_data, test_data = train_test_split(demos, test_size=0.2, random_state=42)

    lowest_test_loss = 1e5
    ltl_reward_fn = NNReward() 
    ltl_reward_fn.eval()
    for epoch in range(num_epochs):
        
        reward_fn.train()
        total_loss = 0.0
        for demo in train_data:
            optimizer.zero_grad()
            loss = demo_loss(demo, reward_fn)
            loss.backward()
            optimizer.step()
            total_loss += loss

        total_loss = total_loss/len(train_data)

        test_loss = 0.0
        reward_fn.eval()
        with torch.no_grad():
            for demo in test_data:
                loss = demo_loss(demo, reward_fn)
                test_loss += loss.item()

        test_loss = test_loss/len(test_data)
        
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            print((f'Epoch {epoch:2d}, Loss: {total_loss:.6f}, '
                    f'Test Loss: {test_loss:.6f}'))
        if test_loss < lowest_test_loss:
            lowest_test_loss = test_loss
            ltl_reward_fn.load_state_dict(reward_fn.state_dict())
    
    return ltl_reward_fn

def eval_rw(reward_fn, test_data):
    fractional_rankings = []
    A2_sizes = []
    for demo in test_data:
        expert_action = demo[2]
        expert_energy = demo[1][1][expert_action]
        E_scale, wn_range, tol = demo[4]
        print(f'------- Finding {expert_energy}')
        x = []
        xd = []
        energies = []
        for i in range(len(demo[1][1])):  # don't care about no-op
            action_graph = demo[1][0][i]
            action_edge_indices = demo[1][3][i]
            energies.append(demo[1][1][i])
            x.append(NN_reward_input(action_graph, action_edge_indices, E_scale))
            xd.append(reward_classified_lines(action_graph, action_edge_indices, E_scale, wn_range, tol, learn_reward=True))
        log_p = []
        for line in x:
            log_p.append(reward_fn(line))
        log_p = torch.stack(log_p)  # (N_cand, )
        
        A2_size = len(log_p)
        A2_sizes.append(A2_size)
        D_scores = torch.sigmoid(log_p)
        sorted_indices = torch.argsort(D_scores, descending=True)
        
        for j, i in enumerate(sorted_indices):
            N_cand = f'{xd[i][0][0]:.1f}'.rjust(4)
            prnt_str = f'{energies[i]:.4f}, old rw input ['+N_cand+f', {xd[i][0][1]:.2f}, {xd[i][0][2]:.2f}], old rw: {xd[i][1]:.2f}, NN rw {D_scores[i]:.2f}'
            if i != expert_action:
                print(prnt_str)
            else:
                print(prnt_str + ' expert action!')
                fractional_rankings.append((A2_size - j) / A2_size)
    print('----- Summary -----')
    print(f'Average fractional ranking of expert action: {np.mean(fractional_rankings):.3f}')
    print(f'Median A2 size: {np.median(A2_sizes):.1f}')

# %%
