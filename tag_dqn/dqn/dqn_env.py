import torch
import os
import hashlib
import numpy as np

from importlib import resources
from torch_geometric.data import Data
from cachetools import LRUCache
from .dqn_data_proc import fit_pop
from .dqn_reward import reward_classified_lines, NN_reward_input, NNReward
from . import levham
#%%
class Env():
    def __init__(self, data, lev_name, J, fixed_lev_indices, fixed_lev_values, 
                 ep_length=500, epsilon=0.1, z=None, wn_range=.25, tol=5e-5, int_tol=1.,
                 A2_max=256,
                 reward_params=None, NN_ham=False):
        '''
        graph is from torch.geometric, it does not change shape at the moment, only its attributes change
        '''
        self.ep_length = ep_length  # number of steps to simulate in an episode
        self.timestep = torch.scalar_tensor(0)
        self.epsilon = epsilon  # exploration rate
        self.lev_name = lev_name  # for printing
        self.J = J  # angular momentum for printing
        self.z = z  # fixed support atoms axis for dist RL

        # Define graph state 
        self.init_graph = Data(x=data[0].x.clone(), 
                               edge_index=data[0].edge_index.clone(), 
                               edge_attr=data[0].edge_attr.clone()
                              )
        self.graph = Data(x=data[0].x.clone(), 
                               edge_index=data[0].edge_index.clone(), 
                               edge_attr=data[0].edge_attr.clone()
                              )
        self.linelist = data[1].clone()  # shape [N_lines, 3]
        self.action_type = 0  # start with selecting a level to find, 1 is for selecting candidates
        self.level_to_find = torch.tensor([-1])  # start with no level to find selected
        self.levels_found = 0

        # Env params
        self.E_scale = data[2]  # wavenumbers are normalised by max wn
        self.wn_range = wn_range  # kK
        self.tol = tol  # kK
        self.int_tol = int_tol  # log_10(int)

        # Define fixed levels
        self.fixed_lev_indices = fixed_lev_indices
        self.fixed_lev_values = fixed_lev_values  / self.E_scale  # kK / self.E_scale

        # Save precomputed A2 spaces
        self.A2_cache = LRUCache(maxsize=1000)  # Cache for A2 spaces
        self.A2_cache_saves = 0  # number of times A2 space was already computed
        
        # Track MDP complexity
        self.A2_max = A2_max
        self.A1_sizes = []
        self.A2_sizes = []

        # NN reward
        self.reward_params = reward_params
        self.reward = NNReward()
        if reward_params is not None:
            self.reward.load_state_dict(torch.load(reward_params, weights_only=True)) 
        else:
            # Using package data
            default_rw_path = 'tag_dqn.pkg_data'
            with resources.open_binary(default_rw_path, 'reward.pth') as f:
                self.reward.load_state_dict(torch.load(f, weights_only=True)) 


        # NN LEVHAM component
        if NN_ham:
            self.NN_ham = levham.NNHam()
            self.NN_ham.load_state_dict(torch.load('./dqn/nnham.pth', weights_only=True))  # assuming running from ../
        else:
            self.NN_ham = None


        self._print_decisions = False

        self.reset()

    def reset(self):
        '''Reset environment to initial graph state.'''
        self.graph = Data(x=self.init_graph.x.clone(), 
                               edge_index=self.init_graph.edge_index.clone(), 
                               edge_attr=self.init_graph.edge_attr.clone()
                              )
        self.linelist_mapping = torch.zeros(self.graph.edge_attr.size(0), dtype=torch.long) - 1 # mapping from graph edge_attr to linelist
        self.action_type = 0
        self.timestep = torch.scalar_tensor(0)
        self.level_to_find = torch.tensor([-1])
        self.levels_found = 0
        self.refund = 0
        self.lev_blacklist = []  # no-op was chosen for these levels...
        self.action_space = self.A1_space(self.graph)  # always start with action 1
        self.A1_sizes.append(len(self.action_space))  # track action space size for action 1
        self.state = self.get_state()
        return self.state

    def step(self, Q_values):
        '''
        Perform the action using Q_values output by the DQN 
        '''
        # Whether an episode ends---------------------------------------------------------------------------
        self.timestep += 1 # steps completed in the episode
        done_flag = self._check_termination()

        reward = torch.tensor(0, dtype=torch.float64)

        if self.z.size(0) > 1: # If are Q distributions
            Q_values = (Q_values * self.z).sum(-1) # mean Q values of the Q value distributions

        action_index = self.epsilon_greedy(Q_values, self.epsilon)

        if self.action_type == 0: # if selecting a level to find-------------------------------------------

            valid_node_indices = self.action_space

            self.level_to_find = valid_node_indices[action_index].unsqueeze(0) # unsqueeze for forward pass in DQN

            # Change one hot encoding of selection for the selected level
            # [E_calc, E_obs, J, even, odd, known, unknown, selected, unselected]
            self.graph.x[self.level_to_find[0]][[4, 5]] = torch.tensor([1., 0.], dtype=torch.float64)

            # Reward for selecting the level
            cline_index = torch.where(self.level_to_find[0] == self.graph.edge_index[0])  # undirected so compare with only source is fine
            clev_index = self.graph.edge_index[1][cline_index]
            known_clev_boo = self.graph.x[clev_index][:, 2] == 1
            expected_lines = self.graph.edge_attr[cline_index[0][known_clev_boo]]
            snr_calc = expected_lines[:, 6] * 5  # so that snr_calc is in log10 units
            tot_exp_snr = torch.log10(torch.sum(10 ** snr_calc))
            self.refund = tot_exp_snr
            reward = reward + tot_exp_snr  # award strong lines searching for a level
            # If all lines expected_lines are found 
            # tot_exp_snr is the base reward for action 2, which is discounted
            # so net reward every two steps is always < 0
        
            self.action_type = 1
            self.action_space = self.A2_space()  # action 2 space is computed here 
            self.A2_sizes.append(len(self.action_space[0]))  # track action space size for action 2

            if len(self.action_space[0]) == 1 and self.level_to_find[0].item() not in self.lev_blacklist:
                self.lev_blacklist.append(self.level_to_find[0].item())  # don't try this level again for the rest of the episode
                #print(f'No candidates found for lev id {self.level_to_find.item()} ' 
                #      + self.lev_name[self.level_to_find[0]] 
                #      + ', this level will not be attempted again for the rest of the episode')

            if self._print_decisions:
                print('----------------------------------') # separator for readability
                print(f'|A1| = {len(valid_node_indices):>3.0f},   |A2| = {len(self.action_space[0]):>3.0f}')
                print('Action 1 - lev', self.level_to_find.item(), self.lev_name[self.level_to_find.item()])
                print(f'Reward: {reward.item()}')

        else:  # if selecting a candidate for the level to find----------------
            reward = reward - self.refund
           
            cand_graphs, cand_energies, cand_wn_obs_indices, cand_edge_indices = self.action_space # All list lengths are one smaller than Q_values length because of no-op
            
            N_cand = len(cand_graphs)
            # For evaluation usually
            if self._print_decisions:
                Q_values_np = Q_values.detach().numpy()
                cand_energies = [f'{energy:.4f}' for energy in cand_energies]
                cand_rewards = []
                for g, i in zip(cand_graphs, cand_edge_indices):  # this does not loop no-op graph because len(cand_edge_indices) = len(cand_graphs) - 1
                    cand_rewards.append(reward + self.refund * self.get_NN_reward(g, i))
                    # old non-NN rw function
                    # cand_rewards.append(reward + reward_classified_lines(g, i, self.E_scale, self.wn_range, self.tol)) # reward for classified lines
                cand_energies = cand_energies + ['no-op     ']
                cand_rewards = cand_rewards + [reward]
                sorted_pairs = sorted(zip(Q_values_np, cand_energies, cand_rewards), reverse=True, key=lambda x: x[0])
                print(f'Action 2 - {N_cand} candidates:')
                for q, energy, rew in sorted_pairs:
                    print(f'Q: {q:.8f}, Energy: {energy}, Reward: {rew.item():.4f}')
    

            # Update graph
            cand_graph = cand_graphs[action_index]
            self.graph = Data(x=cand_graph.x.clone(), 
                        edge_index=cand_graph.edge_index.clone(), 
                        edge_attr=cand_graph.edge_attr.clone()
                        )

            # If choosing final option/no candidates
            if action_index == N_cand - 1: 
                reward = reward
                if self._print_decisions:
                    print(f'Chose no-op')
                    print(f'Reward: {reward.item():.4f}')
            # If chose a LEVHAM candidate
            else:              
                # Update reward
                cand_edge_indices = cand_edge_indices[action_index] # tensor of edge_attr indices to consider reward from
                reward = reward + self.refund * self.get_NN_reward(self.graph, cand_edge_indices)
                # old non-NN rw function
                # reward = reward + reward_classified_lines(self.graph, cand_edge_indices, self.E_scale, self.wn_range, self.tol) # give reward

                # Update linelist mapping
                cand_wn_obs_index = cand_wn_obs_indices[action_index]  # tensor of line list indices for the classified lines
                self.linelist_mapping[cand_edge_indices] = cand_wn_obs_index  # Update linelist mapping
                # Update also the opposite direction edges because undirected graph
                shift = self.graph.edge_attr.size(0) // 2
                sign = (cand_edge_indices < self.graph.edge_attr.size(0) // 2).long() * 2 - 1
                self.linelist_mapping[cand_edge_indices + sign * shift] = cand_wn_obs_index 

                # Update I_calc for all lines because new known lines
                # self.graph = self.update_I_calc(self.graph, self.E_scale) 

                if self._print_decisions:
                    cand_energy = cand_energies[action_index]
                    print(f'Chose {cand_energy}')
                    print(f'Reward: {reward.item():.4f}')
                    print('----------------------------------') # separator for readability

                self.levels_found = self.levels_found + 1
            
            self.action_type = 0 
            self.action_space = self.A1_space(self.graph)  # back to action 1
            self.A1_sizes.append(len(self.action_space))  # track action space size for action 1

            if len(self.action_space) == 0:
                #print('No more valid nodes, episode terminated')
                done_flag = torch.tensor(1)  # done

        reward = reward / 10

        self.state = self.get_state() 

        return self.state, action_index, reward, done_flag

    def get_state(self):
        state = { 
                'graph'            : self.graph, 
                'linelist_mapping' : self.linelist_mapping,
                'action_type'      : self.action_type, # next action type
                'action_space'     : self.action_space, # only deducible from a forward pass by Agent, to be assigned after each pass
                'level_to_find'    : self.level_to_find, # action_type 0 - the level that was searched, 1 - the level to search for
                'steps_till_done'  : [(self.ep_length - self.timestep), self.ep_length],  # normalised steps till done
                'time_step'        : self.timestep, # steps completed in the episode
                'levels_found'     : self.levels_found,
                'refund'           : self.refund,
                'lev_blacklist'    : self.lev_blacklist
        } 
        return state

    def epsilon_greedy(self, Q_values, epsilon):
        '''
        Selects an action using epsilon-greedy policy.

        Parameters:
        - Q_values (torch.Tensor): Tensor of shape [N] containing Q-values.
        - epsilon (float): Exploration rate (0 ≤ epsilon ≤ 1).

        Returns:
        - int: Selected action index.
        '''
        if torch.rand(1).item() < epsilon:  # With probability epsilon, explore
            self.exploring = True
            return torch.randint(0, Q_values.shape[0], (1,)).item()  # Random action
        else:  # Otherwise, exploit (greedy action)
            self.exploring = False
            return torch.argmax(Q_values).item()  # Action with highest Q-value

    def A1_space(self, graph):
        '''
        Action space determination for Action 1
        Get indices of nodes with at least TWO edges to distinct nodes in known_indices.
        '''
        # Identify known levels (nodes with feature[1] = 1)
        # [E_calc, E_obs, J, even, odd, known, unknown, selected, unselected]
        known_levels = graph.x[:, 2] == 1  # Boolean mask for known levels
        known_indices = torch.where(known_levels)[0]  # Indices of known nodes
        
        # Select edges where the source node is in known_indices
        mask = torch.isin(graph.edge_index[0], known_indices)
        known_nodes_connected = graph.edge_index[0, mask]
        neighbour_nodes = graph.edge_index[1, mask]
        
        # Create a mapping from neighbour -> known node connections
        neighbour_to_known = torch.stack([neighbour_nodes, known_nodes_connected], dim=1)
        
        # Use unique to count distinct known node connections per neighbour
        unique_pairs = torch.unique(neighbour_to_known, dim=0)
        
        # Count occurrences of each neighbour in the unique pairs
        neighbour_counts = torch.bincount(unique_pairs[:, 0], minlength=graph.x.size(0))  # Count per node
        neighbour_counts[known_indices] = 0 # exclude known nodes for the next step

        # Get nodes connected to at least two distinct known nodes
        valid_node_indices = torch.where(neighbour_counts >= 2)[0]

        # Filter out nodes that are blacklisted
        valid_node_indices = valid_node_indices[~torch.isin(valid_node_indices, torch.tensor(self.lev_blacklist, dtype=torch.long))]
    
        #if len(valid_node_indices) == 0:
            #print('Warning: no valid nodes found for action 1')

        return valid_node_indices

    def A2_space_compute(self, learning=False):
        '''
        Action space determination for Action 2
        Get all possible new graph states using LEVHAM + LOPT filtering
        '''
        self.levham_lev_indices, self.levham_lev_energies, self.levham_lev_lines = levham.get_levham_inputs(self.level_to_find, self.graph)
        
        temp = levham.LEVHAM(self.graph, self.linelist, self.E_scale, self.level_to_find, self.linelist_mapping,
                            self.levham_lev_indices, self.levham_lev_energies, self.levham_lev_lines, 
                            self.fixed_lev_indices, self.fixed_lev_values,
                            self.wn_range, self.tol, self.int_tol,
                            learning, self.NN_ham
                )
        if not learning:
            cand_graphs, cand_energies, cand_wn_obs_indices, cand_edge_indices = temp

            # If A2 too big, no-op becomes the only option
            if len(cand_graphs) > self.A2_max:
                cand_graphs, cand_energies, cand_wn_obs_indices, cand_edge_indices = ([],[],[],[])

            # Add no-op graph to candidates
            no_op_graph = Data(x=self.graph.x.clone(),
                                edge_index=self.graph.edge_index.clone(), 
                                edge_attr=self.graph.edge_attr.clone())
            # [E_calc, E_obs, J, even, odd, known, unknown, selected, unselected]
            no_op_graph.x[self.level_to_find, [4, 5]] = torch.tensor([0., 1.], dtype=torch.float64)  # No longer selected
            cand_graphs.append(no_op_graph)  # Add no-op graph to candidates
            
            return cand_graphs, cand_energies, cand_wn_obs_indices, cand_edge_indices
        
        else:
            cand_graphs, cand_energies, cand_wn_obs_indices, cand_edge_indices, cand_prelopt = temp
            # don't care about no-op in reward and levham learning
            return cand_graphs, cand_energies, cand_wn_obs_indices, cand_edge_indices, cand_prelopt

    def A2_space(self):
        '''
        Check if A2 space is already computed, if not compute it
        '''
        # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
        # Hash the wn_obs and lev_to_find to create a unique key
        wn_obs = np.round(self.graph.edge_attr[:, 1].numpy(), decimals=9)  # round to 10 decimal places
        lev_to_find = self.level_to_find.item()  # integer
        byte_repr = np.append(wn_obs, lev_to_find).tobytes()  # convert to bytes
        hash_key = hashlib.sha256(byte_repr).hexdigest()  # create a hash key
        # Check if the hash key is already in the cache
        if hash_key in self.A2_cache:
            # If it is, return the cached value
            self.A2_cache_saves += 1
            return self.A2_cache[hash_key]
        else:
            # If not, compute the A2 space
            new_A2_space = self.A2_space_compute()
            self.A2_cache[hash_key] = new_A2_space
            # Return the computed A2 space
            return new_A2_space

    def state_to_attr(self):
        '''
        Copy graph state dictionary values to graph attributes
        self.state = { 'graph'     : self.graph, 
                'linelist_mapping'  : self.linelist_mapping,
                'action_type'   : self.action_type, # next action type
                'action_space' : self.action_space, # only deducible from a forward pass by Agent, to be assigned after each pass
                'level_to_find' : self.level_to_find, # action_type 0 - the level that was searched, 1 - the level to search for
                'steps_till_done' : [(self.ep_length - self.timestep), self.ep_length],  # normalised steps till done
                'time_step'    : self.timestep, # steps completed in the episode
                'levels_found' : self.levels_found}
        '''
        self.graph = self.state['graph']
        self.linelist_mapping = self.state['linelist_mapping']
        self.action_type = self.state['action_type']
        self.action_space = self.state['action_space']
        self.level_to_find = self.state['level_to_find']
        self.steps_till_done = self.state['steps_till_done']
        self.timestep = self.state['time_step']
        self.levels_found = self.state['levels_found']
        self.refund = self.state['refund']
        self.lev_blacklist = self.state['lev_blacklist']  # no-op was chosen for these levels...

    def update_I_calc(self, graph, E_scale):
        '''
        Update I_calc for all lines in the graph
        '''
        # Update I_calc because new levels and lines are found
        m, c = fit_pop(graph, E_scale, plot=False)  # E_scale only for plotting, so x axis is kK / E_Scale
        # upper_level = torch.maximum(pred_E, levham_lev_energies[i])
        # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
        ul_idx = graph.edge_index[0][:graph.edge_index.size(1) // 2]  # first half is upper level because undirected graph
        ul = graph.x[ul_idx]  # upper levels
        # [E_calc, E_obs, J, even, odd, known, unknown, selected, unselected]
        unknown_ul_boo = ul[:, 2] != 1
        ul_E_obs = ul[:, 1]  # E_obs of upper levels
        ul_E_calc = ul[:, 0]  # E_calc of upper levels
        ul_E_calc = ul_E_calc * unknown_ul_boo  # zero if known
        ul_E = ul_E_obs + ul_E_calc  # E of upper levels, obs if known, calc if unknown
        ul_E = torch.cat((ul_E, ul_E), dim=0)  # double the size of the tensor for undirected graph
        gA_calc = graph.edge_attr[:, 5] * 10  # log10(gA_calc)
        I_calc = gA_calc + m * ul_E + c  # log10(I_calc)
        graph.edge_attr[:, 3] = I_calc / 5  # update I_calc
        return graph

    def _check_termination(self):
        '''Define termination condition (e.g., max steps reached).'''
        if self.timestep == self.ep_length:
            return torch.tensor(1)
        else:
            return torch.tensor(0)
    
    def _get_known_levs(self):
        # [E_calc, E_obs, J, even, odd, known, unknown, selected, unselected]
        boo = self.graph.x[:, 2] > 0
        levs = self.graph.x[boo] * self.E_scale
        levs = levs[:, 1].numpy()
        lev_names = self.lev_name[boo]
        return lev_names, levs
    
    def _get_pop(self, plot=True):
        m, c, x, y = fit_pop(self.graph, self.E_scale, plot=plot)
        return m, c, x, y

    def get_NN_reward(self, graph, cand_edge_indices):
        out = self.reward(NN_reward_input(graph, cand_edge_indices, self.E_scale))
        return torch.sigmoid(out)  # [0, 1]