#%%
'''
TAG-DQN agent
'''

import torch
import torch.nn as nn
from . import dqn_subnets

class Agent(nn.Module):
    def __init__(self, hidden_size=32, heads=8, gat_n_layers=3, mlp_hidden_size=256,
                 noisy=True, sigma_init=0.5, z=None, diff_scale=None, aen=False, duel=True):
        super(Agent, self).__init__()

        # Atoms for dist RL
        self.N_atoms = len(z)
        self.z = z

        # GNN for level embeddings/context
        self.graph_embedder = dqn_subnets.GraphEmbedder(hidden_size, heads, gat_n_layers, diff_scale)
        #self.graph_aggr = dqn_subnets.GraphAggr(hidden_size, heads)

        # Q value net for selecting a level to find - action 1
        self.A_node = dqn_subnets.MLP(hidden_size, heads, mlp_hidden_size, self.N_atoms, noisy, sigma_init, type='node')
        
        # Q value net for selecting a candidate level - action 2
        self.V_mod = dqn_subnets.MLP(hidden_size, heads, mlp_hidden_size, self.N_atoms, noisy, sigma_init, type='mod')
        
        # State value net for Duelling
        self.duel = duel
        if self.duel:
            self.V1_state = dqn_subnets.MLP(hidden_size, heads, mlp_hidden_size, self.N_atoms, noisy, sigma_init, type='state')
            self.V2_state = dqn_subnets.MLP(hidden_size, heads, mlp_hidden_size, self.N_atoms, noisy, sigma_init, type='state')
        
        # Action elimination net
        self.aen = aen
        if aen:
            self.AEN_1 = dqn_subnets.AEN(hidden_size, heads, mlp_hidden_size, type='node')
            self.AER_mean = []
            self.AER_std = []
            #self.AEN_2 = dqn_subnets.AEN(hidden_size, heads, mlp_hidden_size, type='mod')

    def forward(self, state):
        '''
        Forward pass for simulation
        '''
        self.graph = state['graph']
        self.action_type = state['action_type']
        self.action_space = state['action_space']
        if len(self.action_space) == 0:  # only for action 1
            return torch.tensor([0.]), torch.tensor([], dtype=torch.int64)
        self.lev_selection = state['level_to_find']
        self.steps_till_done, self.ep_length = state['steps_till_done'] # torch scalar
    
        self.graph_embedding = self.graph_embedder(self.graph.x, self.graph.edge_index, self.graph.edge_attr)  
        self.graph_embedding_agg = self.graph_embedding.mean(dim=-2, keepdim=True) # [1, H]

        # Misc
        steps_till_done_broadcast = self.steps_till_done.unsqueeze(0).unsqueeze(0) / self.ep_length # scalar to [1, 1]
        self.graph_embedding_agg = torch.concat([self.graph_embedding_agg, steps_till_done_broadcast], dim=-1) # [1, H + 1]

        # Action 1 - Select a level to find-------------------------------------------------------------
        if self.action_type == 0: # looking to select a level to search for
            
            # Actions
            valid_node_indices = self.action_space
            known_neighbour_embeds = self.graph_embedding[valid_node_indices]  # [N_known_neighb, H]
            # Let Q net know of global emebdding and how many steps till end of episode for each valid level to search for
            # [N_known_neighb, 2H + 1]
            known_neighbour_embeds = torch.cat([known_neighbour_embeds, 
                                                self.graph_embedding_agg.expand(known_neighbour_embeds.size(0), -1)], dim=1)  

            # Estimate advantage values
            A_values = self.A_node(known_neighbour_embeds)

            # Duel
            if self.duel:
                # State value using V1
                state_value = self.V1_state(self.graph_embedding_agg) # [1, N_atoms]
                if self.N_atoms == 1: # If Q values
                    A_values = A_values.flatten() # [N_known_neighb]
                    Q_values = state_value.flatten() + A_values - A_values.mean()
                else: # If distributions
                    Q_values = state_value + A_values - A_values.mean(dim=0, keepdim=True) # Q_val distributions [N_known_neighb, N_atoms]
                    Q_values = torch.softmax(Q_values, dim=-1) # normalise along atoms axis for prob. dist. [N_known_neighb, N_atoms]
            else:  # No dist RL
                Q_values = A_values.flatten()

            # Action elimination
            if self.aen:
                soft_mask = self.AEN_1(known_neighbour_embeds)  # [N_known_neighb, ]
                self.AER_mean.append(torch.mean(soft_mask.detach()))
                self.AER_std.append(torch.std(soft_mask.detach()))
                Q_values = Q_values * soft_mask

            return Q_values, valid_node_indices

        # Action 2 - Select a candidate level-----------------------------------------------------------
        if self.action_type == 1: # looking to select a candidate
            cand_graphs, cand_energies, cand_wn_obs_indices, cand_edge_indices = self.action_space

            # Loop over candidates and embed them
            N_cand = len(cand_graphs)
            cand_graphs_embedding_list = []  # Store embeddings for all batches
            for graph in cand_graphs:
                #print(f'Action 2 looping cands', graph.x[self.lev_selection[0]][[7, 8]])
                #graph_embedding = self.graph_embedder(graph.x.float(), graph.edge_index, graph.edge_attr.float())  # float32 for GNN
                graph_embedding = self.graph_embedder(graph.x, graph.edge_index, graph.edge_attr) 
                cand_graphs_embedding_list.append(graph_embedding)
            # Concatenate all processed embeddings
            cand_graphs_embedding = torch.cat(cand_graphs_embedding_list, dim=0)  # [N_cand * N_lev, H]
            # Split back into individual candidate graphs [N_cand, N_lev, H]
            cand_graphs_embedding = cand_graphs_embedding.view(N_cand, self.graph_embedding.shape[0], cand_graphs_embedding.shape[-1]) 
            # Get the embeddings for the level to search for
            # cand_lev_embeddings = cand_graphs_embedding[:, self.lev_selection[0], :]  # [N_cand, H]
            # Aggregate candidate embeddings [N_cand, H]
            cand_graphs_embedding = cand_graphs_embedding.mean(dim=-2, keepdim=False)  # [N_cand, H]
            # Let Q net know how many steps till end of episode for each candidate level, -1 because s'
            cand_graphs_embedding = torch.cat([#cand_lev_embeddings, 
                                               cand_graphs_embedding, 
                                               steps_till_done_broadcast.expand(N_cand, -1) - 1 / self.ep_length],
                                               dim=1) # [N_cand, H + 1]

            # # Action elimination
            # if self.aen:
            #     mask = self.AEN_2(cand_graphs_embedding[:-1])  # [N_cand - 1, ], excluding no-op
            #     cand_graphs_embedding = torch.cat([cand_graphs_embedding[:-1][mask], cand_graphs_embedding[-1].unsqueeze(0)], dim=0)  # [N_cand, H + 1], include no-op
            #     #N_cand = len(cand_graphs_embedding)  # N_cand is reduced by AEN

            # Evaluate the candidates
            next_state_values = self.V_mod(cand_graphs_embedding) # V(s') [N_cand, 1] has grad

            if self.duel:
                # State value using V2
                state_value = self.V2_state(self.graph_embedding_agg) # [1, N_atoms]
                # Duel
                if self.N_atoms == 1: # If Q values
                    state_value = state_value.flatten()
                    next_state_values = next_state_values.flatten() # [N_cand + 1]
                    A_values = next_state_values - state_value  # advantages
                    Q_values = state_value + A_values - A_values.mean() # Q(s, a) = V(s) + A(s') - A(s').mean()
                else: # If distributions
                    A_values = next_state_values - state_value
                    Q_values = state_value + A_values - A_values.mean(dim=0, keepdim=True) # Q_val distributions [N_known_neighb, N_atoms]
                    Q_values = torch.softmax(Q_values, dim=-1) # normalise along atoms axis for prob. dist.
            else:
                Q_values = next_state_values.flatten()

            return Q_values, (cand_graphs, cand_energies, cand_wn_obs_indices, cand_edge_indices)

# %%
