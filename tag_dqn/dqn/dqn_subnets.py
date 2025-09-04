'''
TAG-DQN component networks: noisy nets, GAT, MLPs
'''

import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation

class GraphEmbedder(nn.Module):
    '''
    Graph Attention State embedder
    '''
    def __init__(self, hidden_size=32, heads=8, n_layers=2, diff_scale=None):
        super(GraphEmbedder, self).__init__()

        self.diff_scale = diff_scale
        node_dim = 6   # [E_calc, E_obs, J, even, odd, known, unknown, selected, unselected]
        edge_dim = 11  # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
        
        self.n_layers = n_layers

        self.gat_layers = nn.ModuleList()
        #self.norm_layers = nn.ModuleList()

        # First GAT layer
        self.gat_layers.append(GATv2Conv(in_channels=node_dim,
                            out_channels=hidden_size, 
                            heads=heads,
                            concat=True,
                            edge_dim=edge_dim
                            ))
        #self.norm_layers.append(nn.LayerNorm(hidden_size * heads))

        # Remaining layers
        for _ in range(1, n_layers):
            self.gat_layers.append(GATv2Conv(hidden_size * heads, 
                                           hidden_size, heads=heads, 
                                           concat=True, edge_dim=None))
            #self.norm_layers.append(nn.LayerNorm(hidden_size * heads))

        self.elu = nn.ELU()

    def forward(self, x, edge_index, edge_attr):

        x = x.float()  # 32, is copy
        edge_attr = edge_attr.float()  # 32, is copy

        # Make known E_obs and wn_obs into diff and scale them
        # [E_calc, E_obs, J, even, odd, known, unknown, selected, unselected]
        idx = x[:, 2] == 1
        x[:, 1][idx] = (x[:, 1][idx] - x[:, 0][idx]) / self.diff_scale  # E_obs - E_calc
        # [wn_calc, wn_obs, wn_obs_unc, intensity_calc, intensity_obs, gA_calc, snr_calc, snr_obs, line_den, known, unknown] 
        idx = edge_attr[:, -2] == 1
        edge_attr[:, 1][idx] = (edge_attr[:, 1][idx] - edge_attr[:, 0][idx]) / self.diff_scale  # wn_obs - wn_calc

        for i in range(self.n_layers):
            # no edge_attr unless first layer
            if i > 0:  
                edge_attr = None
            x = self.gat_layers[i](x, edge_index, edge_attr)
            # layernorm and activation if not final layer
            if i < self.n_layers - 1:
                #x = self.norm_layers[i](x)
                x = self.elu(x)

        return x  # Shape [N_lev, output_dim]

class MLP(nn.Module):
    '''
    Three distinct networks for:
        Q value of selecting a level to search for using node embeddings [N_node, 2H + 1]
        Q value of selecting a candidate level for the selected level    [N_mod, H + 1]
        State value of the current state                                 [1, H + 3]
    H = hidden_size * heads * heads
    Set N_atoms = 1 if not using distributional RL
    '''
    def __init__(self, hidden_size=32, heads=8, mlp_hidden_size=256, N_atoms=51, noisy=True, sigma_init=0.5, type=None):
        super(MLP, self).__init__()

        node_embedding_dim = hidden_size * heads
        if type == 'node':
            embedding_dim = 2 * node_embedding_dim + 1 # 2x from concat with global embed, plus 1 from steps till done
        elif type == 'mod':
            embedding_dim = node_embedding_dim + 1 # plus 1 from steps till done
        else: # type == 'state'
            embedding_dim = node_embedding_dim + 1 # plus 1 from steps till done

        hidden_dim = mlp_hidden_size # max(embedding_dim, N_atoms) # hidden dim at least N_atoms

        if noisy: # If using noisy nets for exploration
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                NoisyLinear(hidden_dim, N_atoms, sigma_init)

                # NoisyLinear(embedding_dim, hidden_dim, sigma_init),
                # nn.PReLU(),
                # NoisyLinear(hidden_dim, hidden_dim, sigma_init),
                # nn.PReLU(),
                # NoisyLinear(hidden_dim, N_atoms, sigma_init)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, N_atoms)
            )

    def forward(self, x):
        return self.mlp(x)  # Shape [(N_node, N_mod, or 1), N_atoms]

    def reset_noise(self):
        '''
        Resets the noise for all NoisyLinear layers in this module.
        '''
        for layer in self.mlp:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

class BinaryGateSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score):
        return (score > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # straight-through

class AEN(nn.Module):
    '''
    Action 2 elimination network for more efficient learning
        Aggregated graph node embedding is input, like value and advantage MLPs
    '''
    def __init__(self, hidden_size=32, heads=4, mlp_hidden_size=256, type=None):
        super(AEN, self).__init__()

        node_embedding_dim = hidden_size * heads  # H + 1 from steps_till_done

        if type == 'node':
            embedding_dim = 2 * node_embedding_dim + 1 # 2x from concat with global embed, plus 1 from steps till done
        elif type == 'mod':
            embedding_dim = node_embedding_dim + 1 # plus 1 from steps till done


        hidden_dim = mlp_hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x (|A2|, H)
        x = self.mlp(x)  # (|A2|, 1)
        x = torch.sigmoid(x * 5)  # (|A2|, 1)
        return x.squeeze(-1)  # (|A2|,)

class GraphAggr(nn.Module):
    '''
    Network for aggregating node embeddings to a single state embedding H = hidden_size * heads * heads
    '''
    def __init__(self, hidden_size=32, heads=2):
        super(GraphAggr, self).__init__()

        embedding_dim = hidden_size * heads * heads
        
        # This calculates attention weights
        self.gate_nn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.PReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.PReLU(),
            nn.Linear(embedding_dim, 1)
        )

        self.node_agg = AttentionalAggregation(self.gate_nn)

    def forward(self, x):
        '''
        Weighted sum of node embeddings, weights are from the gate_nn
        '''
        #x = torch.concat([x, steps_till_done, one_hot_atype], dim=-1) # [1, lev_embedding_dim + 3]
        return self.node_agg(x) # [1, lev_embedding_dim]

class NoisyLinear(nn.Module):
    '''
    Fortunato et al. 2018
    Is not factorised, meaning noisy is sampled for each entry in the weight and bias matrices.
    '''
    def __init__(self, in_features, out_features, sigma_init=0.5):
        '''
        0.5 for sigma_init from Rainbow paper I think
        '''
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias parameters
        self.weight_mu = nn.Parameter(torch.rand(out_features, in_features, dtype=torch.float32))
        self.bias_mu = nn.Parameter(torch.rand(out_features, dtype=torch.float32))

        # Parameters for noise (sigma)
        self.weight_sigma = nn.Parameter(torch.rand(out_features, in_features, dtype=torch.float32))
        self.bias_sigma = nn.Parameter(torch.rand(out_features, dtype=torch.float32))

        # Noise buffer, not trainable parameters
        weight_epsilon = torch.rand(out_features, in_features, dtype=torch.float32)
        weight_epsilon.requires_grad = False
        self.register_buffer('weight_epsilon', weight_epsilon)

        bias_epsilon = torch.rand(out_features, dtype=torch.float32)
        bias_epsilon.requires_grad = False
        self.register_buffer('bias_epsilon', bias_epsilon)

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        '''Initialize parameters when model is created.'''
        mu_range = (1 / self.in_features)**0.5 
        sigma_init = self.sigma_init * (1 / self.in_features)**0.5 
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

    def reset_noise(self):
        # '''Sample new noise values for each forward pass.'''
        # self.weight_epsilon.normal_()
        # self.bias_epsilon.normal_()
        '''Factorised Gaussian noise: Sample noise per input/output dimension.'''
        # Sample noise vectors
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        # Outer product for weight noise
        self.weight_epsilon = epsilon_out.ger(epsilon_in)  # (out, in)
        self.bias_epsilon = epsilon_out                    # (out,)

    def forward(self, x):
        '''Compute Noisy Linear transformation.'''
        if self.training: # this is when we set model.train() for grad descent
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)