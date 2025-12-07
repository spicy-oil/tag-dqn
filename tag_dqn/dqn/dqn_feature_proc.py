class FeatureProcessor:
    def __init__(self):
        self.node_feature_names = ['E_calc', 'E_obs', 'known', 'unknown', 'selected', 'unselected']
        self.node_feature_map = {name: i for i, name in enumerate(self.node_feature_names)}

        self.edge_feature_names = ['wn_calc', 'wn_obs', 'wn_obs_unc', 'I_calc', 'I_obs', 
                                   'gA_calc', 'snr_calc', 'snr_obs', 'line_den', 'known', 'unknown'] 
        self.edge_feature_map = {name: i for i, name in enumerate(self.edge_feature_names)}

    def slice_node_features(self, tensor, selected_features):
        '''tensor of shape [num_nodes, num_node_features]'''
        idx = [self.node_feature_map[f] for f in selected_features]
        return tensor[:, idx]

    def slice_edge_features(self, tensor, selected_features):
        '''tensor of shape [num_edges, num_edge_features]'''
        idx = [self.edge_feature_map[f] for f in selected_features]
        return tensor[:, idx]