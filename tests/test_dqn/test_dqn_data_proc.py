from tag_dqn.dqn import dqn_data_proc


def test_input_to_graph():
    config_file = 'data/envs/nd3/config.yaml'
    preproc_in = dqn_data_proc.get_preproc_input(config_file, float_levs=False)
    init_graph, init_linelist, E_scale = dqn_data_proc.preproc(preproc_in)
    print(init_graph)
    print(init_linelist.shape)