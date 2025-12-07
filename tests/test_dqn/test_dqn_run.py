import tag_dqn

def test_greedy_agent():
    tag_dqn.run_tag_dqn('tests/data/envs/nd3/config.yaml', seed=3) 