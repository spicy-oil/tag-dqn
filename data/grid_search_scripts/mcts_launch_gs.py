import os
import itertools
import yaml

output_base = 'mcts_gs_nd2_k'
os.makedirs(output_base, exist_ok=True)

grid_searching = True  # whether to do grid search or just multi-seed runs with fixed params

# Grid values
case_list = ['nd2_k']
C_p_list = [0.4]
C_ps_list = [2]  # this does nothing because the MCTS Q-values are relative to the avg not an absolute value
depth_list = [4]
N_sim_list = [512]

# Iterate over all combinations
job_id = 0
for case, C_p, C_ps, depth, N_sim in itertools.product(
    
    case_list,
    C_p_list,
    C_ps_list,
    depth_list,
    N_sim_list,

    ):

    folder_name = (f'{case}_'
                   f'Cp{C_p}_'
                   f'Cps{C_ps}_'
                   f'd{depth}_'
                   f'N{N_sim}'
    )
    full_folder_name = os.path.join(output_base, folder_name)

    params = {
    'C_p': C_p,
    'depth': depth,
    'N_sim': N_sim
    }

    # Load base config file to update because many params (e.g. env) are not altered
    with open('./data/envs/' + case + '/config.yaml') as f:
        base_cfg = yaml.safe_load(f)

    cfg = base_cfg.copy()
    if grid_searching:
        cfg['params'].update(params)

    # Save the modified config file in the output folder
    config_path = os.path.join(full_folder_name, f'config.yaml')
    os.makedirs(full_folder_name, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.safe_dump(cfg, f)

    for seed in range(5):  # Run multiple seeds per config
        
        env_vars = f'CASE={case},CONFIG={config_path},SEED={seed},FOLDER_NAME={full_folder_name}'
        cmd = f'qsub -v {env_vars} mcts_job.sh'
        print(f"Submitting job {job_id}: {cmd}")
        os.system(cmd)
        job_id += 1

    if not grid_searching:
        break


