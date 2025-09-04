import os
import yaml
import itertools

output_base = 'dqn_gs_nd2_k'
os.makedirs(output_base, exist_ok=True)

grid_searching = True  # whether to do grid search or just multi-seed runs with fixed params

# Grid values
case_list = ['nd2_k']  # to locate base config file and name folder
batch_size_list = [16]
n_step_list = [2]
gamma_list = [0.99]
gat_n_layers_list = [3]
gat_hidden_size_list = [32]
gat_heads_list = [4]
mlp_hidden_size_list = [32]
adam_lr_list = [1e-3]
tau_list = [0.001]
steps_per_train_list = [16]

# Iterate over all combinations
job_id = 0
for case, batch_size, n_step, gamma, gat_n_layers, gat_hidden_size, gat_heads, mlp_hidden_size, adam_lr, tau, steps_per_train in itertools.product(
    
        case_list,
        batch_size_list, 
        n_step_list,
        gamma_list,
        gat_n_layers_list, 
        gat_hidden_size_list,
        gat_heads_list,
        mlp_hidden_size_list,
        adam_lr_list, 
        tau_list,
        steps_per_train_list

    ):

    folder_name = (f'{case}_'
                   f'bs{batch_size}_'
                   f'{n_step}st_'
                   f'g{gamma}_'
                   f'{gat_n_layers}l{gat_hidden_size}x{gat_heads}_{mlp_hidden_size}mlp_'
                   f'lr{adam_lr:.0e}_'
                   f'tau{tau:.0e}_'
                   f'{steps_per_train}spt'
    )
    full_folder_name = os.path.join(output_base, folder_name)

    params = {
    'batch_size': batch_size,
    'n_step': n_step,
    'gamma': gamma,
    'gat_n_layers': gat_n_layers,
    'gat_hidden_size': gat_hidden_size,
    'gat_heads': gat_heads,
    'mlp_hidden_size': mlp_hidden_size,
    'adam_lr': adam_lr,
    'tau': tau,
    'steps_per_train': steps_per_train,
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
        cmd = f'qsub -v {env_vars} dqn_job.sh'
        print(f"Submitting job {job_id}: {cmd}")
        os.system(cmd)
        job_id += 1

    if not grid_searching:
        break