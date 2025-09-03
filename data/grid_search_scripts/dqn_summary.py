import glob
import pandas as pd

csv_files = glob.glob('./*.csv')  # Use `./path_to_folder/*.csv` for a specific folder

# (trainer.largest_total_episode_reward, total_reward, N_correct, N_found, total_reward_l, N_correct_l, N_found_l, N_correct_ls, N_found_ls)
columns = ['R_lt', 'R_f', 'N_fc', 'N_f', 'R_l', 'N_fc_l', 'N_f_l', 'N_fc_ls', 'N_f_ls']

df_list = []
for file in csv_files:
    df = pd.read_csv(file, header=None)
    if len(df.columns) == 9:
        df.columns = columns
    else:
        df.columns = columns + ['N_fc_id', 'N_fc_l_id', 'N_fc_ls_id']
    df['seed'] = file.split('_')[-1].split('.')[0]  # Extract seed from filename
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

ascii_df = df.to_string(index=False)

R_lt_mean = df['R_lt'].mean()
R_lt_std = df['R_lt'].std() * 2
R_f_mean = df['R_f'].mean()
R_f_std = df['R_f'].std() * 2
frac_N_fc_mean = (df['N_fc']).mean()
frac_N_fc_std = (df['N_fc']).std() * 2
R_l_mean = df['R_l'].mean()
R_l_std = df['R_l'].std() * 2
frac_N_fc_l_mean = (df['N_fc_l']).mean()
frac_N_fc_l_std = (df['N_fc_l']).std() * 2
frac_N_fc_ls_mean = (df['N_fc_ls']).mean()
frac_N_fc_ls_std = (df['N_fc_ls']).std() * 2

# Get params
info_files = glob.glob('./info_*.txt')
with open(info_files[0], 'r') as log:
    param_lines = log.readlines()[:33]

with open('./summary.txt', 'w') as f:
    f.writelines(param_lines)
    f.write('\n') 
    f.write('Summary of all trials recorded by .csv files in this directory\n')
    f.write('\n\n')
    f.write(ascii_df)
    f.write('\n\n')
    f.write('Largest ep reward during training\n')
    f.write(f'{R_lt_mean:.2f} +- {R_lt_std:.2f}\n')
    f.write('Ep reward from deployment of the final agent\n')
    f.write(f'{R_f_mean:.2f} +- {R_f_std:.2f}\n')
    f.write('Fraction of N lev match with human \n')
    f.write(f'{frac_N_fc_mean:.2f} +- {frac_N_fc_std:.2f}\n')
    f.write('Ep reward from deployment of the agent that achieved largest ep reward during training\n')
    f.write(f'{R_l_mean:.2f} +- {R_l_std:.2f}\n')
    f.write('N lev match with human \n')
    f.write(f'{frac_N_fc_l_mean:.2f} +- {frac_N_fc_l_std:.2f}\n')
    f.write('N lev match with human for the final state with the largest ep reward during training\n')
    f.write(f'{frac_N_fc_ls_mean:.2f} +- {frac_N_fc_ls_std:.2f}\n')