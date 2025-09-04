import os
import re
import numpy as np

from collections import defaultdict

# Data storage: {(case, Cp, Cps d, N): list of (a, b, c, d, e)}
results = defaultdict(list)

# Regex to extract from folder name: prefix_Cp0.025_Cps2_d8_N500
folder_re = re.compile(r'(?P<case>.*)_Cp(?P<Cp>[0-9.]+)_Cps(?P<Cps>[0-9.]+)_d(?P<d>[0-9]+)_N(?P<N>[0-9]+)')
# Regex to extract values from lines
line1_re = re.compile(r'Total stepping reward ([0-9.]+), ([\d.]+) correct levs out of [\d.]+')
line2_re = re.compile(r'Max trajectory reward ([0-9.]+), ([\d.]+) correct levs out of ([\d.]+)(?: and \d+ correct IDs)?')

root_dir = '.'  # Change if needed

for dirpath, dirnames, filenames in os.walk(root_dir):
    match = folder_re.search(os.path.basename(dirpath))
    if not match:
        continue

    case = match.group('case')
    Cp = float(match.group('Cp'))
    Cps = float(match.group('Cps'))
    d_val = int(match.group('d'))
    N = int(match.group('N'))
    print(case, Cp, Cps, d_val, N)

    for i in range(999): 
        fname = os.path.join(dirpath, f'log_{i}.txt')
        if not os.path.isfile(fname):
            #print('no', fname)
            continue

        with open(fname) as f:
            lines = f.readlines()
            # if len(lines) < 2:
            #     continue

            m1 = line1_re.search(lines[-25])
            m2 = line2_re.search(lines[-24])
            if m1 and m2:
                a = float(m1.group(1))
                b = float(m1.group(2))
                d = float(m2.group(1))
                e = float(m2.group(2))
                f = float(m2.group(3))
                results[(case, Cp, Cps, d_val, N)].append((a, b, d, e, e/f))

with open('summary.txt', 'w') as out:
    out.write('Sorted by Max T Rw\n')
    header = f"{'Case':>8} {'Cp':>6} {'Cps':>8} {'d':>4} {'N':>5} {'Step Rw':>15} {'Step N_cor':>15} {'Max T Rw':>15} {'Max T N_corr':>15} {'Max N_cor':>10} {'Acc':>15}\n"
    out.write(header)
    out.write('=' * len(header) + '\n')

    rows = []
    for key, vals in results.items():
        data = np.array(vals)
        if data.shape[0] < 2:
            continue
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)

        max_1 = np.max(data[:, 1])
        max_2 = np.max(data[:, 3])
        max_N_cor = np.max([max_1, max_2])

        rows.append((key, means, stds, max_N_cor))

    # Sort by descending d_avg (means[2])
    rows.sort(key=lambda x: x[1][2], reverse=True)

    for (case, Cp, Cps, d_val, N), means, stds, max_N_cor in rows:
        a_str = f'{means[0]:.2f} ± {2*stds[0]:.2f}'
        b_str = f'{means[1]:.1f} ± {2*stds[1]:.1f}'
        d_str = f'{means[2]:.2f} ± {2*stds[2]:.2f}'
        e_str = f'{means[3]:.1f} ± {2*stds[3]:.1f}'
        f_str = f'{means[4]:.2f} ± {2*stds[4]:.2f}'
        max_N_cor_str = f'{max_N_cor:.0f}'

        out.write(f'{case:>8} {Cp:6.3f} {Cps:8.3f} {d_val:4d} {N:5d} {a_str:>15} {b_str:>15} {d_str:>15} {e_str:>15} {max_N_cor_str:>10} {f_str:>15} \n')