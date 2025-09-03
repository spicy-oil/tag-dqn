Each environment requires the following 1D arrays (grouped into separate csv files):

----(line_list.csv)----
wn_obs            - float64 (1000 cm-1)
wn_obs_unc        - float64 (1000 cm-1)
I_obs             - float64 (relative photon flux)
snr_obs           - float64 (S/N)


----(theo_levels.csv)----
lev_id            - integer
E_calc            - float64 (1000 cm-1)
J                 - float64 (currently only for printing, does nothing in RL)
P                 - integer (currently placeholder and does nothing)
lev_name          - str     (level label of choice, I use E_calc + leading % + leading % eigenvector)


----(theo_lines.csv)----
wn_calc           - float64 (1000 cm-1)
gA_calc           - float64 (s-1)
upper_lev_id      - integer
lower_lev_id      - integer


----(known_levels.csv)----
known_lev_indices - integer
known_lev_values  - float64 (1000 cm-1)


----(known_lines.csv)----
L1                - integer
L2                - integer
wn                - float64 (same scale as wn_obs of line list)
wn_unc            - float64 (same scale as wn_obs_unc of line list)
I_obs             - float64 (same scale as I_obs of line list)
snr               - float64 (same scale as snr_obs of line list)


---- Optional ----
-all_known_levels.csv is a 1D array of known level energies (1000 cm-1) for evaluation with no column name
-all_known_levels_and_labels.csv is shape [N, 2] array of known level labels (lev_id) and energies (1000 cm-1) for evaluation with column names


---- Notes ----
-Single precision float is insufficient for high precision (8+ s.f.) level energies and transition wavenumbers
-The integers of (L1, L2, known_lev_indices, upper_lev_id, lower_lev_id) are all from lev_id
-lev_id must start from 0 (ground) and end at max(lev_id) with size (number of levels) max(lev_id) + 1
-env_config.yaml file contains directories to above files and parameters (i.e. Table 1 and 3 of the paper)