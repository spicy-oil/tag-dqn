Logs are available for greedy search and MCTS, logs are too large for tag_dqn so excluded in the repository, but will be provided upon reasonable request

Exact numbers are not guaranteed for MCTS and TAG-DQN, but should be reproduceable for greedy search.

nd2_k_results.xlsx is human readable version of Nd II known lines of the final MDP state of the largest reward episode across all seeded TAG-DQN runs. 
Rows are grouped by level (separated by two blank rows) and each line appears twice so that each group represents all known lines of a level.
The columns are:
lid1       - lower level index
lid2       - upper level index
E1         - lower level energy
E2         - upper level energy
J1         - lower level J 
J2         - upper level J
snr        - signal-to-noise-ratio
wn_ritz    - Ritz WN from level energy optimisation
wn_obs     - observed wavenumber of the matched line list entry
wn_obs_unc - standard unc. of the above
obs-ritz   - wn_obs minus wn_ritz, usually indicates problems with wrong match, poor fit from blend, etc. if >> wn_obs_unc
I_calc     - corresponding to the graph edge feature (relative photon flux)
I_obs      - corresponding to the graph edge feature (relative photon flux, same scale as I_calc)
gA         - weighted transition probability (/s)
L1         - lower level label
L2         - upper level label
This data is tentative and not validated by humans, most levels & their lines are believable, but there are likely also a few wrong levels (e.g. lid2 395 is very suspicious).

There exist flaws in our N_c and Acc. metrics, e.g.:
-Matching with (published) known values within a tolerance (0.05 cm-1) can overestimate N_c as there could be false level energies in this tolerance (e.g. lid 395).
-But not all levels found by RL could be known, which underestimates N_c.
-So these metrics are approximate (probably within ~5%).