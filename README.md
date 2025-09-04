<!-- omit in toc -->
# TAG-DQN
The Term Analysis with Graph Deep Q-Network (TAG-DQN) package contains all code and data associated with the paper 'Atomic Fine Structure Determination with Graph Reinforcement Learning'. The package contains the Markov decision process (MDP) environment for a term analysis integrated with a graph deep Q-network for reinforcement learning (RL).

The main idea is having an algorithm that learns from automatic and strictly defined (programmed) term analysis procedures, where tens of thousands of attempts to determine unknown levels are learned (e.g. which level, in what order, and which energy + lines) using reward signals (i.e. human guidance). 

---
<!-- omit in toc -->
## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
    - [Data Files](#data-files)
    - [Running with Configuration File and Reward Parameters](#running-with-configuration-file-and-reward-parameters)
    - [Suggestions](#suggestions)
- [Data](#data)
- [Comments](#comments)
    - [Reward Function](#reward-function)
    - [Environment Parameters and Hyperparameters](#environment-parameters-and-hyperparameters)

---

## Requirements
The program is in Python and the required Python packages are listed in `setup.py`, the neural network (NN) implementations do not use GPU.

It is possible to run TAG-DQN on a personal computer. Memory cost mainly arise from replay buffer size and NN complexity, both of which can be reduced from their final values from the paper, but with likely reduced performance unless MDP complexity is also reduced. Buffer size of 2000 was feasible with 2 GNN attention heads on a personal computer with 32 GB RAM.

Hyperparameter tuning for new environments and running multiple seeds for best conclusions are preferable. These should be realised in parallel on a remote high performance computing (HPC) facility. Example scripts for the Imperial College HPC can be found under `./data/grid_search_scripts/`.

---

## Installation
Ideally install Python >= 3.11.11 (developed using this version) in a separate virtual environment (e.g. conda) to avoid conflicts with packages and dependencies. Download this repository, under the directory containing `setup.py` and under the new virtual environment run `setup.py`:

```bash
conda create -n tag-dqn python=3.11.11
conda activate tag-dqn
pip install -e .
```

-e if let source code be editable.

---

## Usage

#### Data Files
Running TAG-DQN requires minimum 5 data files defining a term analysis state
- `line_list.csv`
- `known_levels.csv`
- `known_lines.csv`
- `theo_levels.csv`
- `theo_lines.csv `

These are listed under `./data/envs/` for each of the four case studies and explained in a `readme.txt`. 

#### Running with Configuration File and Reward Parameters
An example TAG-DQN run would be

```python
import tag_dqn
tag_dqn.run_tag_dqn('config.yaml', seed=42, reward_params='reward.pth')
```

Each MDP environment also requires a `config.yaml` configuration file that points to the five files above and contain MDP environment parameters and model hyperparameters. It is a text file for human editing and available for each case study under `./data/envs/`.

The reward parameter file `reward.pth` is optional. It will be `None` if unspecified, in which case the reward parameters of the paper `./data/envs/reward.pth` will be used. The parameters must correspond to the reward model `NNReward()` of `tag_dqn/dqn/dqn_reward.py`.

#### Suggestions
More details in `examples.py` with comments.

To check correct installation, running greedy search in the Nd III or Co II MDP to reproduce results numbers is desirable (takes about 5 mins):

```python
tag_dqn.run_greedy('./data/envs/nd3/config.yaml', reward_params='./data/envs/reward.pth') 
```
and/or change `ep_length`, `episodes`, and `tr_start_ep` to small numbers in `config.yaml` to check if TAG-DQN runs.

We advise against taking TAG-DQN outputs with certainty. Firstly, some of the levels determined will be wrong. But more importantly, validation by spectrum inspections and semi-empirical caluclation improvements should be vital procedures in acceptable term analysis (these are neither part of the MDP nor RL). 

---

## Data
All MDP environment data of the paper are under `./data/envs/`. For each case study, we additionally include the Cowan code parameters, if applicable. 

Final results reported for the paper are under `./data/results/final/`. Grid search results are compiled under `./data/results/grid_search/`.


---

## Comments

#### Reward Function
RL performance for level determination is highly dependent on the reward function, which ideally generalises to unknown levels and lines of the term analysis of interest. Reward learning in `example.py` takes about 10 minutes on a personal computer, HPC is not required.

When the MDP environment and its parameters (e.g. $\delta E$, $\mathit{\Delta} E$, $\mathit{\Delta} I$) differ drastically from the four case studies (e.g. data from different spectrometers or spectral resolutions), training a new reward function instead of ours (`./data/envs/reward.pth`) is considerable. But this would need sufficient number of known levels for sufficient number of training MDP state transitions. In the paper, we trained using only Co II expert MDP state transitions, and results for Co II was the best. Training using initial MDP states of the Nd II-III case studies were not possible due to small number of known levels. Inevitably, the trained reward function was less ideal for Nd II-III as their theoretical calculations, line list, and human experts were different.


#### Environment Parameters and Hyperparameters
Let's talk about each parameter in the `config.yaml` file (they are defined in the comments of example config file under `./data/envs/`)

- `min_snr` limits MDP complexity by limiting the number of graph edges (theoretical transitions) based on whether we expect to observe them. Certainly, some neglected theoretical transitions are observable but not omitted from $\mathcal{A}^2$ due to uncertainties in $gA_{\text{calc}}$ and $S/N_{\text{calc}}$ estimation, while some theoretical transitions remaining on the graph are also not observable. The number is chosen to compromise between graph complexity and the inclusion rate of observable theoretical transitions.
- `spec_range` limits MDP complexity by limiting the line list range. In determining unknown levels using groups of lines expected from a particular spectral region, this parameter makes the RL process efficient.
- `wn_range` is $\mathit{\Delta} E$ of the paper, it limits MDP complexity by setting a candidate line wavenumber search range in $\mathcal{A}^2$. Setting this requires domain knowledge, where for typical ab initio calculations this is around 10% of level energies and for semi-empirical calculations this can be less than 1% (e.g. if searching for levels in a known config with calculations parameterised for that config).
- `tol` is $\delta E$ of the paper, it limits MDP complexity by setting a maximum tolerance for repeating candidate level energies $E_{\text{obs}}$. Setting this requires domain knowledge, where it is around 0.05 cm-1 for Fourier transform (FT) spectroscopy in the visible and UV, but could be smaller for IR FT spectra and larger for grating spectra.
- `int_tol` is $\mathit{\Delta} I$ of the paper, in units of one order of magnitude. It limits MDP complexity by setting a candidate line intensity search range in $\mathcal{A}^2$. Setting this requires domain knowledge, as uncertainties in theoretical transition probabilities (TP) can be greater than one order of magnitude. However, larger TPs tend to be more accurately calculated (up to ~10% accuracy, especially if only considering branching ratios) and tend to be the observable lines, so it does not need to cover a few orders of magnitude - around one was found to be OK.
- `A2_max` limits MDP complexity by setting a maximum size for $\mathcal{A}^2$, above which the no-op action is enforced. This represents the automatic forfeit of a level determination if the number of ambiguous energy values are too large, which can be reduced by first determining more connecting levels or other means outside of the current RL framework (e.g. line profile expectations). Setting this number effectively tells TAG-DQN to ignore a particular level until certain connecting levels are found, otherwise it would be wasting exploration time and computational memory in large action spaces.
- `float_levs` is a flag which, when false, limits MDP complexity by fixing initial MDP state known level energies in all level energy optimisations (this happens once every two steps in the environment). For the Nd II case studies, only 12 levels are known in the MDP initial state, so there was no reason to fix levels. It is likely best to fix levels as the initial MDP state level energies are likely validated by humans.
- `ep_length` is $H$ of the paper and limits MDP complexity by limiting the maximum number of level determinations in an episode. Increasing this is at the cost of computation time and does not necessarily give more correctly determined level energies, especially if atomic structure calculations require improvements. 
- `gamma` is the MDP discount factor. We did some tuning on this and found 0.9 and 1.0 to be slightly worse than the most common value 0.99 (which we use).
- `gat_n_layers` is the number of layers in the GNN. It offered diminishing returns in performance after 2 or 3 given our hyperparameter search ranges, while increasing computation time and memory cost significantly.
- `gat_hidden_size` is sensitive until up to a certain value (complex enough, ~32), but increasing it does cost more resources.
- `gat_heads` greatly affects GNN complexity and resource cost, it offered diminishing returns for us past 4 given our hyperparameter search ranges.
- `mlp_hidden_size` is sensitive until up to a certain value (complex enough, ~32), then slowly decreases RL performance as it is increased. The latter likely comes from exploration by noisy-nets.
- `episodes` sets the total number of steps in the enivronment for training, it can be as long as resources allow, but we recommend keeping track of reward alignment with the number of correct level energies, as the latter can plateau or decrease after training for many episodes, depending on the MDP, especially the reward function.
- `buffer_capacity` is one of the main memory cost, ideally it should cover the number of steps required for convergence. We chose 10000 because it allows us the smallest HPC job specification (128 GB) for feasible queue times.
- `tr_start_ep` is the number of episodes to randomly step in the MDP environment to partially fill the buffer so that RL training could begin on random MDP transitions. The standard practice is to fill about 10%. We did not find sensitivities to this parameter.
- `batch_size` directly affects computation time as the loss from each batch is looped over (due to varying action space sizes). We tried up to 32, it was better than 16 but took twice as long per episode, if we half the number of episodes for training, then 32 is about as good as 16 without halving the number of episodes.
- `adam_lr` was found surprisingly best at the default value 0.001, it was expected lower. This is possibly because the more common DQN problems require much higher number of steps to reach convergence.
- `min_epsilon` for $\epsilon$-greedy exploration is used if noisy-nets are switched off, 0.05 ~ 0.2 should be fine, though tuning would be best. The rate of decay is a factor of 0.99 per episode, can be changed in `tag_dqn/dqn/dqn_trainer.py`.
- `tau` is rather insensitive but best at the smaller values we tried ~0.001.
- `steps_per_train` affects training time when combined with `ep_length` and `episodes`, it is sensitive and tuning is recommended.
- `patience` is an early-stopping parameter for training, if needed. We did not use this.
- `duel` enables the duelling architecture that is critical for TAG-DQN, do not switch off.
- `double` did not seem to affect RL, try in different environments if interested.
- `noisy` enables noisy networks, which were found to be more efficient than epsilon-greedy exploration.
- `sigma_0` is the initialisation value for all weights multiplying noise in noisy-nets, it is somewhat insensitive but can be tuned if resources allow, we use the common value 0.5 and initialisation.
- `per_alpha` and `per_beta` is for prioritised experience replay (PER), which lowered TAG-DQN performance. PER is switched off (`per_alpha` is zero) in all example configs.
- `n_step` is the multi-step returns parameter, it was found sensitive across MDP environments (Nd II vs Nd III).
- `C_p` is for the Upper Confidence Bound for Trees (UCT) exploration of Monte-Carlo tree search (MCTS), tuning is recommended if running MCTS.
- `depth` is the rollout depth for MCTS. The major flaw of standard MCTS in this MDP is that random rollout (random action sampling for future states to estimate Q-value) is very inefficient. Usually in advanced MCTS this rollout is guided by NNs (e.g. AlphaGo).
- `N_sim` is the number of simulation MDP trajectories to sample before making a step in the MDP. It should be large enough compared to the MDP branching factor (action space dimension) to produce a reasonable distribution over the visit counts of the possible next states (the largest visit count is selected in standard MCTS).
  