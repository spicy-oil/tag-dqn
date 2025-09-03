import torch
import numpy as np
import time as time
import matplotlib.pyplot as plt
from copy import deepcopy
from .dqn_subnets import NoisyLinear
from .dqn_data_proc import comp, get_pd_table

class Trainer():
    def __init__(self, q_net, q_net_t, q_net_largest, env, replay_buffer, optimizer, z, 
                 batch_size, steps_per_train, tau, double=True, noisy=True, 
                 min_epsilon=0.1, patience=2e6, 
                 known_lev_values=None, all_known_levs=None, all_known_levels_and_labels=None):
        '''
        patience is n target updates without improvement before stopping training
        '''
        # Training stats to track
        self.total_ep_rewards = []
        self.total_lev_found = []
        self.sigma_mu_ratio = []
        self.ep_losses = []
        self.known_lev_values = known_lev_values
        self.all_known_levs = all_known_levs
        self.all_known_levels_and_labels = all_known_levels_and_labels
        if all_known_levels_and_labels is not None:
            self.term_analysis_checks = [(0,0,0,0,0,0,0,0,0,0,0,0)]
        else:
            self.term_analysis_checks = [(0,0,0,0,0,0,0,0,0)]
        self.term_analysis_check_eps = [0]

        # Key components
        self.q_net = q_net # Online
        self.q_net_t = q_net_t # Target
        self.q_net_largest = q_net_largest # Largest reward Q net
        self.env = env # Environment
        self.replay_buffer = replay_buffer # Prioritised experience replay multi-step return buffer
        self.optimizer = optimizer
        self.z = z # Fixed support atoms for disttributional RL

        # Hyperparams
        self.batch_size = batch_size # Number of state transitions per gradient descent
        self.steps_per_train = steps_per_train # Number of environment steps (state transitions) before a gradient descent
        self.tau = tau  # number of episodes for an update of the target q net
        self.double = double  # Whether double DQN
        self.noisy = noisy  # Use of noisy nets for exploration
        self.min_epsilon = min_epsilon  # minimum value for exploration rate, if not using noisy nets
        self.patience = patience  # Number of gradient descents to wait for improvement before stopping training
        self.patience_counter = 0  # Counter for patience

    def train_q_net(self, episodes, tr_start_ep):
        '''
        Training loop for int episodes
        '''
        self.training_start_time = time.time()
        self.env._print_decisions = False
        self.largest_total_episode_reward = -1e5
        self.largest_total_episode_reward_final_state = None
        self.patience_counter = 0  # Counter for patience

        self.q_net.train() 
        for episode in range(episodes):
            ep_start_time = time.time()

            beta = min(1.0, self.replay_buffer.beta_start + episode * (1.0 - self.replay_buffer.beta_start) / episodes)  # Anneal beta

            state = deepcopy(self.env.reset())
            done = False
            total_reward = 0
            steps_since_train = 0  # Counter for steps since last training
            start_training = False
            loss = None
            ep_loss = 0

            if episode < tr_start_ep:
                self.env.epsilon = 1 # randomly explore till training begins
            else:
                if self.noisy:
                    #self.env.epsilon = max(0.01, self.env.epsilon * 0.95) # fast epsilon annealing for noisy nets
                    self.env.epsilon = 0.0  # act full greedily if using noisy NNs
                else: # Anneal epsilon
                    # Decaying epsilon for exploration-exploitation balance
                    self.env.epsilon = max(self.min_epsilon, self.env.epsilon * 0.99)  
                start_training = True
                steps_since_train = 0  # Counter for steps since last training

            while not done:
                with torch.no_grad(): # Get new exp without tracking grad
                    # Forward pass through the Q-network to get action and Q-value
                    Q_values, action_info = self.q_net(state)
                    # Take action, observe reward and next state
                    next_state, action_index, reward, done = self.env.step(Q_values)

                # Save exp only if there are valid levels to search for
                if (state['action_type'] == 0 and len(action_info) > 0) or state['action_type'] == 1:  
                    # Save the transition in the replay buffer, this will calculate n_step return and n_step state
                    self.replay_buffer.push(deepcopy(state), 
                                            action_index, 
                                            reward, 
                                            deepcopy(next_state), 
                                            done)

                # Train the Q-network after every few steps (e.g., every steps_per_train)
                steps_since_train += 1
                if (steps_since_train >= self.steps_per_train) & start_training:
                    #self.q_net.train() 
                    loss = self._q_net_step(self.q_net, self.q_net_t, self.replay_buffer, self.optimizer, self.batch_size, self.noisy, beta)
                    if loss is not None:
                        ep_loss += loss
                    #self.q_net.eval() noisy net requires train() mode for noise in forward pass, so keep train mode
                    steps_since_train = 0  # Reset step counter after training
                    self.patience_counter += 1  # Increment patience counter
                    # Track weight sigmas for noisy nets
                    if self.noisy:
                        mlp_mean_sus = []
                        mlps = [self.q_net.A_node, self.q_net.V_mod]
                        if self.q_net.duel:
                            mlps = mlps + [self.q_net.V1_state, self.q_net.V2_state]
                        for mlp in mlps:
                            layer_mean_sus = []
                            for layer in mlp.mlp:
                                if isinstance(layer, NoisyLinear):
                                    layer_mean_mu = torch.abs(layer.weight_mu).mean().item()
                                    layer_mean_sigma = torch.abs(layer.weight_sigma).mean().item()
                                    layer_mean_sus.append(layer_mean_sigma/layer_mean_mu)
                            mlp_mean_su = np.mean(layer_mean_sus)
                            mlp_mean_sus.append(mlp_mean_su)
                        self.sigma_mu_ratio.append(np.mean(mlp_mean_sus))
                    else:
                        self.sigma_mu_ratio.append(0)
                    # Soft update the target Q-network
                    self.soft_update(tau=self.tau)
                #record reward for this episode
                reward = reward.item()
                total_reward += reward
                # Change state
                state = deepcopy(next_state)

            self.total_ep_rewards.append(total_reward)
            self.total_lev_found.append(self.env.levels_found)
            self.ep_losses.append(ep_loss)

            # Periodically hard update the target Q-network
            if ((episode-tr_start_ep) % 256 == 0) & start_training:
                self.q_net_t.load_state_dict(self.q_net.state_dict())
                #print('Target Q net hard-updated')

            # Track best episode so far
            if (total_reward > self.largest_total_episode_reward) & start_training:
                self.largest_total_episode_reward = total_reward
                self.largest_total_episode_reward_final_state = next_state
                self.q_net_largest.load_state_dict(self.q_net.state_dict())
                print(f'New largest reward: {round(total_reward, 4)}')
                self.patience_counter = 0 # Counter for patience

            # Periodically track downstream performance
            if ((episode-tr_start_ep) % 8 == 0) and start_training:
                #print('Recording downstream performance')
                self.term_analysis_checks.append(self.term_analysis_check(prnt=False))
                print(f'Most recent largest reward N_c = {self.term_analysis_checks[-1][-2]}')
                self.term_analysis_check_eps.append(episode)

            ep_duration = time.time() - ep_start_time
            w_s = np.mean(self.sigma_mu_ratio[-32:]) if self.sigma_mu_ratio else 0
            aer = np.mean(self.q_net.AER_mean[-64:]) if self.q_net.aen else 0
            aer_s = np.mean(self.q_net.AER_std[-64:]) if self.q_net.aen else 0
            print(f'Ep {episode+1} ({ep_duration:.0f}s), '
                  f'Ep Reward: {total_reward:.4f}, '
                  f'N_lev: {self.env.levels_found}, '
                  f'Epsilon: {self.env.epsilon:.3f}, '
                  f's/u: {w_s:.3f}, '  # from the most recent 32 grad updates
                  f'AER: {aer:.3f}({aer_s:.3f}), '  # from the episode
                  f'Loss: {ep_loss:.2e}')

            if (self.patience_counter >= self.patience) & start_training:
                print(f'No improvement in {self.patience} optimisations, stopping training...')
                break
            
            duration = (time.time() - self.training_start_time)
            if duration > 24 * 3600 - 5 * 60:
                print('Only 5 mins left until 24h walltime, stopping training...')
                break

        self.q_net.eval() 

    def term_analysis(self, q_net, env, prnt=True):
        # make a copy
        env = deepcopy(env)

        state = env.reset()
        env._print_decisions = prnt

        done = False
        total_reward = 0

        rewards = []

        q_net.eval() # no more training, also removes noise contributions in noisy nets
        env.epsilon = 0 # no more exploring
        with torch.no_grad():
            while not done:
                # Forward pass through the Q-network to get action and Q-value
                Q_values, action_info = q_net(state)
                
                # Take action, observe reward and next state
                next_state, action_index, reward, done = env.step(Q_values)

                rewards.append(reward.item())
                total_reward += reward.item()

                state = next_state
        
        G = 0
        for t, r in enumerate(rewards):
            G = G + (self.replay_buffer.gamma ** t) * r

        if prnt:
            print(f'Term analysis total reward: {total_reward:.2f}')
            print(f'Term analysis cumulative discounted return: {G:.2f}')
        q_net.train()
        return round(total_reward, 2), env

    def term_analysis_check(self, prnt=True, results=False):
        if self.all_known_levels_and_labels is None:
            # Agent
            if prnt:
                print('------- Final agent term analysis results -------')
            total_reward, env = self.term_analysis(self.q_net, self.env, prnt)  #  env is deepcopied inside
            lev_names, levs = env._get_known_levs()
            N_correct, N_found = comp(lev_names, levs, self.known_lev_values, 
                                      self.all_known_levs, self.all_known_levels_and_labels, prnt)
            # Largest reward agent
            if prnt:
                print('------- Largest reward agent term analysis results -------')
            total_reward_l, env = self.term_analysis(self.q_net_largest, self.env, prnt)  #  env is deepcopied inside
            lev_names, levs = env._get_known_levs()
            N_correct_l, N_found_l = comp(lev_names, levs, self.known_lev_values, 
                                      self.all_known_levs, self.all_known_levels_and_labels, prnt)
            # Largest reward state
            if prnt:
                print('------- Largest reward final state results -------')
            env = deepcopy(self.env)
            env.state = self.largest_total_episode_reward_final_state
            env.state_to_attr()
            lev_names, levs = env._get_known_levs()
            N_correct_ls, N_found_ls = comp(lev_names, levs, self.known_lev_values,
                                      self.all_known_levs, self.all_known_levels_and_labels, prnt)

            return(self.largest_total_episode_reward, 
                    total_reward, N_correct, N_found, 
                    total_reward_l, N_correct_l, N_found_l, N_correct_ls, N_found_ls)
        else:
            # Agent
            if prnt:
                print('------- Final agent term analysis results -------')
            total_reward, env = self.term_analysis(self.q_net, self.env, prnt)  #  env is deepcopied inside
            lev_names, levs = env._get_known_levs()
            N_correct, N_found, N_id_correct = comp(lev_names, levs, self.known_lev_values, 
                                      self.all_known_levs, self.all_known_levels_and_labels, prnt)
            # Largest reward agent
            if prnt:
                print('------- Largest reward agent term analysis results -------')
            total_reward_l, env = self.term_analysis(self.q_net_largest, self.env, prnt)  #  env is deepcopied inside
            lev_names, levs = env._get_known_levs()
            N_correct_l, N_found_l, N_id_correct_l = comp(lev_names, levs, self.known_lev_values, 
                                      self.all_known_levs, self.all_known_levels_and_labels, prnt)
            # Largest reward state
            if prnt:
                print('------- Largest reward final state results -------')
            env = deepcopy(self.env)
            env.state = self.largest_total_episode_reward_final_state
            env.state_to_attr()
            lev_names, levs = env._get_known_levs()
            N_correct_ls, N_found_ls, N_id_correct_ls = comp(lev_names, levs, self.known_lev_values,
                                      self.all_known_levs, self.all_known_levels_and_labels, prnt)
            return(self.largest_total_episode_reward, 
                    total_reward, N_correct, N_found, 
                    total_reward_l, N_correct_l, N_found_l, N_correct_ls, N_found_ls, 
                    N_id_correct, N_id_correct_l, N_id_correct_ls)

    def plot_training(self, seed='', output_dir='./trial_results'):
        
        # Learning curve
        plt.figure(figsize=(12, 4))
        plt.plot(range(len(self.total_ep_rewards)), self.total_ep_rewards, 'gray')
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.savefig(output_dir+'/learning_curves/training_'+str(seed)+'.png', dpi=100, bbox_inches='tight')
        np.save(output_dir+'/learning_curves/values_'+str(seed)+'.npy', self.total_ep_rewards)
        
        # Exploration curve for noisy nets
        plt.figure(figsize=(12, 4))
        plt.plot(range(len(self.sigma_mu_ratio)), self.sigma_mu_ratio, 'gray')
        plt.xlabel('DQN optimisation step')
        plt.ylabel(r'mean $|w_\sigma|/|w_\mu|$')
        plt.savefig(output_dir+'/exploration_curves/exploration_'+str(seed)+'.png', dpi=100, bbox_inches='tight')
        np.save(output_dir+'/exploration_curves/values_'+str(seed)+'.npy', self.sigma_mu_ratio)

        # Loss curve
        plt.figure(figsize=(12, 4))
        plt.plot(range(len(self.ep_losses)), self.ep_losses, 'gray')
        plt.xlabel('episode')
        plt.ylabel(r'total episode loss')
        plt.savefig(output_dir+'/loss_curves/loss_'+str(seed)+'.png', dpi=100, bbox_inches='tight')
        np.save(output_dir+'/loss_curves/values_'+str(seed)+'.npy', self.ep_losses)
        
        # Check correct levels found
        plt.figure(figsize=(12, 4))
        color = ['r', 'g', 'b']
        labels = ['current agent', 'largest reward agent', 'largest reward final state']
        TA = np.array(self.term_analysis_checks)
        np.save(output_dir+'/alignment_curves/TA' + str(seed) + '.npy', TA)
        for i, j in enumerate([2, 5, 7]):
            plt.plot(self.term_analysis_check_eps, TA[:, j], color=color[i], label = labels[i])
        plt.xlabel('episode')
        plt.ylabel(r'correct levels found')
        plt.legend()
        plt.savefig(output_dir+'/alignment_curves/Nc_'+str(seed)+'.png', dpi=100, bbox_inches='tight')
        plt.close('all')

        # Track action spaces
        np.save(output_dir+'/action_spaces/A1_'+str(seed)+'.npy', self.env.A1_sizes)
        np.save(output_dir+'/action_spaces/A2_'+str(seed)+'.npy', self.env.A2_sizes)

        # Largest reward state
        env = deepcopy(self.env)
        env.state = self.largest_total_episode_reward_final_state
        env.state_to_attr()
        m, c, x, y = env._get_pop(plot=False)
        # Only the values for plotting
        np.savez(output_dir+'/boltzmann_plots/pop_'+str(seed)+'.npz', 
                 x=x * self.env.E_scale, y=y, m=m / self.env.E_scale, c=c)
        # Use like this
        # data = np.load('pop.npz')
        # x = data['x']
        # y = data['y']
        # m = data['m'].item()
        # c = data['c'].item()

        result_graph = env.state['graph'] # or
        #result_graph =  trainer.largest_total_episode_reward_final_state['graph']

        # # Get classified line list pd dataframe, check for dodgy levels and add rejected level indices to prune list
        classified_linelist = get_pd_table(result_graph, env.E_scale, env.lev_name, env.J, known_only=True) 
        classified_linelist.to_csv(output_dir+'/classified_linelist/cll_'+str(seed)+'.csv', index=False)

    def _q_net_step(self, q_net, q_net_t, replay_buffer, optimizer, batch_size, noisy, beta):
        '''
        Training step for Rainbow with or without dsitributional RL
        '''
        if len(replay_buffer) < batch_size:
            return  # Not enough data to sample a batch
        
        optimizer.zero_grad()

        # Reset (simulate) noise for each batch sample forward pass
        if noisy:
            with torch.no_grad():
                q_net.A_node.reset_noise()
                q_net.V_mod.reset_noise()
                q_net_t.A_node.reset_noise()
                q_net_t.V_mod.reset_noise()
                if self.q_net.duel:
                    q_net.V1_state.reset_noise()
                    q_net.V2_state.reset_noise()                
                    q_net_t.V1_state.reset_noise()
                    q_net_t.V2_state.reset_noise()

        # Sample from buffer with priority-based weighting
        indices, states, actions, rewards, next_states, dones, weights = replay_buffer.sample(beta)
        
        # If Q-value learning ----------------------------------------------------------------------
        if q_net.N_atoms == 1:

            Qs = []
            target_Qs = []
            A_sizes = []

            for i in range(batch_size): 
                state_i = states[i]
                next_state_i = next_states[i]

                if state_i['action_type'] == 0: # level selection
                    A_sizes.append(len(state_i['action_space'])) # number of valid levels
                else: # candidate selection
                    A_sizes.append(len(state_i['action_space'][0])) # number of valid candidates

                # Q_theta(S_t, A_t)
                Q_values, _ = q_net(state_i)
                Qs.append(Q_values[actions[i]])

                # Double Q-learning
                Q_values_next, _ = q_net(next_state_i)  # next_states[i] is n_step after states[i] because of n_step return
                next_action = Q_values_next.argmax()  # Best action from Q-network

                with torch.no_grad():
                    target_Q_values_next, _ = q_net_t(next_state_i)  # Evaluate using target network
                    if self.double:
                        target_Q_value = target_Q_values_next[next_action]  # but use the online network action
                    else:
                        target_Q_value = target_Q_values_next.max()  # use largest target Q
                    target_Q_value = rewards[i] + (1 - dones[i]) * replay_buffer.gamma ** (replay_buffer.n_step) * target_Q_value  # Q value is zero if the step after n_step is terminal
                    target_Qs.append(target_Q_value)

            Qs = torch.stack(Qs) # [batch_size] if N_atoms = 1, else [batch_size, 1]
            target_Qs = torch.stack(target_Qs)  # [batch_size] if N_atoms = 1, else [batch_size, 1]

            # Store TD Error
            td_errors = (Qs - target_Qs).detach()
            replay_buffer.update_priorities(indices, td_errors.abs().numpy())  # Update priorities

            weights = torch.tensor(weights)
            # Apply importance sampling weights
            loss = torch.nn.functional.mse_loss(Qs * weights, target_Qs * weights)
        
        # If Q-distribution learning ---------------------------------------------------------------
        else: 
            batch_Q_dists = [] # [batch_size, N_atoms]
            batch_target_Q_dists = [] # [batch_size, N_atoms]
            batch_target_z = [] # [batch_size, N_atoms]
            A_sizes = []

            for i in range(batch_size):
                state_i = states[i]
                next_state_i = next_states[i]

                if state_i['action_type'] == 0: # level selection
                    A_sizes.append(len(state_i['action_space'])) # number of valid levels
                else: # candidate selection
                    A_sizes.append(len(state_i['action_space'][0])) # number of valid candidates

                Q_dists, _ = q_net(state_i) # [N_actions, N_atoms]
                batch_Q_dists.append(Q_dists[actions[i]]) # append Q_dist p(S_t, A_t)) of action taken [N_atoms]

                # Double Q-learning
                Q_dists_next, _ = q_net(next_state_i) # next_states[i] is n_step after states[i] because of n_step return
                Q_values_next = (Q_dists_next * self.z).sum(-1)  # Obtain mean Q values from Q distributions
                next_action = Q_values_next.argmax()  # Best action from online Q-network

                with torch.no_grad():
                    target_Q_dists_next, _ = q_net_t(next_state_i)  # Evaluate using target network
                    target_Q_dist = target_Q_dists_next[next_action]  # using the online network action for Double Q-learning
                    target_z = rewards[i] + (1 - dones[i]) * replay_buffer.gamma ** (replay_buffer.n_step) * self.z 
                    batch_target_Q_dists.append(target_Q_dist)
                    batch_target_z.append(target_z)

            batch_Q_dists = torch.stack(batch_Q_dists)
            batch_target_Q_dists = torch.stack(batch_target_Q_dists)
            batch_target_z = torch.stack(batch_target_z)

            # Compute projected target Q dist
            projected_target_Q_dists = projection_distribution(self.z, batch_target_z, batch_target_Q_dists, dones)
            
            # Compute KL divergence, make sure all dists are normalised
            kl_divergence = (projected_target_Q_dists * (torch.log(projected_target_Q_dists + 1e-5) - 
                                                        torch.log(batch_Q_dists + 1e-5))).sum(dim=-1) # [batch_size]
            replay_buffer.update_priorities(indices, kl_divergence.detach().numpy() * (np.array(A_sizes)/100) ** self.apr) # Update priorities

            weights = torch.tensor(weights)
            # Apply importance sampling weights and compute the loss as the mean KL divergence
            loss = (kl_divergence * weights).mean()

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def soft_update(self, tau=0.005):
        """
        Soft update the target network weights:
        θ_target = τ * θ_online + (1 - τ) * θ_target
        """
        for target_param, online_param in zip(self.q_net_t.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

def projection_distribution(z, batch_target_z, batch_target_Q_dists, dones):
    '''
    Projects the target Q-value distribution onto the fixed support.
    '''
    # return projected_dist
    N_atoms = len(z)
    delta_z = torch.diff(z)[0]  # Atom spacing

    # Create empty projected distribution
    projected_dist = torch.zeros_like(batch_target_Q_dists)  # [batch_size, N_atoms]

    for i in range(batch_target_Q_dists.shape[0]):  # Batch loop
        if dones[i]:
            # Terminal state: Create a Dirac delta at the reward value (batch_target_z[i])
            # Find the index of the support atom closest to batch_target_z[i]
            reward = batch_target_z[i]
            closest_index = torch.argmin(torch.abs(z - reward))
            projected_dist[i, closest_index] = 1.0  # All probability mass at the closest atom
        else:
            # Non-terminal state: Project the distribution as usual
            b_j = (batch_target_z[i] - z[0]) / delta_z  # Compute projection index
            l = b_j.floor().long().clamp(0, N_atoms - 1)  # Lower bound index
            u = b_j.ceil().long().clamp(0, N_atoms - 1)   # Upper bound index

            # Distribute probability mass
            projected_dist[i].scatter_add_(0, l, batch_target_Q_dists[i] * (u - b_j))
            projected_dist[i].scatter_add_(0, u, batch_target_Q_dists[i] * (b_j - l))

    # Renormalize to ensure valid probability distributions
    projected_dist = projected_dist / (projected_dist.sum(dim=-1, keepdim=True) )

    return projected_dist

def moving_average(arr, window_size=64):
    arr = np.pad(arr, (window_size // 2, window_size // 2), mode='mean', stat_length=window_size)
    arr = np.convolve(arr, np.ones(window_size) / window_size, mode='same')
    return arr[window_size // 2: - window_size // 2]

