# FIND OUT WHY RESULTS GO BEYOND EPISODE LENGTH????

import numpy as np
import torch
from copy import deepcopy
from math import sqrt, log
from tqdm import tqdm
from time import time

class MCTSNode:
    def __init__(self, state, reward=0, parent=None, done=False, depth_remain=0):
        self.state = state
        self.done = done
        self.reward = reward  # reward received to reach this state
        self.parent = parent  # only one parent
        self.children = {}  # MCTSNode instances for child nodes, keys are action indices

        self.action_space = state['action_space']

        # Depth remaining for the sim traj
        self.depth_remain = depth_remain

        self.N = 0.0
        self.Q = 0.0

        # Action type and action space
        self.type = state['action_type']  # 0 for A1 state, 0 for A2 state
        if self.type == 0:
            # action_space is torch 1D tensor of state graph node indices
            self.action_indices = np.arange(self.action_space.size(0))
            self.dqn_Qs = torch.zeros(self.action_space.size(0))
        else:
            # action_space is tuple (cand_graphs, cand_energies, cand_wn_obs_indices, cand_edge_indices) used in env.step
            self.action_indices = np.arange(len(self.action_space[0]))
            self.dqn_Qs = torch.zeros(len(self.action_space[0]))

        self.num_actions = len(self.action_indices)
        self.visited_actions = np.full(self.num_actions, False, dtype=bool)  # Init no actions have been visited

    def choose_action(self):
        '''Chooses the action corresponding to an unvisited tree search node to be expanded next.'''
        remaining_actions_indices = self.action_indices[~self.visited_actions]
        idx = np.random.randint(len(remaining_actions_indices))  # Choose random action using uniform prob
        chosen_idx = remaining_actions_indices[idx]
        self.visited_actions[chosen_idx] = True
        chosen_Qs = self.dqn_Qs.clone()  # for env.step()
        chosen_Qs[chosen_idx] += 1
        return chosen_Qs, self.action_space  # for env.step()

    def is_fully_expanded(self):
        '''Check if all valid actions have been used for expansion'''
        return self.visited_actions.all()
    
    def update_estimates(self, R):
        '''Update visit count and Q-estimates'''
        self.N += 1
        self.Q = self.Q + ((R - self.Q) / self.N)

class MCTSAgent():
    def __init__(self, env, params=None):
        '''
        dqn_agent is used for action masking
        '''
        self.env = env

        # Hyparams
        self.C_p = params['C_p']
        self.C_ps = 1  # does not matter at start because root Q=0
        self.depth = params['depth']
        self.N_sim = params['N_sim']

        self.root_nodes = []
        self.root_node = MCTSNode(deepcopy(self.env.state), reward=0, parent=None, done=False, depth_remain=self.depth)
        self.root_nodes.append(self.root_node)

        # Debugging
        self.prnt = False

    def search(self):
        counter = 0
        search_start = time()
        self.total_stepped_rewards = 0  # from stepping only
        self.largest_traj_R = -float('inf')
        while not self.root_node.done:
            t0 = time()
            counter += 1
            #self.N_sim = len(self.root_node.action_indices)  # if greedy
            self.step()
            t1 = time()
            lev_idx = self.root_node.state['level_to_find'].item()
            if self.root_node.state['action_type'] == 0:
                E_obs = self.root_node.state['graph'].x[lev_idx][1] * self.env.E_scale * 1e3
                print(f'id {lev_idx} label {self.env.lev_name[lev_idx]} E_obs {E_obs:.4f} cm-1')
            print(f'Step {counter} ({(t1-t0)/60:.0f}m) - total rewards so far: {self.total_stepped_rewards:.4f}')
            if (time() - search_start) / 3600 > 23.5:  # if search takes too long, stop
                print('Search took too long, stopping')
                break

    def step(self):
        '''
        Make a step in the environment using MCTS after N_sim simulations
        '''
        # Simulate N_sim trajectories
        for i in tqdm(range(self.N_sim), disable=True):
            if self.fully_explored(self.root_node):
                print('Tree fully explored! Ending sims')
                break
            self.simulation()
            # Update exploration param
            self.C_ps = self.C_p * abs(self.root_node.Q)  # C_ps is always scaled to a factor of C_p of avg root node Q
        #print('root_Q', self.root_node.Q)
        a_type = self.root_node.state['action_type'] + 1
        print(f'|A{a_type}| = {len(self.root_node.children)}')
        highest_visits = -100000
        highest_visits_children = []
        for child in self.root_node.children:
            child_node = self.root_node.children[child]
            if child_node.N > highest_visits:
                highest_visits = child_node.N
                highest_visits_children = [child_node]
            elif child_node.N == highest_visits:
                highest_visits_children.append(child_node)
        highest_visits_child = np.random.choice(highest_visits_children)
        # Prune all other children (siblings of new root node)
        for action_idx, child in list(self.root_node.children.items()):
            if child is not highest_visits_child:
                self._delete_subtree(child)
        # Change root node into the decided child node
        self.root_node = MCTSNode(deepcopy(highest_visits_child.state), highest_visits_child.reward, parent=None, 
                                  done=highest_visits_child.done, depth_remain=self.depth)  # New root node but without children and old stats (fresh_start)
        self._delete_subtree(highest_visits_child)  # Remove subtree of best child
        self.root_nodes.append(self.root_node)
        self.total_stepped_rewards += self.root_node.reward
        print(f'Reward: {self.root_node.reward:.4f}')

    def simulation(self):
        '''Carry out the four steps per simulation of the MCTS algorithm'''
        self.R = 0  # reset reward for this simulation
        node_for_exp = self.selection()  # select node for expansion, self.R is updated
        expanded_node = self.expansion(node_for_exp)  # expand into a leaf node for rollout, self.R is updated using the expanded node
        sim_R = self.rollout(expanded_node)  # collect rewards from rollout with finite depth, self.R is updated and equal to sim_R
        if sim_R + self.total_stepped_rewards > self.largest_traj_R:
            self.largest_traj_R = sim_R + self.total_stepped_rewards
            self.largest_traj_R_state = deepcopy(self.env.state)  # end of trajectory state
            print(f'New largest trajectory reward: {self.largest_traj_R:.4f}')
        if self.prnt:
            print('4. Backprop ---------------')
        self.backprop(expanded_node, sim_R)  # update stats for all nodes in the trajectory of this simulation

    def get_ucb1_term(self, Q, parent_N, child_N, model_prior=1):
        '''Computes the UCB1 value based on the supplied node statistics.'''
        parent_N = parent_N + 1e-5
        child_N = child_N + 1e-5
        ci_term = sqrt((2 * log(parent_N)) / child_N)
        ucb1_value = Q + self.C_ps * model_prior * ci_term
        return ucb1_value

    def selection(self):
        '''Selection step during one simulation of MCTS'''
        current_node = self.root_node
        if self.prnt:
            print('1. Selection ---------------')
        # if terminal, no children, and/or not fully expanded, skip this loop and use current node for expansion
        while (not current_node.done and  # select current_node if terminal state
               current_node.is_fully_expanded() and  # select current_node if not fully expanded
               len(current_node.children) > 0):  # has children to select
            if self.prnt:
                print(current_node)
                print(current_node.state)
                print('a_space', current_node.action_indices)
                print('is not terminal and fully expanded and has children to select, children (a_idx, Q, N, pN, UCB1):')
            best_score = -float('inf')
            best_action = None
            best_children = []  # set up incase ucb1 are the same
            for a_idx, child in current_node.children.items():
                if child.done:
                    continue  # don't care about terminal children
                child_N = child.N
                Q = child.Q
                parent_N = current_node.N
                ucb1 = self.get_ucb1_term(Q, parent_N, child_N)
                if self.prnt:
                    print(a_idx, round(Q, 2), child_N, parent_N, round(ucb1, 2))
                if ucb1 > best_score:
                    best_score = ucb1
                    best_children = [(a_idx, child)]
                    best_action = a_idx
                elif ucb1 == best_score:
                    best_children.append((a_idx, child))
            if not best_children:
                #print('Warning - no valid children to select')
                break  # No valid children to select

            # Pick a random child from children with the same UCB1 scores
            idx = np.random.randint(len(best_children))
            a_idx, best_child = best_children[idx]
            current_node = best_child
            self.R += current_node.reward  # add reward from selected node to total reward
            if self.prnt:
                print(best_child, 'selected with index', a_idx)
        if self.prnt:
            print(current_node, 'selected for expansion')
        return current_node  # the selected node to expand

    def expansion(self, current_node):
        '''Expand selected node with a child node'''
        if self.prnt:
            print('2. Expansion ---------------')
            print('from', current_node)
        # an untried action is selected, where Qs is 1 for the action and 0 otherwise, for env.epsilon=0
        if current_node.done:
            print('Warning - tried to expand terminal node, returning current_node')
            return current_node  # Can't expand a terminal node
        # If node has reached the simulation depth, do not expand
        if current_node.depth_remain == 0:
            return current_node
        # If node is fully expanded, it shouldn't be selected for expansion, unless all of its child nodes are done states
        if current_node.is_fully_expanded():
            #print('Expansion not valid, all children are done, rollout will occur for this node')
            return current_node
        dqn_Qs, action_space = current_node.choose_action()  # for dqn_Qs, all actions are 0 except for the chosen action
        if self.prnt:
            print('Random action Qs:')
            print(dqn_Qs)
            print('Current node visited action Qs:')
            print(current_node.visited_actions)
        # Prepare env state for stepping
        self.env.epsilon = 0  # greedy because we are expanding
        state_for_step = deepcopy(current_node.state)
        self.env.state = state_for_step
        self.env.state_to_attr()
        # Step into expanded state
        exp_state, action_index, reward, done= self.env.step(dqn_Qs)
        # Store expanded child node
        child_node = MCTSNode(deepcopy(exp_state), reward=reward.item(), parent=current_node, done=done, 
                              depth_remain=current_node.depth_remain - 1)
        current_node.children[action_index] = child_node  # store child in parent_node.children dict
        self.R += reward.item()  # expansion happened, reward is added to total trajectory reward
        if self.prnt:
            print('Expanded node', child_node)
            print(child_node.state)
            print(f'Expansion reward {self.R:.2f}')
        return child_node  # the expanded node to rollout from

    def rollout(self, current_node):
        '''Roll out from a node (new child)'''
        if self.prnt:
            print('3. Rollout ---------------')
            print('from', current_node)
        if current_node.done:  # do not rollout if done node
            return self.R
        # Prepare env state for rollout
        state_for_rollout = deepcopy(current_node.state)
        self.env.state = deepcopy(state_for_rollout)
        self.env.state_to_attr()
        self.env.epsilon = 1  # using uniform random sampling policy
        # Record results
        if self.prnt:
                print(f'Just from expansion - total reward {self.R:.2f}')
        for i in range(current_node.depth_remain):
            if self.env.state['action_type'] == 0:
                Qs = torch.zeros(self.env.state['action_space'].size(0))  # for A1 lsit of nodes
            else:
                Qs = torch.zeros(len(self.env.state['action_space'][0]))  # for A2 list of candidate graphs
            # epsilon is one so zero Qs is placeholder
            next_state, action_index, reward, done = self.env.step(Qs)  # env will change state from stepping
            self.R += reward.item()
            if done:
                break
            if self.prnt:
                print(f'Step {i} - total reward {self.R:.2f}')
        return self.R

    def backprop(self, current_node, R):
        '''Recursively update node stats involved in the trajectory of the simulation'''
        current_node.update_estimates(R)
        if current_node.parent is None:
            return
        parent_node = current_node.parent
        self.backprop(parent_node, R)

    def _delete_subtree(self, node):
        '''Recursively delete all descendants of a node to free memory.'''
        for child in node.children.values():
            self._delete_subtree(child)
        node.children.clear()
        node.parent = None
        del node

    def fully_explored(self, node):
        '''Recursiveley check if all nodes have been fully explored up to depth (no more sim required)'''
        if node.depth_remain == 0:
            return True
        if node.is_fully_expanded() == False:
            return False
        for child in node.children.values():
            if not self.fully_explored(child):
                return False
        return True