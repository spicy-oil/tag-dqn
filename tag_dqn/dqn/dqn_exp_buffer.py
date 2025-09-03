import numpy as np
from collections import deque

class PrioritisedReplayBuffer:
    def __init__(self, capacity, batch_size, alpha=0.5, beta_start=0.4, n_step=4, gamma=0.99):
        '''
        Prioritised experience buffer with multi-step return
        '''
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.pos = 0  # buffer index for the oldest transition and the newest transition to replace the oldest transition
        self.priorities = np.zeros((capacity,), dtype=np.float64)
        self.alpha = alpha  # prioritisation exponent
        self.beta_start = beta_start
        self.n_step = n_step  # n for multi-step returns
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, state, action_index, reward, next_state, done):
        '''
        Stores transition in buffer, handling multi-step rewards
        '''
        transition = (state, action_index, reward, next_state, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return
        
        # Compute multi-step return quantities for loss:
        # |R^n_t + gamma^n_t * max_a Q_theta_target(S_(t+n), a) - Q_theta(S_t, A_t)|^2
        state, action_index = self.n_step_buffer[0][:2]  # the oldest state and action of the n_step_buffer for Q_theta(S_t, A_t) of the loss
        reward, next_state, done = self._get_n_step_info()  # R^n_t, S_(t+n), d_(t+n)

        # Store transition in main buffer
        max_priority = self.priorities.max() if self.buffer else 1  # if self.buffer is empty [], use 1e5 so highest priority
        #max_priority = 0.03  # highest priority is ~max TD error, this is ~1/2 of max reward per two steps
        if len(self.buffer) < self.capacity:  # if buffer is not full, store multi-step transition
            self.buffer.append((state, action_index, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action_index, reward, next_state, done)  # oldest transition of the whole buffer gets replaced
        self.priorities[self.pos] = max_priority  # latest multi-step transition is always highest priority
        self.pos = (self.pos + 1) % self.capacity

    def _get_n_step_info(self):
        '''
        Returns n-step discounted return and next state
        '''
        reward, next_state, done = 0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]  # the most recent state transition used for loss
        for i in reversed(range(len(self.n_step_buffer))):  # start from the latest state transition
            r, ns, d = self.n_step_buffer[i][2:5]
            reward = r + self.gamma * reward * (1 - d) # (1-d) effectively makes n_step cap at steps till done
            if d:
                next_state, done = ns, d # if less steps till done vs n_step, then the done state will be used for loss function
        return reward, next_state, done

    def sample(self, beta):
        '''
        Samples batch using prioritised experience replay
        '''
        if len(self.buffer) < self.batch_size:
            return None

        priorities = self.priorities[: len(self.buffer)] ** self.alpha  # indexing for when buffer is not at capacity
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)  # using priorities probs to sample from buffer
        samples = [self.buffer[idx] for idx in indices]

        # Lower probs are sampled less often, so higher weights for loss
        weights = (len(self.buffer) * probs[indices]) ** (-beta)  # beta increases throughout training 
        weights /= weights.max()  # Normalise

        states, actions, rewards, next_states, dones = zip(*samples)
        return indices, states, actions, rewards, next_states, dones, weights

    def update_priorities(self, indices, n_step_transition_loss):
        '''
        Updates priorities using TD errors or KL divs
        '''
        self.priorities[indices] = n_step_transition_loss + 1e-6  # small number to avoid zeros

    def __len__(self):
        return len(self.buffer)