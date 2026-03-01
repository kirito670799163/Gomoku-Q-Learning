import numpy as np
import random

class QLearningAgent:
    """
    Q-learning agent for board game.
    Q-table: dict mapping state (tuple) -> numpy array of Q-values for all actions.
    """
    def __init__(self, board_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        self.board_size = board_size
        self.num_actions = board_size * board_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}  # state -> np.array of length num_actions

    def _get_valid_from_state(self, state):
        """Return list of valid action indices from state tuple."""
        valid = []
        idx = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if state[r][c] == 0:
                    valid.append(idx)
                idx += 1
        return valid

    def get_action(self, state, valid_actions):
        """
        Epsilon-greedy policy.
        valid_actions: list of currently legal actions.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_vals = self.q_table[state]
        best_value = -np.inf
        best_actions = []
        for a in valid_actions:
            if q_vals[a] > best_value:
                best_value = q_vals[a]
                best_actions = [a]
            elif q_vals[a] == best_value:
                best_actions.append(a)
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)

        current_q = self.q_table[state][action]

        if done:
            max_next_q = 0
        else:
            valid_next = self._get_valid_from_state(next_state)
            next_q_vals = self.q_table[next_state]
            if valid_next:
                max_next_q = np.max(next_q_vals[valid_next])
            else:
                max_next_q = 0

        target = reward + self.gamma * max_next_q
        self.q_table[state][action] += self.lr * (target - current_q)