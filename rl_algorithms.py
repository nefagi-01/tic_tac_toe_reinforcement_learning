import numpy as np
import random
from collections import defaultdict


class QPlayer:
    """
    Description:
        A class to implement a Q algorithm-based player.
    """

    def __init__(self, alpha, gamma, epsilon):
        # Algorithm parameters
        self.previous_state = None
        self.previous_action = None
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration level

        # Algorithm state
        self.Q_states_dict = defaultdict(lambda: np.zeros(9))

    def empty(self, grid):
        available_actions = [i for i in range(9) if grid[int(i / 3), i % 3] == 0]
        return available_actions

    def act(self, grid):
        self.previous_state = np.array2string(grid)
        available_actions = self.empty(grid)
        action_scores = self.Q_states_dict[self.previous_state][available_actions]

        # Choose action
        self.previous_action = np.random.choice(available_actions) if np.random.random() < self.epsilon else \
            available_actions[
                np.argmax(action_scores)]

        return int(self.previous_action)

    def learn(self, grid, reward, end):
        if not end:
            new_state = np.array2string(grid)
            new_available_actions = self.empty(grid)
            new_action_scores = self.Q_states_dict[new_state][new_available_actions]
            new_best_action = new_available_actions[np.argmax(new_action_scores)]

            self.Q_states_dict[self.previous_state][self.previous_action] += self.alpha * (
                    reward - self.gamma * self.Q_states_dict[new_state][new_best_action])
        else:
            self.Q_states_dict[self.previous_state][self.previous_action] += self.alpha * reward
