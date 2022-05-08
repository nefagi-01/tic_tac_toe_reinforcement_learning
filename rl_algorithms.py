import numpy as np
import random
from collections import defaultdict
from tic_env import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from scipy.special import softmax

class QPlayer:
    """
    Description:
        A class to implement a Q algorithm-based player.
    """

    def __init__(self, alpha, gamma, epsilon, q_values=None):
        # Algorithm parameters
        self.previous_state = None
        self.previous_action = None
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration level

        # Algorithm state
        self.Q_states_dict = defaultdict(lambda: np.zeros(9)) if q_values is None else q_values

    def act(self, grid):
        self.previous_state = np.array2string(grid)
        available_actions = self.empty(grid)
        action_scores = self.Q_states_dict[self.previous_state][available_actions]

        # Choose action
        self.previous_action = np.random.choice(available_actions) if np.random.random() < self.epsilon else \
            self.choose_best_action(available_actions, action_scores)

        return int(self.previous_action)

    def learn(self, grid, reward, end):
        if not end:
            new_state = np.array2string(grid)
            new_available_actions = self.empty(grid)
            new_action_scores = self.Q_states_dict[new_state][new_available_actions]
            new_best_action = self.choose_best_action(new_available_actions, new_action_scores)

            self.Q_states_dict[self.previous_state][self.previous_action] += self.alpha * (
                    reward + self.gamma * self.Q_states_dict[new_state][new_best_action] - self.Q_states_dict[self.previous_state][self.previous_action])
        else:
            self.Q_states_dict[self.previous_state][self.previous_action] += self.alpha * (reward - self.Q_states_dict[self.previous_state][self.previous_action])

    def empty(self, grid):
        available_actions = [i for i in range(9) if grid[int(i / 3), i % 3] == 0]
        return available_actions

    def choose_best_action(self, available_actions, available_actions_scores):
        max_scores_indices = np.where(available_actions_scores == np.max(available_actions_scores))[0]
        chosen_max_score_index = np.random.choice(max_scores_indices)
        return available_actions[chosen_max_score_index]

    def get_Q_values(self):
        return self.Q_states_dict
    
    def act_test(self, grid):
        self.previous_state = np.array2string(grid)
        available_actions = self.empty(grid)
        action_scores = self.Q_states_dict[self.previous_state][available_actions]

        # Choose action
        self.previous_action = self.choose_best_action(available_actions, action_scores)

        return int(self.previous_action)

class VariableEpsilonQPlayer(QPlayer):
    """
    Description:
        A class to implement a Q algorithm-based player.
    """

    def __init__(self, alpha, gamma, epsilon_max, epsilon_min, n_star, q_values=None):
        # Algorithm parameters
        super().__init__(alpha, gamma, None, q_values)
        self.previous_state = None
        self.previous_action = None
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.n_star = n_star

    def update_epsilon(self, current_episode):
        self.epsilon = max(self.epsilon_min, self.epsilon_max * (1 - current_episode / self.n_star))

def play_game(env, q_player, other_player, turns, other_learning=False, testing=False):
    env.reset()
    grid, _, __ = env.observe()
    for j in range(9):
        if env.current_player == turns[1]:
            if other_learning and j > 1 and not testing:  # if it's not the first time the QPlayer is playing
                other_player.learn(grid, 0, False)
            move = other_player.act(grid)
        else:
            if j > 1 and not testing:  # if it's not the first time the QPlayer is playing
                q_player.learn(grid, 0, False)
            move = q_player.act(grid) if not testing else q_player.act_test(grid)
        grid, end, winner = env.step(move, print_grid=False)
        if end:
            break

    if not testing:
        q_player.learn(grid, env.reward(turns[0]), True)

        if other_learning:
            other_player.learn(grid, env.reward(turns[1]), True)

    return env.reward(turns[0])


def compute_measures(env, q_player, n_trials=500):
    M_opt_average = 0.
    M_rand_average = 0.

    turns = ['X', 'O']

    optimal_player = OptimalPlayer(0., turns[1])
    random_player = OptimalPlayer(1., turns[1])

    for test in range(n_trials):
        opt_reward = play_game(env, q_player, optimal_player, turns, testing=True)
        rand_reward = play_game(env, q_player, random_player, turns, testing=True)

        if test == n_trials/2:
            turns = turns[::-1]
            optimal_player.set_player(turns[1])
            random_player.set_player(turns[1])

        M_opt_average += opt_reward
        M_rand_average += rand_reward

    return M_opt_average / n_trials, M_rand_average / n_trials




class DeepQPlayer:
    """
    Description:
        A class to implement a Deep Q-learning algorithm-based player.
    """
    def __init__(self, epsilon, gamma=0.99, lr=5e-4):
        # Algorithm parameters
        self.epsilon = epsilon
        self.gamma = gamma  # discount factor
        self.network = nn.Sequential(
                        nn.Linear(18,128),
                        nn.ReLU(),
                        nn.Linear(128,128),
                        nn.ReLU(),
                        nn.Linear(128,9),
                        nn.ReLU()
                        )
        self.criterion = nn.HuberLoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.previous_Q = None
        self.current_Q = None

    def grid_to_tensor(self, grid):
        state = torch.zeros((3,3,2))
        state[:,:,0][np.where(grid == 1)] = 1
        state[:,:,1][np.where(grid == -1)] = 1
        return state
    
    def act(self, grid):
        #From grid to state tensor
        state = self.grid_to_tensor(grid)
        #Compute Q_values
        Q_values = self.network(state.view(-1))
        # Choose action
        action = np.random.randint(9) if np.random.random() < self.epsilon else self.choose_best_action(Q_values)
        with torch.no_grad():
            self.previous_Q = self.current_Q
            self.current_Q = Q_values[action]
        return action
    
    def act_test(self, grid):
        #From grid to state tensor
        state = self.grid_to_tensor(grid)
        #Compute Q_values
        Q_values = self.network(state.view(-1))
        # Choose action
        action = self.choose_best_action(Q_values)

    def learn(self, reward, end):
        if not end:
            target = self.current_Q.detach()*self.gamma + reward
            output = self.previous_Q
            loss = self.criterion(output, target)
            self.network.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            target = reward
            output = self.current_Q
            loss = self.criterion(output, target)
            self.network.zero_grad()
            loss.backward()
            self.optimizer.step()
            

    

    def choose_best_action(self, Q_values):
        with torch.no_grad():
            Q_values_numpy = Q_values.numpy()
            Q_values_softmax = softmax(Q_values_numpy)
            max_scores_indices = np.where(Q_values_softmax == np.max(Q_values_softmax))[0]
            chosen_max_score_index = np.random.choice(max_scores_indices)
            return int(chosen_max_score_index)
        

def play_deep_game(env, q_player, other_player, turns, other_learning=False, testing=False):
    env.reset()
    grid, _, __ = env.observe()
    for j in range(9):
        if env.current_player == turns[1]:
            if other_learning and j > 1 and not testing:  # if it's not the first time the QPlayer is playing
                other_player.learn(grid, 0, False)
            move = other_player.act(grid)
        else:
            move = q_player.act(grid) if not testing else q_player.act_test(grid)
            if j > 1 and not testing:  # if it's not the first time the QPlayer is playing
                q_player.learn(torch.tensor(0.), False)
        
        try:
            grid, end, winner = env.step(move, print_grid=False)
        #catch unavailable action and finish game
        except ValueError:
            if not testing:
                q_player.learn(torch.tensor(-1.), True)
                return -1
        if end:
            break

    if not testing:
        q_player.learn(torch.tensor(float(env.reward(turns[0]))), True)

        if other_learning:
            other_player.learn(grid, env.reward(turns[1]), True)

    return env.reward(turns[0])