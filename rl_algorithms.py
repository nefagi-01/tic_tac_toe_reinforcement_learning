import numpy as np
import random
from collections import defaultdict, namedtuple, deque
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
                    reward + self.gamma * self.Q_states_dict[new_state][new_best_action] -
                    self.Q_states_dict[self.previous_state][self.previous_action])
        else:
            self.Q_states_dict[self.previous_state][self.previous_action] += self.alpha * (
                    reward - self.Q_states_dict[self.previous_state][self.previous_action])

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


def compute_measures(env, q_player, n_trials=500, deep=False):
    M_opt_average = 0.
    M_rand_average = 0.

    turns = ['X', 'O']

    optimal_player = OptimalPlayer(0., turns[1])
    random_player = OptimalPlayer(1., turns[1])

    for test in range(n_trials):
        if not deep:
            opt_reward = play_game(env, q_player, optimal_player, turns, testing=True)
            rand_reward = play_game(env, q_player, random_player, turns, testing=True)
        else:
            _, opt_reward = play_deep_game(env, q_player, optimal_player, turns, testing=True)
            _, rand_reward = play_deep_game(env, q_player, random_player, turns, testing=True)

        if test == n_trials / 2:
            turns = turns[::-1]
            optimal_player.set_player(turns[1])
            random_player.set_player(turns[1])

        M_opt_average += opt_reward
        M_rand_average += rand_reward

    return M_opt_average / n_trials, M_rand_average / n_trials


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQPlayer:
    """
    Description:
        A class to implement a Deep Q-learning algorithm-based player.
    """

    def __init__(self, epsilon, gamma=0.99, lr=5e-4, capacity=10000, batch_size=64):
        # Algorithm parameters
        self.epsilon = epsilon
        self.gamma = gamma  # discount factor
        self.target_net = nn.Sequential(
            nn.Linear(18, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )
        self.policy_net = nn.Sequential(
            nn.Linear(18, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())  # initialize networks to have same weights
        self.criterion = nn.HuberLoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)  # optimizer used on policy_net
        self.memory = ReplayMemory(capacity)  # buffer
        self.batch_size = batch_size
        self.steps_done = 0
        self.previous_state = None
        self.action = None
        self.player = None

    def grid_to_tensor(self, grid):
        """
        Description:
            A function to transform the grid into a 1-D tensor of size 18.
        """
        if grid is None:
            return None
        state = torch.zeros((3, 3, 2))
        state[:, :, 0][np.where(grid == self.player)] = 1
        state[:, :, 1][np.where(grid == -self.player)] = 1
        return state.view(-1)

    def act(self, grid, reward):
        # From grid to state tensor
        state = self.grid_to_tensor(grid)

        # if at least one move has been done add Transition in the buffer
        if self.steps_done > 0:
            self.memory.push(self.previous_state, torch.tensor([self.action]), state, torch.tensor([reward]))

        # None grid means game is finished
        if grid is None:
            return

        # Save state in previous_state
        self.previous_state = state

        # Choose action with epsilon-greedy
        self.steps_done += 1
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                action = self.policy_net(state).argmax().int().item()
        else:
            action = int(np.random.randint(9))
        self.action = action
        return action

    def act_test(self, grid):
        # From grid to state tensor
        state = self.grid_to_tensor(grid)

        with torch.no_grad():
            action = self.policy_net(state).argmax().int().item()

        return action

    def optimize_policy(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                             if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.view(-1, 1))

        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def set_player(self, player='X'):
        self.player = 1 if player == 'X' else -1


def play_deep_game(env, q_player, other_player, turns, other_learning=False, testing=False):
    env.reset()
    grid, _, __ = env.observe()
    average_loss = 0.
    losses = []
    for j in range(9):
        if env.current_player == turns[1]:
            move = other_player.act(grid)
        else:
            move = q_player.act(grid, 0) if not testing else q_player.act_test(grid)
            if not testing:
                loss = q_player.optimize_policy()
                if loss is not None:
                    losses.append(loss.item())
        try:
            grid, end, winner = env.step(move, print_grid=False)
        except ValueError:
            if not testing:
                q_player.act(None, -1)
            return losses, -1
        if end:
            break

    if not testing:
        q_player.act(None, env.reward(turns[0]))

    return losses, env.reward(turns[0])


class DeepVariableEpsilonQPlayer(DeepQPlayer):

    def __init__(self, epsilon_max, epsilon_min, n_star, epsilon, gamma=0.99, lr=5e-4, capacity=10000, batch_size=64):
        # Algorithm parameters
        super().__init__(epsilon, gamma=0.99, lr=5e-4, capacity=10000, batch_size=64)
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.n_star = n_star

    def update_epsilon(self, current_episode):
        self.epsilon = max(self.epsilon_min, self.epsilon_max * (1 - current_episode / self.n_star))
