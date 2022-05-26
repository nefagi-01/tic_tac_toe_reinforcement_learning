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

    def update_learning_rate(self, episodes):
        ratio = 0.9
        update_frequency = 2500
        if episodes % update_frequency == update_frequency - 1:
            self.alpha *= ratio

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

    def set_player(self, player):
        pass

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


def play_game(env, q_player, other_player, turns, episodes=None, other_learning=False, testing=False):
    env.reset()
    grid, _, __ = env.observe()
    if episodes != None:
        q_player.update_learning_rate(episodes)
        if other_learning:
            other_player.update_learning_rate(episodes)
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
    q_player.set_player(turns[0])

    for test in range(n_trials):
        if not deep:
            opt_reward = play_game(env, q_player, optimal_player, turns, testing=True)
            rand_reward = play_game(env, q_player, random_player, turns, testing=True)
        else:
            _, opt_reward = play_deep_game(env, q_player, optimal_player, turns, testing=True)
            _, rand_reward = play_deep_game(env, q_player, random_player, turns, testing=True)

        if test == n_trials / 2:
            turns = turns[::-1]
            q_player.set_player(turns[0])
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


def printami(lista, nome):
    print("\n" + nome)
    for x in lista:
        print(x)
    input("Press Enter to continue...")


class DeepQPlayer:
    """
    Description:
        A class to implement a Deep Q-learning algorithm-based player.
    """

    def __init__(self, epsilon, gamma=0.99, lr=5e-5, capacity=10000, batch_size=64, shared_networks=None):
        # Algorithm parameters
        self.epsilon = epsilon
        self.gamma = gamma  # discount factor
        if shared_networks is not None:
            self.policy_net, self.target_net, self.memory = shared_networks
        else:
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
            self.target_net.eval()
            self.memory = ReplayMemory(capacity)  # buffer
        self.criterion = nn.HuberLoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)  # optimizer used on policy_net
        self.milestones = 10
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=np.linspace(0,60000, num = self.milestones, dtype=int), gamma=0.6)
        self.batch_size = batch_size
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
        # print("PLAYER: ", self.player)
        state = torch.zeros((3, 3, 2))
        state[:, :, 0][np.where(grid == self.player)] = 1
        state[:, :, 1][np.where((grid != self.player) & (grid != 0))] = 1
        # print("GRID", grid, "\nSTATE Q player", state[:, :, 0], "\nSTATE other player", state[:, :, 1])
        return state.view(-1)

    def empty(self, grid):
        available_actions = [i for i in range(9) if grid[int(i / 3), i % 3] == 0]
        return available_actions

    def act(self, grid):
        # From grid to state tensor
        state = self.grid_to_tensor(grid)

        # Save state in previous_state
        self.previous_state = state

        # Choose action with epsilon-greedy
        if np.random.random() >= self.epsilon:
            with torch.no_grad():
                action = self.choose_best_action(state)
        else:
            action = int(np.random.choice(self.empty(grid)))
        self.action = action
        return action

    def act_test(self, grid):
        # From grid to state tensor
        state = self.grid_to_tensor(grid)

        with torch.no_grad():
            action = self.choose_best_action(state)

        return action

    def choose_best_action(self, state):
        with torch.no_grad():
            q_values = self.policy_net(state)
            max_q_value, _ = torch.max(q_values, 0)
            best_indices = torch.argwhere(q_values == max_q_value)
            best_action = np.random.choice(best_indices.view(-1))
            return int(best_action)

    def save_transition(self, grid, reward):
        # From grid to state tensor
        state = self.grid_to_tensor(grid)
        self.memory.push(self.previous_state, torch.tensor([self.action]), state, torch.tensor([reward]))

    def compute_loss(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        if self.batch_size > 1:
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), dtype=torch.bool)
            next_state_values = torch.zeros(self.batch_size)

            if torch.any(non_final_mask):
                non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

            state_batch = torch.stack(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = self.policy_net(state_batch).gather(1, action_batch.view(-1, 1))

            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        else:
            state_batch = batch.state[0]
            action_batch = batch.action[0]
            reward_batch = batch.reward[0]
            next_state_batch = batch.next_state[0]
            if next_state_batch is None:
                next_state_values = torch.zeros(self.batch_size)
            else:
                next_state_values = self.target_net(next_state_batch).argmax().detach()
            state_action_values = self.policy_net(state_batch)[action_batch]
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            loss = self.criterion(state_action_values, expected_state_action_values)

        return loss

    def learn(self, loss):
        if len(self.memory) < self.batch_size:
            return None

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.scheduler.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def set_player(self, player='X'):
        self.player = 1. if player == 'X' else -1.

    def get_networks(self):
        return self.policy_net, self.target_net, self.memory

def play_deep_agent_step(j, grid, player, testing, losses):
    # learning phase
    if not testing:
        if j > 1:  # save transition only if there was a previous move in the same game
            player.save_transition(grid, 0)
        loss = player.compute_loss()
        player.learn(loss)

        # Update losses
        if loss is not None:
            losses.append(loss.item())

    # move phase
    move = player.act(grid) if not testing else player.act_test(grid)

    return move, losses


def play_deep_game(env, q_player, other_player, turns, other_learning=False, testing=False):
    # Update players
    q_player.set_player(turns[0])
    other_player.set_player(turns[1])
    # Update env
    env.reset()
    grid, _, __ = env.observe()
    losses = []
    for j in range(9):
        current_player = q_player if env.current_player == turns[0] else other_player
        if current_player == other_player:
            if other_learning:
                move, losses = play_deep_agent_step(j, grid, other_player, testing, losses)
            else:
                move = other_player.act(grid)
        else:
            move, losses = play_deep_agent_step(j, grid, q_player, testing, losses)

        try:
            grid, end, winner = env.step(move, print_grid=False)
        except ValueError:
            if not testing:
                current_player.save_transition(None, -1)
            return losses, -1
        if end:
            break
    if not testing:
        q_player.save_transition(None, env.reward(turns[0]))
        if other_learning:
            other_player.save_transition(None, env.reward(turns[1]))

    return losses, env.reward(turns[0])


class DeepVariableEpsilonQPlayer(DeepQPlayer):

    def __init__(self, epsilon_max, epsilon_min, n_star, epsilon, gamma=0.99, lr=5e-5, capacity=10000, batch_size=64,
                 shared_networks=None):
        super().__init__(epsilon, gamma=gamma, lr=lr, capacity=capacity, batch_size=batch_size,
                         shared_networks=shared_networks)
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.n_star = n_star

    def update_epsilon(self, current_episode):
        self.epsilon = max(self.epsilon_min, self.epsilon_max * (1 - current_episode / self.n_star))
