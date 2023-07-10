from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)


class DDPGMLP(nn.Module):
    def __init__(self, num_inputs, action_dim, max_action, hidden_size=256):
        super(DDPGMLP, self).__init__()

        self.max_action = max_action
        self.v = nn.Sequential(
            nn.Linear(num_inputs + action_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.p = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )

        self.apply(lambda m: init(m, np.sqrt(2)))

    def act(self, x):
        return self.p(x) * self.max_action

    def action_value(self, state, action):
        return self.v(torch.cat([state, action], dim=1))

    def get_policy_params(self):
        return self.p.parameters()

    def get_value_params(self):
        return self.v.parameters()


class SACMLP(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, num_inputs, action_dim, max_action, hidden_size=256):
        super(SACMLP, self).__init__()
        self.max_action = max_action
        self.v = nn.Sequential(
            nn.Linear(num_inputs + action_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.v2 = nn.Sequential(
            nn.Linear(num_inputs + action_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.p = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim * 2),
        )

        self.apply(lambda m: init(m, np.sqrt(2)))

    def act(self, x):
        action_mean, action_log_std = torch.chunk(self.p(x), 2, dim=-1)
        action_log_std = action_log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        dist = Normal(action_mean, action_log_std.exp())

        xs = dist.rsample()
        action = xs.tanh() * self.max_action

        action_log_prob = dist.log_prob(xs) - torch.log(1 - action.pow(2) + self.eps)
        entropy = action_log_prob.sum(-1, keepdim=True).neg()

        return action, entropy, action_mean.tanh() * self.max_action

    def action_value(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.v(x), self.v2(x)

    def get_policy_params(self):
        return self.p.parameters()

    def get_value_params(self):
        return chain(self.v.parameters(), self.v2.parameters())


class TD3MLP(nn.Module):
    def __init__(self, num_inputs, action_dim, max_action, hidden_size=256):
        super(TD3MLP, self).__init__()
        self.max_action = max_action
        self.v = nn.Sequential(
            nn.Linear(num_inputs + action_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.v2 = nn.Sequential(
            nn.Linear(num_inputs + action_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.p = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )

        self.apply(lambda m: init(m, np.sqrt(2)))

    def act(self, x):
        return self.p(x) * self.max_action

    def action_value(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.v(x), self.v2(x)

    def get_policy_params(self):
        return self.p.parameters()

    def get_value_params(self):
        return chain(self.v.parameters(), self.v2.parameters())
