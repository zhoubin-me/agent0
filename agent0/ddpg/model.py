from abc import ABC

import numpy as np
import torch
import torch.nn as nn


def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)


class DDPGMLP(nn.Module, ABC):
    def __init__(self, num_inputs, action_dim, max_action, hidden_size=256):
        super(DDPGMLP, self).__init__()

        self.max_action = max_action
        self.v = nn.Sequential(
            nn.Linear(num_inputs + action_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.p = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim), nn.Tanh()
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
