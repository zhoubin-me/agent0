
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import chain

def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)


class NatureCNN(nn.Module):
    def __init__(self, in_channels, action_dim):
        super(NatureCNN, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU())

        self.v = nn.Linear(512, 1)
        self.p = nn.Linear(512, action_dim)

        self.convs.apply(lambda m: init(m, nn.init.calculate_gain('relu')))
        self.p.apply(lambda m: init(m, 0.01))
        self.v.apply(lambda m: init(m, 1.0))

    def forward(self, x):
        features = self.convs(x)
        v = self.v(features)
        adv = self.p(features)
        q = v.expand_as(adv) + (adv - adv.mean(dim=-1, keepdim=True).expand_as(adv))
        return q