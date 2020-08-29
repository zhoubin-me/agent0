from abc import ABC

import numpy as np
import torch
import torch.nn as nn


def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)


class ModelPrediction(nn.Module, ABC):
    def __init__(self, in_channels, action_dim, dueling=False, num_atoms=1, noisy=False, noise_std=0.5):
        super(ModelPrediction, self).__init__()
        self.num_atoms = num_atoms
        self.action_dim = action_dim
        self.noise_std = noise_std
        dense = NoisyLinear if noisy else nn.Linear

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(),
            dense(64 * 7 * 7, 512), nn.ReLU())


class NatureCNN(nn.Module, ABC):
    def __init__(self, in_channels, action_dim, dueling=False, num_atoms=1, noisy=False, noise_std=0.5, feature_mult=1):
        super(NatureCNN, self).__init__()

        self.num_atoms = num_atoms
        self.action_dim = action_dim
        self.noise_std = noise_std
        self.feature_mult = feature_mult
        dense = NoisyLinear if noisy else nn.Linear

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32 * feature_mult, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32 * feature_mult, 64 * feature_mult, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64 * feature_mult, 64 * feature_mult, 3, stride=1), nn.ReLU(), nn.Flatten())
        self.first_dense = nn.Sequential(dense(64 * 7 * 7 * feature_mult, 512 * feature_mult), nn.ReLU())

        self.convs.apply(lambda m: init(m, nn.init.calculate_gain('relu')))
        self.first_dense.apply(lambda m: init(m, nn.init.calculate_gain('relu')))

        self.p = dense(512 * feature_mult, action_dim * num_atoms)
        self.p.apply(lambda m: init(m, 0.01))

        if dueling:
            self.v = dense(512 * feature_mult, num_atoms)
            self.v.apply(lambda m: init(m, 1.0))
        else:
            self.v = None

    def forward(self, x):
        features = self.first_dense(self.convs(x))
        adv = self.p(features).view(-1, self.action_dim, self.num_atoms)
        if self.v is not None:
            v = self.v(features).view(-1, 1, self.num_atoms)
            q = v.expand_as(adv) + (adv - adv.mean(dim=1, keepdim=True).expand_as(adv))
        else:
            q = adv

        if self.num_atoms == 1:
            q = q.squeeze(-1)
        return q

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# noinspection PyArgumentList
class NoisyLinear(nn.Module, ABC):
    def __init__(self, in_features, out_features, std_init=0.4, noisy_layer_std=0.1):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.noisy_layer_std = noisy_layer_std
        self.weight_mu = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=True)
        self.weight_sigma = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=True)
        self.register_buffer('weight_epsilon', torch.zeros((out_features, in_features)))
        self.bias_mu = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.bias_sigma = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        self.register_buffer('noise_in', torch.zeros(in_features))
        self.register_buffer('noise_out_weight', torch.zeros(out_features))
        self.register_buffer('noise_out_bias', torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        self.noise_in.normal_(std=self.noisy_layer_std)
        self.noise_out_weight.normal_(std=self.noisy_layer_std)
        self.noise_out_bias.normal_(std=self.noisy_layer_std)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    @staticmethod
    def transform_noise(x):
        return x.sign().mul(x.abs().sqrt())
