import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)


class NatureCNN(nn.Module):
    def __init__(self, in_channels, action_dim, dueling=False, num_atoms=1, noisy=False, noise_std=0.5):
        super(NatureCNN, self).__init__()

        self.num_atoms = num_atoms
        self.action_dim = action_dim
        self.noise_std = noise_std
        FC = NoisyLinear if noisy else nn.Linear

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(),
            FC(64 * 7 * 7, 512), nn.ReLU())

        # self.convs.apply(lambda m: init(m, nn.init.calculate_gain('relu')))
        self.p = FC(512, action_dim * num_atoms)
        # self.p.apply(lambda m: init(m, 0.01))

        if dueling:
            self.v = FC(512, num_atoms)
            # self.v.apply(lambda m: init(m, 1.0))
        else:
            self.v = None

    def forward(self, x):
        features = self.convs(x)
        adv = self.p(features).view(-1, self.action_dim, self.num_atoms)
        if self.v is not None:
            v = self.v(features).view(-1, 1, self.num_atoms)
            q = v.expand_as(adv) + (adv - adv.mean(dim=1, keepdim=True).expand_as(adv))
        else:
            q = adv
        return q

    def reset_noise(self, std=None):
        if std is None: std = self.noise_std
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise(std)


class NoisyLinear(nn.Module):
    def __init__(self, in_size, out_size, sigma=0.5):
        super(NoisyLinear, self).__init__()
        self.linear_mu = nn.Linear(in_size, out_size)
        self.linear_sigma = nn.Linear(in_size, out_size)

        self.register_buffer('noise_w', torch.zeros_like(self.linear_mu.weight))
        self.register_buffer('noise_b', torch.zeros_like(self.linear_mu.bias))

        self.sigma = sigma

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            x = F.linear(x,
                         self.linear_mu.weight + self.linear_sigma.weight * self.noise_w,
                         self.linear_mu.bias + self.linear_sigma.bias * self.noise_b)
        else:
            x = self.linear_mu(x)
        return x

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.linear_mu.weight.size(1))
        self.linear_mu.weight.data.uniform_(-stdv, stdv)
        self.linear_mu.bias.data.uniform_(-stdv, stdv)

        self.linear_sigma.weight.data.fill_(self.sigma * stdv)
        self.linear_sigma.bias.data.fill_(self.sigma * stdv)

    def reset_noise(self):
        self.noise_w.data.normal_()
        self.noise_b.data.normal_()


class NoisyLinear_(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear_, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

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

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self, std=None):
        if std is None: std = self.std_init
        self.noise_in.normal_(std=std)
        self.noise_out_weight.normal_(std=std)
        self.noise_out_bias.normal_(std=std)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    def transform_noise(self, x):
        return x.sign().mul(x.abs().sqrt())
