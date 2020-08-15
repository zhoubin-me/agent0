import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas import np


def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)


class NatureCNN_(nn.Module):
    def __init__(self, in_channels, action_dim, dueling=True):
        super(NatureCNN_, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU())


        self.convs.apply(lambda m: init(m, nn.init.calculate_gain('relu')))
        self.p = nn.Linear(512, action_dim)
        self.p.apply(lambda m: init(m, 0.01))

        if dueling:
            self.v = nn.Linear(512, 1)
            self.v.apply(lambda m: init(m, 1.0))
        else:
            self.v = None

    def forward(self, x):
        features = self.convs(x)
        adv = self.p(features)
        if self.v is not None:
            v = self.v(features)
            q = v.expand_as(adv) + (adv - adv.mean(dim=-1, keepdim=True).expand_as(adv))
        else:
            q = adv
        return q


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

    def reset_noise(self, std=None):
        self.noise_w.data.normal_()
        self.noise_b.data.normal_()


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
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_q = nn.Sequential(
            FC(64 * 7 * 7, 512), nn.ReLU(),
            FC(512, num_atoms * action_dim)
        )

        if dueling:
            self.fc_v = nn.Sequential(
                FC(64 * 7 * 7, 512), nn.ReLU(),
                FC(512, num_atoms)
            )
        else:
            self.fc_v = None

    def forward(self, x):
        phi = self.convs(x)
        q = self.fc_q(phi).view(-1, self.action_dim, self.num_atoms)

        if self.fc_v is not None:
            v = self.fc_v(phi)
            q = v.view(-1, 1, self.num_atoms) + q - q.mean(dim=1, keepdim=True)

        return q

    def reset_noise(self, std=None):
        if std is None: std = self.noise_std
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise(std)
