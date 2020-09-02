from abc import ABC
from itertools import chain

import numpy as np
import torch
import torch.nn as nn

from agent0.deepq.config import Config


def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)


def init_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class DeepQNet(nn.Module, ABC):
    def __init__(self, action_dim, **kwargs):
        super(DeepQNet, self).__init__()
        self.cfg = Config(**kwargs)
        self.cfg.update_atoms()

        self.action_dim = action_dim
        dense = NoisyLinear if self.cfg.noisy else nn.Linear

        self.convs = nn.Sequential(
            nn.Conv2d(self.cfg.frame_stack, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        self.convs.apply(lambda m: init(m, nn.init.calculate_gain('relu')))

        if self.cfg.algo in ['iqr', 'fqf']:
            self.cosine_emb = nn.Sequential(dense(self.cfg.num_cosines, 64 * 7 * 7), nn.ReLU())
            self.cosine_emb.apply(lambda m: init(m, nn.init.calculate_gain('relu')))
        else:
            self.cosine_emb = None

        if self.cfg.algo == 'fqf':
            self.fraction_net = dense(64 * 7 * 7, self.cfg.N_fqf)
            self.fraction_net.apply(lambda m: init_xavier(m, 0.01))
        else:
            self.fraction_net = None

        self.first_dense = nn.Sequential(dense(64 * 7 * 7, 512), nn.ReLU())
        self.first_dense.apply(lambda m: init(m, nn.init.calculate_gain('relu')))

        self.p = dense(512, action_dim * self.cfg.num_atoms)
        self.p.apply(lambda m: init(m, 0.01))

        if self.cfg.dueling:
            self.v = dense(512, self.cfg.num_atoms)
            self.v.apply(lambda m: init(m, 1.0))
        else:
            self.v = None

    def params(self):
        layers = (self.convs, self.cosine_emb, self.first_dense, self.p, self.v)
        params = (x.parameters() for x in layers if x is not None)
        return chain(*params)

    def forward(self, x, iqr=False, n=32):
        if iqr:
            # Frames input
            if x.ndim == 4:
                features, taus = self.feature_embed(self.convs(x), n)
            # Feature input
            elif x.ndim == 2:
                features, taus = self.feature_embed(x, n)
            else:
                raise ValueError("No such input dim")
            features = self.first_dense(features)
        else:
            features = self.first_dense(self.convs(x))
            taus = None

        adv = self.p(features).view(-1, self.action_dim, self.cfg.num_atoms)
        if self.cfg.dueling:
            v = self.v(features).view(-1, 1, self.cfg.num_atoms)
            q = v.expand_as(adv) + (adv - adv.mean(dim=1, keepdim=True).expand_as(adv))
        else:
            q = adv

        if self.cfg.num_atoms == 1:
            q = q.squeeze(-1)

        if iqr:
            q = q.view(-1, n, self.action_dim)
            return q, taus
        else:
            return q

    # noinspection PyArgumentList
    def taus_prop(self, x):
        batch_size = x.size(0)
        log_probs = self.fraction_net(x).log_softmax(dim=-1)
        probs = log_probs.exp()
        tau0 = torch.zeros(batch_size, 1).to(x)
        tau_1n = torch.cumsum(probs, dim=-1)

        taus = torch.cat((tau0, tau_1n), dim=-1)
        taus_hat = (taus[:, :-1] + taus[:, 1:]).detach() / 2.0
        entropies = probs.mul(log_probs).neg().sum(dim=-1, keepdim=True)
        return taus, taus_hat, entropies

    # noinspection PyArgumentList
    def feature_embed(self, x, n, taus=None):
        batch_size = x.size(0)
        if taus is None:
            taus = torch.rand(batch_size, n, 1).to(x)
        ipi = np.pi * torch.arange(1, self.cfg.num_cosines + 1).to(x).view(1, 1, self.cfg.num_cosines)
        cosine = ipi.mul(taus).cos().view(batch_size * n, self.cfg.num_cosines)

        tau_embed = self.cosine_emb(cosine).view(batch_size, n, -1)
        state_embed = x.view(batch_size, 1, -1)
        features = (tau_embed * state_embed).view(batch_size * n, -1)
        return features, taus

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


if __name__ == '__main__':
    pass
