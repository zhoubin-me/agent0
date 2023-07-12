from itertools import chain
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.parameter import Parameter

from agent0.deepq.new_config import AlgoEnum, C51Config, ExpConfig, IQNConfig, QRConfig


def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)


def init_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class ConvEncoder(nn.Module):
    def __init__(self, chan_dim):
        super(ConvEncoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(chan_dim, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.convs.apply(lambda m: init(m, nn.init.calculate_gain("relu")))

    def forward(self, x):
        return self.convs(x)


class DQNHead(nn.Module):
    def __init__(self, act_dim: int, feat_dim: int, dueling: bool):
        super(DQNHead, self).__init__()

        self.first_dense = nn.Linear(feat_dim, 512)
        self.first_dense.apply(lambda m: init(m, nn.init.calculate_gain("relu")))
        self.q_head = nn.Linear(512, act_dim)
        self.q_head.apply(lambda m: init(m, 0.01))

        if dueling:
            self.value_head = nn.Linear(512, 1)
            self.value_head.apply(lambda m: init(m, 1.0))
        else:
            self.value_head = None

    def forward(self, x):
        x = F.relu(self.first_dense(x))
        q = self.q_head(x)

        if self.value_head is not None:
            value = self.value_head(x)
            advantage = q - q.mean(dim=-1, keepdim=True)
            q = value + advantage
        return q

    def qval(self, x):
        return self.forward(x)


class C51Head(nn.Module):
    def __init__(self, act_dim: int, feat_dim: int, dueling: bool, cfg: C51Config):
        super(C51Head, self).__init__()
        self.first_dense = nn.Linear(feat_dim, 512)
        self.first_dense.apply(lambda m: init(m, nn.init.calculate_gain("relu")))

        self.q_head = nn.Linear(512, act_dim * cfg.num_atoms)
        self.q_head.apply(lambda m: init(m, 0.01))
        if cfg.vmax is not None:
            self.register_buffer(
                "atoms",
                torch.linspace(cfg.vmin, cfg.vmax, cfg.num_atoms),
            )
            self.atoms = self.atoms.view(1, 1, -1)
            self.delta = (cfg.vmax - cfg.vmin) / (cfg.num_atoms - 1)

        if dueling:
            self.value_head = nn.Linear(512, cfg.num_atoms)
            self.value_head.apply(lambda m: init(m, 1.0))
        else:
            self.value_head = None
        self.action_dim = act_dim

    def forward(self, x):
        x = F.relu(self.first_dense(x))
        q = self.q_head(x)
        q = rearrange(q, "b (a n) -> b a n", a=self.action_dim)

        if self.value_head is not None:
            value = self.value_head(x)
            value = rearrange(value, "b n -> b 1 n")
            advantage = q - q.mean(dim=1, keepdim=True)
            q = value + advantage
        return q

    def qval(self, x):
        q_dist = self.forward(x)
        return q_dist.softmax(dim=-1).mul(self.atoms).sum(dim=-1)


class QRHead(C51Head):
    def __init__(self, act_dim: int, feat_dim: int, dueling: bool, cfg: QRConfig):
        super(QRHead, self).__init__(act_dim, feat_dim, dueling, cfg)
        self.register_buffer(
            "cumulative_density",
            (2 * torch.arange(cfg.num_atoms) + 1) / (2.0 * cfg.num_atoms),
        )

    def qval(self, x):
        qs = self.forward(x)
        return qs.mean(dim=-1)


class IQNHead(nn.Module):
    def __init__(self, act_dim: int, feat_dim: int, dueling: bool, cfg: IQNConfig):
        super(IQNHead, self).__init__()
        self.cfg = cfg
        self.first_dense = nn.Linear(feat_dim, 512)
        self.first_dense.apply(lambda m: init(m, nn.init.calculate_gain("relu")))

        self.q_head = nn.Linear(512, act_dim)
        self.q_head.apply(lambda m: init(m, 0.01))

        if dueling:
            self.value_head = nn.Linear(512, 1)
            self.value_head.apply(lambda m: init(m, 1.0))
        else:
            self.value_head = None

        self.cosine_emb = nn.Sequential(
            nn.Linear(self.cfg.num_cosines, feat_dim), nn.ReLU()
        )
        self.cosine_emb.apply(lambda m: init(m, nn.init.calculate_gain("relu")))

    def forward(self, x, n=None, taus=None):
        # x: b d
        # featues: (b n) d
        # taus: b n 1
        features, taus, n = self.feature_emb(x, n=n, taus=taus)
        features = F.relu(self.first_dense(features))
        # q: (b n) a
        q = self.q_head(features)

        if self.value_head is not None:
            value = self.value_head(features)
            advantage = q - q.mean(dim=-1, keepdim=True)
            q = value + advantage
        q = rearrange(q, "(b n) a -> b n a", n=n)
        return q, taus

    def feature_emb(self, x, n, taus):
        batch_size = x.size(0)
        if taus is None:
            taus = torch.rand(batch_size, n, 1).to(x)
        else:
            n = taus.size(1)

        ipi = np.pi * torch.arange(1, self.cfg.num_cosines + 1).to(x)
        ipi = rearrange(ipi, "d -> 1 1 d")
        cosine = ipi.mul(taus).cos()
        cosine = rearrange(cosine, "b n d -> (b n) d")

        tau_embed = self.cosine_emb(cosine)
        tau_embed = rearrange(tau_embed, "(b n) d -> b n d", b=batch_size)
        state_embed = rearrange(x, "b d -> b 1 d")
        features = rearrange(tau_embed * state_embed, "b n d -> (b n) d")
        return features, taus, n

    def qval(self, x, n=None):
        if n is None:
            n = self.cfg.K
        qs, _ = self.forward(x, n)
        return qs.mean(dim=1)


class FQFHead(IQNHead):
    def __init__(self, act_dim: int, feat_dim: int, dueling: bool, cfg: IQNConfig):
        super(FQFHead, self).__init__(act_dim, feat_dim, dueling, cfg)
        self.fraction_net = nn.Linear(feat_dim, cfg.F)
        self.fraction_net.apply(lambda m: init_xavier(m, 0.01))

    def prop_taus(self, x):
        batch_size = x.size(0)
        log_probs = self.fraction_net(x).log_softmax(dim=-1)
        probs = log_probs.exp()
        tau0 = torch.zeros(batch_size, 1).to(x)
        tau_1n = torch.cumsum(probs, dim=-1)

        taus = torch.cat((tau0, tau_1n), dim=-1)
        taus_hat = (taus[:, :-1] + taus[:, 1:]).detach() / 2.0
        entropies = probs.mul(log_probs).neg().sum(dim=-1, keepdim=True)
        return taus.unsqueeze(-1), taus_hat.unsqueeze(-1), entropies

    def qval(self, x):
        taus, taus_hat, _ = self.prop_taus(x.detach())
        q_hat, _ = self.forward(x, taus=taus_hat)
        q = ((taus[:, 1:, :] - taus[:, :-1, :]) * q_hat).sum(dim=1)
        return q


class DeepQNet(nn.Module):
    def __init__(self, cfg: ExpConfig):
        super(DeepQNet, self).__init__()
        self.encoder = ConvEncoder(cfg.obs_shape[0])

        dummy_x = torch.rand(1, *cfg.obs_shape)
        dummy_y = self.encoder(dummy_x)
        feat_dim = dummy_y.shape[-1]

        if cfg.learner.algo == AlgoEnum.dqn:
            self.head = DQNHead(cfg.action_dim, feat_dim, cfg.learner.dueling_head)
        elif cfg.learner.algo == AlgoEnum.c51:
            self.head = C51Head(
                cfg.action_dim, feat_dim, cfg.learner.dueling_head, cfg.learner.c51
            )
        elif cfg.learner.algo == AlgoEnum.qr:
            self.head = QRHead(
                cfg.action_dim, feat_dim, cfg.learner.dueling_head, cfg.learner.qr
            )
        elif cfg.learner.algo == AlgoEnum.iqn:
            self.head = IQNHead(
                cfg.action_dim, feat_dim, cfg.learner.dueling_head, cfg.learner.iqn
            )
        elif cfg.learner.algo == AlgoEnum.fqf:
            self.head = FQFHead(
                cfg.action_dim, feat_dim, cfg.learner.dueling_head, cfg.learner.iqn
            )
        else:
            raise NotImplementedError(cfg.learner.algo)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

    def qval(self, x):
        x = self.encoder(x)
        return self.head.qval(x)

    def params(self):
        return chain(v for k, v in self.named_parameters() if "fraction" not in k)
