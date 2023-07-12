from abc import ABC
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from agent0.common.MixtureSameFamily import MixtureSameFamily
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

        if self.cfg.noisy:
            if self.cfg.gmm_layer:
                dense = GMMLinear
            else:
                dense = NoisyLinear
        else:
            dense = nn.Linear

        self.convs = nn.Sequential(
            nn.Conv2d(self.cfg.frame_stack, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.convs.apply(lambda m: init(m, nn.init.calculate_gain("relu")))

        if self.cfg.algo == "c51":
            self.register_buffer(
                "atoms",
                torch.linspace(self.cfg.v_min, self.cfg.v_max, self.cfg.num_atoms),
            )
            self.delta_atom = (self.cfg.v_max - self.cfg.v_min) / (
                self.cfg.num_atoms - 1
            )

        if self.cfg.algo == "qr":
            self.register_buffer(
                "cumulative_density",
                (2 * torch.arange(self.cfg.num_atoms) + 1) / (2.0 * self.cfg.num_atoms),
            )

        if self.cfg.algo in ["iqr", "fqf"]:
            self.cosine_emb = nn.Sequential(
                dense(self.cfg.num_cosines, 64 * 7 * 7), nn.ReLU()
            )
            self.cosine_emb.apply(lambda m: init(m, nn.init.calculate_gain("relu")))
        else:
            self.cosine_emb = None

        if self.cfg.algo in ["fqf"]:
            self.fraction_net = dense(64 * 7 * 7, self.cfg.N_fqf)
            self.fraction_net.apply(lambda m: init_xavier(m, 0.01))
        else:
            self.fraction_net = None

        self.first_dense = nn.Sequential(dense(64 * 7 * 7, 512), nn.ReLU())
        self.first_dense.apply(lambda m: init(m, nn.init.calculate_gain("relu")))

        self.p = dense(512, action_dim * self.cfg.num_atoms)
        self.p.apply(lambda m: init(m, 0.01))

        self.phi = None

        if self.cfg.dueling:
            if self.cfg.algo == "gmm":
                self.v = dense(512, self.cfg.num_atoms // 3)
            else:
                self.v = dense(512, self.cfg.num_atoms)
            self.v.apply(lambda m: init(m, 1.0))
        else:
            self.v = None

        self.qvals = {
            "c51": lambda x: self.calc_c51_q(x),
            "qr": lambda x: self.forward(x).mean(-1),
            "iqr": lambda x: self.forward_iqr(x, n=self.cfg.K_iqr)[0].mean(1),
            "dqn": lambda x: self.forward(x),
            "mdqn": lambda x: self.forward(x),
            "fqf": lambda x: self.calc_fqf_q(x),
            "gmm": lambda x: self.calc_gmm_q(x),
        }

        assert self.cfg.algo in self.qvals

    def params(self):
        layers = (self.convs, self.cosine_emb, self.first_dense, self.p, self.v)
        params = (x.parameters() for x in layers if x is not None)
        return chain(*params)

    def forward_iqr(self, x, taus=None, n=32):
        if x.ndim == 4:
            features, taus, n = self.feature_embed(self.convs(x), taus=taus, n=n)
        elif x.ndim == 2:
            features, taus, n = self.feature_embed(x, taus=taus, n=n)
        else:
            raise ValueError("No such input dim")
        features = self.first_dense(features)
        self.phi = features
        adv = self.p(features).view(-1, self.action_dim, self.cfg.num_atoms)

        if self.cfg.dueling:
            v = self.v(features).view(-1, 1, self.cfg.num_atoms)
            q = v.expand_as(adv) + (adv - adv.mean(dim=1, keepdim=True).expand_as(adv))
        else:
            q = adv

        q = q.view(-1, n, self.action_dim * self.cfg.num_atoms)
        return q, taus

    def forward_gmm(self, x):
        features = self.first_dense(self.convs(x))
        self.phi = features
        adv = self.p(features).view(-1, self.action_dim, self.cfg.num_atoms)
        q_mean, q_logstd, q_weight = adv.split(
            dim=-1, split_size=self.cfg.num_atoms // 3
        )
        if self.cfg.dueling:
            v = self.v(features).view(-1, 1, self.cfg.num_atoms // 3)
            q_mean = v.expand_as(q_mean) + (
                q_mean - q_mean.mean(dim=1, keepdim=True).expand_as(q_mean)
            )
        return (
            q_mean,
            q_logstd.tanh().mul(self.cfg.gmm_max_std).exp(),
            q_weight.softmax(-1),
        )

    def forward(self, x):
        features = self.first_dense(self.convs(x))
        self.phi = features
        adv = self.p(features).view(-1, self.action_dim, self.cfg.num_atoms)
        if self.cfg.dueling:
            v = self.v(features).view(-1, 1, self.cfg.num_atoms)
            q = v.expand_as(adv) + (adv - adv.mean(dim=1, keepdim=True).expand_as(adv))
        else:
            q = adv
        if self.cfg.num_atoms == 1:
            q = q.squeeze(-1)
        return q

    def calc_q(self, x):
        return self.qvals[self.cfg.algo](x)

    def calc_c51_q(self, st):
        return self.forward(st).softmax(dim=-1).mul(self.atoms).sum(-1)

    def calc_gmm_q(self, st):
        q_mean, _, q_weights = self.forward_gmm(st)
        return q_mean.mul(q_weights).sum(-1)

    def calc_fqf_q(self, st):
        if st.ndim == 4:
            convs = self.convs(st)
        else:
            convs = st
        # taus: B X (N+1) X 1, taus_hats: B X N X 1
        taus, tau_hats, _ = self.taus_prop(convs.detach())
        q_hats, _ = self.forward_iqr(convs, taus=tau_hats)
        q = ((taus[:, 1:, :] - taus[:, :-1, :]) * q_hats).sum(dim=1)
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
        return taus.unsqueeze(-1), taus_hat.unsqueeze(-1), entropies

    # noinspection PyArgumentList
    def feature_embed(self, x, taus, n):
        batch_size = x.size(0)
        if taus is None:
            taus = torch.rand(batch_size, n, 1).to(x)
        else:
            n = taus.size(1)

        ipi = np.pi * torch.arange(1, self.cfg.num_cosines + 1).to(x).view(
            1, 1, self.cfg.num_cosines
        )
        cosine = ipi.mul(taus).cos().view(batch_size * n, self.cfg.num_cosines)

        tau_embed = self.cosine_emb(cosine).view(batch_size, n, -1)
        state_embed = x.view(batch_size, 1, -1)
        features = (tau_embed * state_embed).view(batch_size * n, -1)
        return features, taus, n

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class GMMLinear(nn.Module, ABC):
    def __init__(
        self, in_features, out_features, max_std=3.0, num_gmm=5, num_samples=3
    ):
        super(GMMLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.N = num_gmm
        self.max_std = max_std
        self.num_samples = 3

        in_features += 1
        self.mean = nn.Parameter(
            torch.randn((in_features * out_features, num_gmm)), requires_grad=True
        )
        self.logstd = nn.Parameter(
            torch.randn((in_features * out_features, num_gmm)), requires_grad=True
        )
        self.weight = nn.Parameter(
            torch.randn((in_features * out_features, num_gmm)), requires_grad=True
        )

    def forward(self, x):
        comp = Normal(self.mean, self.logstd.tanh().mul(self.max_std).exp())
        mix = Categorical(self.weight.softmax(dim=-1))
        gmm = MixtureSameFamily(mix, comp)

        weight_bias = gmm.sample()
        bias = weight_bias[: self.out_features].view(-1)
        weight = weight_bias[self.out_features :].view(
            self.out_features, self.in_features
        )
        return nn.functional.linear(x, weight, bias)


# noinspection PyArgumentList
class NoisyLinear(nn.Module, ABC):
    def __init__(self, in_features, out_features, std_init=0.4, noisy_layer_std=0.1):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.noisy_layer_std = noisy_layer_std
        self.weight_mu = nn.Parameter(
            torch.zeros((out_features, in_features)), requires_grad=True
        )
        self.weight_sigma = nn.Parameter(
            torch.zeros((out_features, in_features)), requires_grad=True
        )
        self.register_buffer("weight_epsilon", torch.zeros((out_features, in_features)))
        self.bias_mu = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.bias_sigma = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.register_buffer("noise_in", torch.zeros(in_features))
        self.register_buffer("noise_out_weight", torch.zeros(out_features))
        self.register_buffer("noise_out_bias", torch.zeros(out_features))

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

        self.weight_epsilon.copy_(
            self.transform_noise(self.noise_out_weight).ger(
                self.transform_noise(self.noise_in)
            )
        )
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    @staticmethod
    def transform_noise(x):
        return x.sign().mul(x.abs().sqrt())


if __name__ == "__main__":
    from agent0.common.utils import set_random_seed
    import numpy as np
    out = []
    for seed in range(30, 60):
        print(seed)
        set_random_seed(seed)
        x = torch.randn(512, 4, 84, 84)
        model = DeepQNet(4, algo="iqr")
        q, taus = model.forward_iqr(model.convs(x), taus=None, n=32)
        out.append([q.mean().item(), taus.sum().item()])
    print(np.array(out).mean(axis=0))
    print(np.array(out).std(axis=0))
    # y = model.calc_c51_q(x)
    # print(
    #     model.forward(x).softmax(dim=-1).mul(model.atoms).sum(-1),
    # )
