import torch
import torch.nn as nn
import torch.nn.functional as F

from agent0.deepq.new_config import ExpConfig, AlgoEnum, C51Config, QRConfig, IQNConfig
from einops import rearrange
import numpy as np

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

class QHead(nn.Module):
    def __init__(self, act_dim: int, feat_dim: int, dueling: bool):
        super(QHead, self).__init__()

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
        q = rearrange(q, 'b (a n) -> b a n', a=self.action_dim)

        if self.value_head is not None:
            value = self.value_head(x)
            value = rearrange(value, 'b n -> b 1 n')
            advantage = q - q.mean(dim=1, keepdim=True)
            q = value + advantage
        return q        

    def qval(self, x):
        q_dist = self.forward(x)
        return q_dist.softmax(dim=-1).mul(self.atoms).sum(dim=-1)


class QRHead(nn.Module):
    def __init__(self, act_dim: int, feat_dim: int, dueling: bool, cfg: QRConfig):
        super(QRHead, self).__init__()
        self.first_dense = nn.Linear(feat_dim, 512)
        self.first_dense.apply(lambda m: init(m, nn.init.calculate_gain("relu")))

        self.q_head = nn.Linear(512, act_dim * cfg.num_quantiles)
        self.q_head.apply(lambda m: init(m, 0.01))
        self.register_buffer(
            "cumulative_density",
            (2 * torch.arange(cfg.num_quantiles) + 1) / (2.0 * cfg.num_quantiles),
        )
        self.action_dim = act_dim
        if dueling:
            self.value_head = nn.Linear(512, cfg.num_quantiles)
            self.value_head.apply(lambda m: init(m, 1.0))
        else:
            self.value_head = None
        self.action_dim = act_dim


    def forward(self, x):
        x = F.relu(self.first_dense(x))
        q = self.q_head(x)
        q = rearrange(q, 'b (a n) -> b a n', a=self.action_dim)

        if self.value_head is not None:
            value = self.value_head(x)
            value = rearrange(value, 'b n -> b 1 n')
            advantage = q - q.mean(dim=1, keepdim=True)
            q = value + advantage
        return q

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

    def forward(self, x, n):
        # x: b d
        # featues: (b n) d 
        # taus: b n 1
        features, taus, n = self.feature_emb(x, n=n)
        features = F.relu(self.first_dense(features))
        # q: (b n) a
        q = self.q_head(features)

        if self.value_head is not None:
            value = self.value_head(x)
            advantage = q - q.mean(dim=-1, keepdim=True)
            q = value + advantage
        q = rearrange(q, '(b n) a -> b n a', n=n)
        return q, taus

    def feature_emb(self, x, n):
        batch_size = x.size(0)
        taus = torch.rand(batch_size, n, 1).to(x)

        ipi = np.pi * torch.arange(1, self.cfg.num_cosines + 1).to(x)
        ipi = rearrange(ipi, 'd -> 1 1 d')
        cosine = ipi.mul(taus).cos()
        cosine = rearrange(cosine, 'b n d -> (b n) d')

        tau_embed = self.cosine_emb(cosine)
        tau_embed = rearrange(tau_embed, '(b n) d -> b n d', b=batch_size)
        state_embed = rearrange(x, 'b d -> b 1 d')
        features = rearrange(tau_embed * state_embed, 'b n d -> (b n) d')
        return features, taus, n

    def qval(self, x, n=None):
        if n is None:
            n = self.cfg.K
        qs, _ = self.forward(x, n)
        return qs.mean(dim=1)

class DeepQNet(nn.Module):
    def __init__(self, cfg: ExpConfig):
        super(DeepQNet, self).__init__()
        self.encoder = ConvEncoder(cfg.obs_shape[0])

        dummy_x = torch.rand(1, *cfg.obs_shape)
        dummy_y = self.encoder(dummy_x)
        feat_dim = dummy_y.shape[-1]


        if cfg.learner.algo == AlgoEnum.dqn:
            self.head = QHead(
                cfg.action_dim, 
                feat_dim, 
                cfg.learner.dueling_head)
        elif cfg.learner.algo == AlgoEnum.c51:
            self.head = C51Head(
                cfg.action_dim, 
                feat_dim, 
                cfg.learner.dueling_head, 
                cfg.learner.c51)
        elif cfg.learner.algo == AlgoEnum.qr:
            self.head = QRHead(
                cfg.action_dim,
                feat_dim,
                cfg.learner.dueling_head,
                cfg.learner.qr
            )
        elif cfg.learner.algo == AlgoEnum.iqn:
            self.head = IQNHead(
                cfg.action_dim,
                feat_dim,
                cfg.learner.dueling_head,
                cfg.learner.iqn
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
    

