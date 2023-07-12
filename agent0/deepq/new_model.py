import torch
import torch.nn as nn
import torch.nn.functional as F

from agent0.deepq.new_config import ExpConfig, AlgoEnum, C51Config
from einops import rearrange

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

class DeepQHead(nn.Module):
    def __init__(self, act_dim: int, feat_dim: int, dueling: bool):
        super(DeepQHead, self).__init__()

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

        if self.value_head is None:
            return q
        else:
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

        self.q_head = nn.Linear(512, act_dim * cfg.atoms)
        self.q_head.apply(lambda m: init(m, 0.01))
        self.register_buffer(
            "atoms",
            torch.linspace(cfg.vmin, cfg.vmax, cfg.atoms),
        )
        self.atoms = self.atoms.view(1, 1, -1)

        self.delta = (cfg.vmax - cfg.vmin) / (cfg.atoms - 1)
        if dueling:
            self.value_head = nn.Linear(512, cfg.atoms)
            self.value_head.apply(lambda m: init(m, 1.0))
        else:
            self.value_head = None
        self.action_dim = act_dim

    def forward(self, x):
        x = F.relu(self.first_dense(x))
        q = self.q_head(x)
        q = rearrange(q, 'b (a n) -> b a n', a=self.action_dim)

        if self.value_head is None:
            return q
        else:
            value = self.value_head(x)
            value = rearrange(value, 'b n -> b 1 n')
            advantage = q - q.mean(dim=1, keepdim=True)
            q = value + advantage
            return q        

    def qval(self, x):
        q_dist = self.forward(x)
        return q_dist.softmax(dim=-1).mul(self.atoms).sum(dim=-1)


class DeepQNet(nn.Module):
    def __init__(self, cfg: ExpConfig):
        super(DeepQNet, self).__init__()
        self.encoder = ConvEncoder(cfg.obs_shape[0])

        dummy_x = torch.rand(1, *cfg.obs_shape)
        dummy_y = self.encoder(dummy_x)
        feat_dim = dummy_y.shape[-1]

        self.algo = cfg.learner.algo

        if self.algo == AlgoEnum.dqn:
            self.head = DeepQHead(
                cfg.action_dim, 
                feat_dim, 
                cfg.learner.dueling_head)
        elif self.algo == AlgoEnum.c51:
            self.head = C51Head(
                cfg.action_dim, 
                feat_dim, 
                cfg.learner.dueling_head, 
                cfg.learner.c51)
        else:
            raise NotImplementedError(self.algo)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

    def qval(self, x):
        x = self.encoder(x)
        return self.head.qval(x)
    

