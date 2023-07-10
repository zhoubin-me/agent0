import torch
import torch.nn as nn


def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)


def init_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class DeepQNet(nn.Module):
    def __init__(self, action_dim, chan_dim):
        super(DeepQNet, self).__init__()

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

        self.first_dense = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU())
        self.first_dense.apply(lambda m: init(m, nn.init.calculate_gain("relu")))

        self.p = nn.Linear(512, action_dim)
        self.p.apply(lambda m: init(m, 0.01))

    def forward(self, x):
        x = self.convs(x)
        x = self.first_dense(x)
        x = self.p(x)
        return x
