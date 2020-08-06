import torch.nn as nn


def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)


class NatureCNN(nn.Module):
    def __init__(self, in_channels, action_dim, dueling=True):
        super(NatureCNN, self).__init__()

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