from abc import ABC

import torch.nn as nn
from torch.utils.data import Dataset


class ModelEncoder(nn.Module, ABC):
    def __init__(self, action_dim):
        super(ModelEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 8, 2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 6, 2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 6, 2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 10 * 7, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048)
        )

        self.action_embed = nn.Embedding(action_dim, 2048)

        self.linear_decoder = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.Linear(2048, 128 * 10 * 7),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 6, 2, 0, (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 6, 2, (0, 1), (0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 8, 2),
        )

    def decode(self, h):
        h = self.linear_decoder(h)
        h = self.decoder(h.view(-1, 128, 10, 7))
        return h

    def encode(self, x, a):
        sx = self.encoder(x)
        ax = self.action_embed(a)
        z = sx * ax
        return z

    def forward(self, states, actions):
        h = self.encode(states, actions)
        h = self.decode(h)
        return h


class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        st, at, stp1 = self.data[idx]
        return st, at, stp1
