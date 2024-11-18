import json
import random

import numpy as np
import torch
import torch.nn as nn

class DataPrefetcher:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)
        self.stream = torch.cuda.Stream()
        self.next_data = None
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.data_iter)
        except Exception as e:
            self.data_iter = iter(self.data_loader)
            self.preload()

        with torch.cuda.stream(self.stream):
            self.next_data = (
                x.cuda(non_blocking=True) for x in self.next_data
            )

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data


def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(np.random.randint(int(1e6)))

