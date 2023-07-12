import argparse
import random

import numpy as np
import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val


class DataPrefetcher:
    def __init__(self, data_loader, device):
        self.data_loader = iter(data_loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.next_data = None
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.data_loader)
        except Exception as e:
            raise StopIteration

        # noinspection PyTypeChecker
        with torch.cuda.stream(self.stream):
            self.next_data = (
                x.to(self.device, non_blocking=True) for x in self.next_data
            )

    def next(self):
        # noinspection PyTypeChecker
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=3)


def parse_arguments(config):
    parser = argparse.ArgumentParser()
    for k, v in vars(config).items():
        if type(v) == bool:
            parser.add_argument(f"--{k}", dest=k, action="store_true")
            parser.add_argument(f"--no_{k}", dest=k, action="store_false")
            parser.set_defaults(**{k: v})
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(np.random.randint(int(1e6)))
