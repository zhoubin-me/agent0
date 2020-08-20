import random

import numpy as np
import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.data import Dataset, DataLoader


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
            print(e)
            raise StopIteration

        with torch.cuda.stream(self.stream):
            self.next_data = (x.to(self.device, non_blocking=True) for x in self.next_data)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data


class ReplayDataset(Dataset):
    def __init__(self, replay_size):
        self.replay_size = replay_size
        self.data = []
        self.lens_cumsum = None
        self.lens = None

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        ep_idx = np.searchsorted(self.lens_cumsum, idx, side='right')
        transit_idx = self.lens_cumsum[ep_idx] - idx
        transit_idx = np.clip(transit_idx, 4, self.lens[ep_idx])

        ep_transitions = self.data[ep_idx]
        obs, action, reward, done = ep_transitions[transit_idx]
        if not done:
            st = [x[0] for x in ep_transitions[transit_idx - 4:transit_idx + 1]]
        else:
            st = [x[0] for x in ep_transitions[transit_idx - 4:transit_idx]] + np.zeros_like(obs)
        st = np.concatenate(st, axis=-1).transpose((2, 0, 1))

        return st, action, reward, done

    def append(self, transitions):
        self.data.append(transitions)
        self.lens = [x['ep_len'] for x in self.data]

        while sum(self.lens) > self.replay_size:
            self.data.pop(0)
            self.lens.pop(0)

        self.lens_cumsum = np.cumsum(self.lens)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=3)


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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(np.random.randint(int(1e6)))
