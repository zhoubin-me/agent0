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
    def __init__(self, replay_size, frame_stack=4, n_step=3, discount=0.99):
        self.replay_size = replay_size
        self.n_step = n_step
        self.frame_stack = frame_stack
        self.discount = discount
        self.data = []
        self.lens = []
        self.lens_cum_sum = None

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        ep_idx = np.searchsorted(self.lens_cum_sum, idx, side='right')
        if ep_idx == 0:
            transit_idx = idx
        else:
            transit_idx = idx - self.lens_cum_sum[ep_idx - 1]

        transit_idx = max(transit_idx, self.frame_stack - 1)
        transit_idx_next = transit_idx + self.n_step
        transit_idx_next = min(transit_idx_next, self.lens[ep_idx] - 1)

        ep_transitions = self.data[ep_idx]['transits']
        obs, action, reward, done = ep_transitions[transit_idx]

        st = [x[0] for x in ep_transitions[transit_idx - 3:transit_idx + 1]]
        st_next = [x[0] for x in ep_transitions[transit_idx_next - 3:transit_idx_next + 1]]
        rs = [x[2] for x in ep_transitions[transit_idx:transit_idx_next]]
        rx = 0
        for r in reversed(rs):
            rx = rx * self.discount + r
        assert len(st) == self.frame_stack
        assert len(st) == self.frame_stack

        st = np.concatenate(st, axis=-1).transpose((2, 0, 1))
        st_next = np.concatenate(st_next, axis=-1).transpose((2, 0, 1))

        return st, action, rx, done, st_next

    def extend(self, transitions):
        self.data.extend(transitions)
        self.lens.extend([x['ep_len'] for x in transitions])

        while sum(self.lens) > self.replay_size:
            self.data.pop(0)
            self.lens.pop(0)

        self.lens_cum_sum = np.cumsum(self.lens)


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
