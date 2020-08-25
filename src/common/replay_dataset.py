from collections import deque

import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.data import Dataset, DataLoader, Sampler

from src.common.utils import LinearSchedule
from src.deepq.config import Config


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


class ReplayDataset(Dataset, Sampler):
    def __init__(self, **kwargs):
        self.cfg = Config(**kwargs)

        self.data = deque(maxlen=self.cfg.replay_size)
        self.top = 0

        if self.cfg.prioritize:
            self.beta_schedule = LinearSchedule(self.cfg.priority_beta0, 1.0, self.cfg.total_steps)
            self.prob = torch.ones(size=(self.cfg.replay_size * 2,), names='weights', requires_grad=False)
            self.beta = self.cfg.priority_beta0
            self.max_p = 1.0

    def __len__(self):
        return self.top

    def __getitem__(self, idx):
        idx = idx % len(self)

        st, at, rt, dt, st_next = self.data[idx]

        if self.cfg.prioritize:
            weight = self.prob[idx]
        else:
            weight = 1.0

        return st, at, rt, dt, st_next, weight, idx

    def __iter__(self):
        for _ in range(self.top // self.cfg.batch_size):
            yield torch.multinomial(self.prob[:len(self)], self.cfg.batch_size, False).tolist()

    def extend(self, transitions):
        self.data.extend(transitions)
        num_entries = len(transitions)
        self.top = min(self.top + num_entries, self.cfg.replay_size)

        if self.cfg.prioritize:
            self.prob.roll(-num_entries, 0)
            self.prob[-num_entries:] = self.max_p ** self.cfg.priority_alpha
            self.beta = self.beta_schedule(num_entries)

    def update_priorities(self, idxes, priorities):
        self.prob[idxes] = priorities.add(1e-8).pow(self.cfg.priority_alpha)
        self.max_p = max(priorities.max().item(), self.max_p)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=3)
