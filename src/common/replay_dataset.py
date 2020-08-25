import numpy as np
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
    def __init__(self, device, **kwargs):
        self.cfg = Config(**kwargs)

        self.data = []
        self.lens = []
        self.lens_cum_sum = [0]

        if self.cfg.prioritize:
            self.beta_schedule = LinearSchedule(self.cfg.priority_beta0, 1.0, self.cfg.total_steps)
            self.weights = torch.ones(self.cfg.replay_size * 2, requires_grad=False)
            self.beta = self.cfg.priority_beta0

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        idx = idx % len(self)
        ep_idx = np.searchsorted(self.lens_cum_sum, idx, side='right')
        if ep_idx == 0:
            transit_idx = idx
        else:
            transit_idx = idx - self.lens_cum_sum[ep_idx - 1]

        transit_idx = max(transit_idx, self.cfg.frame_stack - 1)
        transit_idx_next = transit_idx + self.cfg.n_step
        transit_idx_next = min(transit_idx_next, self.lens[ep_idx] - 1)

        ep_transitions = self.data[ep_idx]['transits']
        obs, action, reward, done = ep_transitions[transit_idx]

        st = [x[0] for x in ep_transitions[transit_idx - 3:transit_idx + 1]]
        st_next = [x[0] for x in ep_transitions[transit_idx_next - 3:transit_idx_next + 1]]
        rs = [x[2] for x in ep_transitions[transit_idx:transit_idx_next]]
        rx = 0
        for r in reversed(rs):
            rx = rx * self.cfg.discount + r
        assert len(st) == self.cfg.frame_stack
        assert len(st) == self.cfg.frame_stack

        st = np.concatenate(st, axis=-1).transpose((2, 0, 1))
        st_next = np.concatenate(st_next, axis=-1).transpose((2, 0, 1))
        weight = self.weights[idx].pow(-self.beta)

        return st, action, rx, done, st_next, 1.0, idx

    def __iter__(self):
        while True:
            yield torch.multinomial(self.weights[:len(self)], self.cfg.batch_size, False).tolist()

    def extend(self, transitions):
        self.data.extend(transitions)
        self.lens.extend([x['ep_len'] for x in transitions])

        in_frame_count = sum([x['ep_len'] for x in transitions])
        out_frame_count = 0
        while sum(self.lens) > self.cfg.replay_size:
            self.data.pop(0)
            out_frame_count += self.lens.pop(0)
        self.lens_cum_sum = np.cumsum(self.lens)

        if self.cfg.prioritize:
            self.weights.roll(-out_frame_count, 0)
            self.weights[sum(self.lens):] = 1.0
            self.beta_schedule(in_frame_count)

    def update_priorities(self, idxes, priorities):
        self.weights[idxes] = priorities.add(1e-8).exp(self.cfg.priority_alpha)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=3)
