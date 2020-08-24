import random

import numpy as np
import torch
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from prefetch_generator import BackgroundGenerator
from torch.utils.data import Dataset, DataLoader, Sampler, SequentialSampler

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

        self.data = []
        self.lens = []
        self.lens_cum_sum = [0]

        if self.cfg.prioritize:
            assert self.cfg.priority_alpha >= 0
            self._alpha = self.cfg.priority_alpha

            it_capacity = 1
            while it_capacity < self.cfg.replay_size:
                it_capacity *= 2

            self._it_sum = SumSegmentTree(it_capacity)
            self._it_min = MinSegmentTree(it_capacity)
            self._max_priority = 1.0
            self._idx_producer = SequentialSampler(range(self.cfg.total_steps * self.cfg.batch_size))
            self._beta_schedule = LinearSchedule(self.cfg.priority_beta0, 1.0, self.cfg.total_steps)
            self._beta = self.cfg.priority_beta0

    def __len__(self):
        return self.lens_cum_sum[-1]

    def __getitem__(self, idx):
        try:
            idx = idx % len(self)
            ep_idx = np.searchsorted(self.lens_cum_sum, idx, side='right')
            if ep_idx == 0:
                transit_idx = idx
            else:
                transit_idx = idx - self.lens_cum_sum[ep_idx - 1]

            transit_idx = max(transit_idx, self.cfg.frame_stack - 1)
            transit_idx_next = transit_idx + self.cfg.n_step
            transit_idx_next = min(transit_idx_next, self.lens[ep_idx] - 1)
        except:
            print("IDX Produced", idx)
            print("No. of Transits", len(self))
            print("EP idx", ep_idx)
            print("Total EP", len(self.lens))
            print("Cur EP Len", self.lens[ep_idx])
            print(transit_idx)
            print(transit_idx_next)

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

        if self.cfg.prioritize:
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * len(self)) ** (-self._beta)

            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-self._beta)
            weight = weight / max_weight
        else:
            weight = 1.0

        return st, action, rx, done, st_next, weight, idx

    def __iter__(self):
        batch = []
        for i in self._idx_producer:
            mas = (random.random() + i % self.cfg.batch_size) / self.cfg.batch_size * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mas)
            batch.append(idx)
            if len(batch) == self.cfg.batch_size:
                yield batch
                batch = []

    def extend(self, transitions):
        self.data = transitions + self.data
        self.lens = [x['ep_len'] for x in transitions] + self.lens

        in_frame_count = sum([x['ep_len'] for x in transitions])
        out_frame_count = 0
        while sum(self.lens) > self.cfg.replay_size:
            self.data.pop()
            out_frame_count += self.lens.pop()
        self.lens_cum_sum = np.cumsum(self.lens)

        if self.cfg.prioritize:
            for idx in range(len(self)):
                self._it_sum[idx] = self._it_sum[idx + out_frame_count]
                self._it_min[idx] = self._it_min[idx + out_frame_count]
                self._beta = self._beta_schedule()

            for idx in range(len(self) - in_frame_count, len(self)):
                self._it_sum[idx] = self._max_priority ** self._alpha
                self._it_min[idx] = self._max_priority ** self._alpha

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            priority += 1e-8
            assert 0 <= idx < self.lens_cum_sum[-1]
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=3)
