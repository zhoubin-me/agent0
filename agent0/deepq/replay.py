import copy
import random
from collections import deque

import numpy as np
import torch
from lz4.block import decompress
from torch.utils.data import Dataset, Sampler

from agent0.common.utils import LinearSchedule
from agent0.deepq.config import ExpConfig, ReplayEnum


class ReplayDataset(Dataset, Sampler):
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg

        self.data = deque(maxlen=cfg.replay.size)
        self.priority = torch.ones(cfg.replay.size)
        self.top = 0

        if self.cfg.replay.policy == ReplayEnum.prioritize:
            self.beta_schedule = LinearSchedule(self.cfg.replay.beta0, 1.0, self.cfg.trainer.total_steps)
            self.beta = self.cfg.replay.beta0
            self.max_p = 1.0

    def __len__(self):
        return self.top

    def __getitem__(self, idx):
        idx = idx % self.top
        frames, at, rt, dt = self.data[idx]
        frames = np.frombuffer(decompress(frames), dtype=np.uint8)
        priority = self.priority[idx]
        return np.array(frames), at, rt, dt, priority, idx

    def __iter__(self):
        for _ in range(self.top // self.cfg.learner.batch_size):
            yield torch.multinomial(
                self.priority[:self.top], self.cfg.learner.batch_size, False
            ).tolist()

    def extend(self, transitions):
        self.data.extend(transitions)
        num_entries = len(transitions)
        self.top = min(self.top + num_entries, self.cfg.replay.size)

        if self.cfg.replay.policy == ReplayEnum.prioritize:
            self.priority.roll(-num_entries, 0)
            self.priority[-num_entries:] = self.max_p ** self.cfg.replay.alpha
            self.beta = self.beta_schedule(num_entries)
    
    def update_priority(self, ids, priorities):
        self.priority[ids] = (priorities + self.cfg.replay.eps).pow(self.cfg.replay.alpha)
        self.max_p = max(priorities.max().item(), self.max_p)
