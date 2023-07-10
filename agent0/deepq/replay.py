import copy
import random
from collections import deque

import numpy as np
import torch
from agent0.common.utils import LinearSchedule
from agent0.deepq.new_config import ExpConfig
from lz4.block import decompress
from torch.utils.data import Dataset, Sampler


class ReplayDataset(Dataset, Sampler):
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.replay_size = self.cfg.replay.size
        self.batch_size = self.cfg.learner.batch_size

        self.data = deque(maxlen=self.replay_size)
        self.top = 0
        self.prob = torch.ones(self.replay_size)

    def __len__(self):
        return self.top

    def __getitem__(self, idx):
        idx = idx % self.top

        frames, at, rt, dt = self.data[idx]
        frames = np.frombuffer(decompress(frames), dtype=np.uint8)


        return np.array(frames), at, rt, dt

    def __iter__(self):
        for _ in range(self.top // self.batch_size):
            yield torch.multinomial(self.prob[:self.top], self.batch_size, False).tolist()

    def extend(self, transitions):
        self.data.extend(transitions)
        num_entries = len(transitions)
        self.top = min(self.top + num_entries, self.replay_size)
