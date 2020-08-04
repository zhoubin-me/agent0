import torch
import numpy as np
from prefetch_generator import BackgroundGenerator
from torch.utils.data import Dataset, DataLoader
from src.common.atari_wrappers import wrap_deepmind, make_atari

class DataPrefetcher:
    def __init__(self, dataloader, device):
        self.dataloader = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.dataloader)
        except:
            raise StopIteration

        with torch.cuda.stream(self.stream):
            self.next_data = (x.to(self.device, non_blocking=True) for x in self.next_data)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

class ReplayDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        S, A, R, NS, T = self.data[idx]
        return np.array(S), A, R, np.array(NS), T

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


def make_env(game, episode_life=True, clip_rewards=True):
    env = make_atari(f'{game}NoFrameskip-v4')
    env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=clip_rewards, frame_stack=True, scale=False, transpose_image=True)
    return env


def pprint(var_name, xs):
    if len(xs) > 0:
        print("{0} mean/std/max/min\t {1:12.6f}\t{2:12.6f}\t{3:12.6f}\t{4:12.6}".format(
            var_name, np.mean(xs), np.std(xs), np.max(xs), np.min(xs)))