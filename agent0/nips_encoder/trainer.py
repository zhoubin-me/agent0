from abc import ABC
from collections import deque
from dataclasses import dataclass

import gym
import numpy as np
import ray
import torch
import torch.nn.functional as fx
import torchvision as tv
from lz4.block import compress, decompress
from ray import tune
from torch.utils.data import Dataset
from tqdm import tqdm

from agent0.common.atari_wrappers import make_atari
from agent0.common.utils import DataLoaderX, DataPrefetcher
from agent0.common.vec_env import ShmemVecEnv
from agent0.nips_encoder.model import ModelEncoder


@dataclass
class Config:
    game: str = "Breakout"
    epochs: int = 100
    batch_size: int = 64
    num_envs: int = 32
    num_actors: int = 16
    replay_size: int = 30000
    adam_lr: float = 1e-4
    num_data_workers: int = 4
    pin_memory: bool = True
    sha: str = ""
    restore_checkpoint: str = ""


class EncoderDataset(Dataset):
    def __init__(self, data, state_shape):
        self.data = data
        self.state_shape = state_shape

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        st, at, rt, dt = self.data[idx]
        st = np.frombuffer(decompress(st), dtype=np.uint8).reshape(*self.state_shape)
        if dt:
            st_next = self.data[idx][0]
        else:
            st_next = self.data[idx + 1][0]
        st_next = np.frombuffer(decompress(st_next), dtype=np.uint8).reshape(
            *self.state_shape
        )
        return np.array(st), at, rt, dt, np.array(st_next)


@ray.remote
def sample(cfg):
    envs = ShmemVecEnv(
        [lambda: make_atari(f"{cfg.game}NoFrameskip-v4") for _ in range(cfg.num_envs)],
        context="fork",
    )
    action_dim = envs.action_space.n
    print("Sampling replay")
    obs = envs.reset()
    steps = int(cfg.replay_size) // (cfg.num_envs * cfg.num_actors) + 1
    replay = []
    for _ in tqdm(range(steps)):
        action_random = np.random.randint(0, action_dim, cfg.num_envs)
        obs_next, reward, done, info = envs.step(action_random)
        # replay.append((obs, action_random, reward, done))
        for st, at, rt, dt in zip(obs, action_random, reward, done):
            replay.append((compress(st), at, rt, dt))
        obs = obs_next
    envs.close()
    return replay


class Trainer(tune.Trainable, ABC):
    def __init__(self, config=None, logger_creator=None):
        self.cfg = None
        self.device = None
        self.env = None
        self.obs_shape = None
        self.action_dim = None
        self.model = None
        self.optimizer = None
        self.optimizer_fn = None
        self.replay = None
        self.envs = None
        self.data_fetcher = None

        super(Trainer, self).__init__(config, logger_creator)

    def setup(self, config):
        self.cfg = Config(**config)

        self.device = torch.device("cuda:0")
        self.env = gym.make(f"{self.cfg.game}NoFrameskip-v4")
        self.obs_shape = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

        self.model = ModelEncoder(self.action_dim).to(self.device)
        self.replay = deque(maxlen=self.cfg.replay_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.cfg.adam_lr)

        if self.cfg.num_actors == 1:
            self.replay = sample(self.cfg)
        else:
            sample_ops = [sample.remote(self.cfg) for _ in range(self.cfg.num_actors)]
            datas = ray.get(sample_ops)
            for data in datas:
                self.replay.extend(data)

    def get_data_fetcher(self):
        dataset = EncoderDataset(self.replay, self.obs_shape)
        data_loader = DataLoaderX(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_data_workers,
            pin_memory=self.cfg.pin_memory,
        )

        data_fetcher = DataPrefetcher(data_loader, self.device)
        return data_fetcher

    def step(self):
        try:
            data = self.data_fetcher.next()
        except (StopIteration, AttributeError):
            self.data_fetcher = self.get_data_fetcher()
            data = self.data_fetcher.next()

        states, actions, rewards, dones, next_states = data
        states = states.float().div(255.0).permute(0, 3, 1, 2)
        next_states = next_states.float().div(255.0).permute(0, 3, 1, 2)
        actions = actions.long()

        states_pred = self.model(states, actions)

        loss = fx.mse_loss(states_pred, next_states)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        result = dict(
            loss=loss.item(),
            adam_lr=self.cfg.adam_lr,
            epoch=(self.training_iteration * self.cfg.batch_size)
            // self.cfg.replay_size,
            speed=self.training_iteration
            * self.cfg.batch_size
            / (self._time_total + 1),
            time_past=self._time_total,
            time_remain=(
                self.cfg.epochs * self.cfg.replay_size
                - self.training_iteration * self.cfg.batch_size
            )
            / (
                (self.training_iteration * self.cfg.batch_size + 1)
                / (self._time_total + 1)
            ),
        )
        return result

    def cleanup(self):
        epoch = (self._iteration * self.cfg.batch_size) / len(self.replay)
        if epoch >= self.cfg.epochs:
            torch.save({"model": self.model.state_dict()}, "./final.pth")

    def save_checkpoint(self, checkpoint_dir):
        try:
            data = self.data_fetcher.next()
        except (StopIteration, AttributeError):
            self.data_fetcher = self.get_data_fetcher()
            data = self.data_fetcher.next()

        states, actions, rewards, dones, next_states = data
        states = states.float().div(255.0).permute(0, 3, 1, 2)
        next_states = next_states.float().div(255.0).permute(0, 3, 1, 2)
        actions = actions.long()

        states_pred = self.model(states, actions)
        obs = torch.cat((next_states, states_pred), dim=-1)
        img = tv.utils.make_grid(obs).mul(255.0).permute(1, 2, 0).byte().numpy()

        return {
            "img": img,
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
        }

    def load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optim"])

    def reset_config(self, new_config):
        if "adam_lr" in new_config:
            self.cfg.adam_lr = new_config["adam_lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_config["adam_lr"]

        self.config = new_config
        return True
