import time
from collections import deque

import numpy as np
import ray
import torch

from src.common.atari_wrappers import make_deepq_env
from src.common.vec_env import ShmemVecEnv
from src.deepq.config import Config
from src.deepq.model import NatureCNN


@ray.remote(num_gpus=0.1)
class Actor:
    def __init__(self, rank, **kwargs):

        self.rank = rank
        self.cfg = Config(**kwargs)
        # Training
        self.envs = ShmemVecEnv([lambda: make_deepq_env(self.cfg.game, True, True, False, False)
                                 for _ in range(self.cfg.num_envs)], context='fork')
        self.action_dim = self.envs.action_space.n
        self.state_shape = self.envs.observation_space.shape

        self.device = torch.device('cuda:0')
        if self.cfg.distributional:
            self.atoms = torch.linspace(self.cfg.v_min, self.cfg.v_max, self.cfg.num_atoms).to(self.device)

        self.model = NatureCNN(self.state_shape[0], self.action_dim, dueling=self.cfg.dueling,
                               noisy=self.cfg.noisy, num_atoms=self.cfg.num_atoms).to(self.device)
        self.obs = deque(maxlen=4)
        obs = self.envs.reset()
        for _ in range(4): self.obs.append(obs)

    def sample(self, steps, epsilon, state_dict, testing=False, test_episodes=10):
        self.model.load_state_dict(state_dict)
        replay = deque(maxlen=self.cfg.replay_size)
        rs, qss = [], []
        tic = time.time()
        step = 0
        while True:
            step += 1
            action_random = np.random.randint(0, self.action_dim, self.cfg.num_envs)
            if self.cfg.noisy and step % self.cfg.reset_noise_freq == 0:
                self.model.reset_noise()

            with torch.no_grad():
                st = torch.from_numpy(np.array(self.obs)).to(self.device).float().div(255.0).squeeze(-1).permute(1, 0,
                                                                                                                 2, 3)
                if self.cfg.distributional:
                    qs_prob = self.model(st).softmax(dim=-1)
                    qs = qs_prob.mul(self.atoms).sum(dim=-1)
                elif self.cfg.qr:
                    qs = self.model(st).mean(dim=-1)
                else:
                    qs = self.model(st).squeeze(-1)

            qs_max, qs_argmax = qs.max(dim=-1)
            action_greedy = qs_argmax.tolist()
            qss.append(qs_max.mean().item())

            if self.cfg.noisy:
                action = action_greedy
            else:
                action = [act_greedy if p > epsilon else act_random for p, act_random, act_greedy in
                          zip(np.random.rand(self.cfg.num_envs), action_random, action_greedy)]

            obs_next, reward, done, info = self.envs.step(action)
            frames = np.zeros((self.cfg.num_envs, self.state_shape[0] + 1, *self.state_shape[1:]), dtype=np.uint8)
            frames[:, :-1, :, :] = self.obs
            frames[:, -1, :, :] = obs_next[:, -1, :, :]
            if not testing:
                for entry in zip(frames, action, reward, done):
                    replay.append(entry)
            self.obs = obs_next

            for inf in info:
                if 'real_reward' in inf:
                    rs.append(inf['real_reward'])

            if testing and len(rs) > test_episodes:
                break
            if not testing and step > steps:
                break

        toc = time.time()
        return replay, rs, qss, self.rank, len(replay) / (toc - tic)

    def close_envs(self):
        self.envs.close()
