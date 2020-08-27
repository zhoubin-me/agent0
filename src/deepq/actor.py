import copy
import time

import numpy as np
import ray
import torch
from lz4.block import compress

from src.common.atari_wrappers import make_deepq_env
from src.common.vec_env import ShmemVecEnv
from src.deepq.config import Config, GPU_SIZE
from src.deepq.model import NatureCNN


@ray.remote(num_gpus=0.1 / GPU_SIZE)
class Actor:
    def __init__(self, rank, **kwargs):

        self.rank = rank
        self.cfg = Config(**kwargs)
        # Training
        self.envs = ShmemVecEnv([lambda: make_deepq_env(game=self.cfg.game, episode_life=True, clip_rewards=True,
                                                        frame_stack=4, transpose_image=True, n_step=self.cfg.n_step,
                                                        discount=self.cfg.discount)
                                 for _ in range(self.cfg.num_envs)], context='fork')
        self.action_dim = self.envs.action_space.n
        self.device = torch.device('cuda:0')

        self.step = {
            'c51': lambda logits: logits.softmax(dim=-1).mul(self.atoms).sum(-1),
            'qr': lambda logits: logits.mean(-1),
            'dqn': lambda logits: logits.squeeze(-1),
            'mdqn': lambda logits: logits.squeeze(-1),
            'kl': lambda logits: logits.squeeze(-1),
        }
        assert self.cfg.algo in self.step

        if self.cfg.algo == 'c51':
            self.atoms = torch.linspace(self.cfg.v_min, self.cfg.v_max, self.cfg.num_atoms).to(self.device)
        self.model = NatureCNN(self.cfg.frame_stack, self.action_dim, dueling=self.cfg.dueling,
                               noisy=self.cfg.noisy, num_atoms=self.cfg.num_atoms).to(self.device)
        self.obs = self.envs.reset()

    def sample(self, steps, epsilon, state_dict, testing=False, test_episodes=10, render=False):
        self.model.load_state_dict(state_dict)
        rs, qs, data = [], [], []
        tic = time.time()
        step = 0
        while True:
            step += 1
            action_random = np.random.randint(0, self.action_dim, self.cfg.num_envs)
            if self.cfg.noisy and step % self.cfg.reset_noise_freq == 0:
                self.model.reset_noise()

            with torch.no_grad():
                logits = self.model(torch.from_numpy(self.obs).to(self.device).float().div(255.0))
                qt = self.step[self.cfg.algo](logits)

            qt_max, qt_arg_max = qt.max(dim=-1)
            action_greedy = qt_arg_max.tolist()
            qs.append(qt_max.mean().item())

            if self.cfg.noisy:
                action = action_greedy
            else:
                action = [act_greedy if p > epsilon else act_random for p, act_random, act_greedy in
                          zip(np.random.rand(self.cfg.num_envs), action_random, action_greedy)]

            obs_next, reward, done, info = self.envs.step(action)
            if render:
                self.envs.render()
                time.sleep(0.05)

            if not testing:
                if self.cfg.n_step > 1:
                    for inf, st_next in zip(info, obs_next):
                        st = inf['prev_obs']
                        at = inf['prev_action']
                        rt = inf['prev_reward']
                        dt = inf['prev_done']
                        data.append((compress(st), at, rt, dt, compress(st_next)))
                else:
                    for st, at, rt, dt, st_next in zip(self.obs, action, reward, done, obs_next):
                        data.append((compress(st), at, rt, dt, compress(st_next)))

            self.obs = obs_next

            for inf in info:
                if 'real_reward' in inf:
                    rs.append(inf['real_reward'])
                    if render:
                        print(rs[-1], len(rs), np.mean(rs), np.max(rs))

            if testing and len(rs) > test_episodes:
                break
            if not testing and step > steps:
                break

        toc = time.time()
        return copy.deepcopy(data), rs, qs, self.rank, len(data) / (toc - tic)

    def close_envs(self):
        self.envs.close()
