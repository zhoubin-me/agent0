import copy
import time

import numpy as np
import torch
from agent0.common.atari_wrappers import make_deepq_env
from agent0.common.vec_env import ShmemVecEnv
from agent0.deepq.config import Config
from agent0.deepq.model import DeepQNet
from lz4.block import compress
from torch.distributions import Categorical


class Actor:
    def __init__(self, rank, **kwargs):

        self.rank = rank
        self.cfg = Config(**kwargs)
        self.cfg.update_atoms()
        # Training
        self.envs = ShmemVecEnv([lambda: make_deepq_env(game=self.cfg.game, episode_life=True, clip_rewards=True,
                                                        frame_stack=4, transpose_image=True, n_step=self.cfg.n_step,
                                                        discount=self.cfg.discount, state_count=False,
                                                        norm_reward=False, record_best_ep=self.cfg.best_ep,
                                                        gaussian_reward=False, seed=None)
                                 for _ in range(self.cfg.num_envs)], context='spawn')
        self.action_dim = self.envs.action_space.n
        self.device = torch.device('cuda:0')
        self.model = DeepQNet(self.action_dim, **kwargs).to(self.device)
        self.obs = self.envs.reset()

        self.act = {
            'epsilon_greedy': self.act_epsilon_greedy,
            'soft_explore': self.act_soft,
            'dltv': self.act_dltv,
        }

        self.steps = 1

    def act_epsilon_greedy(self, st, epsilon):
        qt = self.model.calc_q(st)
        action_random = np.random.randint(0, self.action_dim, self.cfg.num_envs)
        qt_max, qt_arg_max = qt.max(dim=-1)
        action_greedy = qt_arg_max.tolist()

        if self.cfg.noisy:
            action = action_greedy
        else:
            action = [act_greedy if p > epsilon else act_random for p, act_random, act_greedy in
                      zip(np.random.rand(self.cfg.num_envs), action_random, action_greedy)]

        return action, qt_max.mean().item()

    def act_soft(self, st, epsilon=None):
        qt = self.model.calc_q(st)
        # temperature = (epsilon + 0.0001) * 100
        temperature = 1
        dist = Categorical(logits=qt / temperature)
        action = dist.sample()
        qt_max = qt.gather(1, action.unsqueeze(-1))
        return action.tolist(), qt_max.mean().item()

    def act_dltv(self, st, epsilon=None):
        qt = self.model(st)
        qt_median, _ = qt.median(dim=-1, keepdim=True)
        qt_mean = qt.mean(dim=-1)
        sigma = (qt - qt_median).pow(2).mean(-1).sqrt()
        ct = 50 * np.log(self.steps) / self.steps
        action = (qt_mean + ct * sigma).argmax(-1)
        qt_max = qt_mean.gather(1, action.unsqueeze(-1))
        return action.tolist(), qt_max.mean().item()

    def sample(self, steps, epsilon, state_dict, testing=False, test_episodes=20, render=False):
        self.model.load_state_dict(state_dict)
        rs, qs, data, ep_len, best_ep = [], [], [], [], []
        tic = time.time()
        step = 0
        while True:
            step += 1
            if self.cfg.noisy and step % self.cfg.reset_noise_freq == 0:
                self.model.reset_noise()

            with torch.no_grad():
                st = torch.from_numpy(self.obs).to(self.device).float().div(255.0)
                action, qt_max = self.act[self.cfg.policy](st, epsilon)

            qs.append(qt_max)
            obs_next, reward, done, info = self.envs.step(action)
            if render:
                self.envs.render()
                time.sleep(0.001)

            if not testing:
                self.steps += self.cfg.num_envs
                if self.cfg.n_step > 1:
                    for inf, st_next in zip(info, obs_next):
                        st = inf['prev_obs']
                        at = inf['prev_action']
                        rt = inf['prev_reward']
                        dt = inf['prev_done']
                        if inf['prev_bad_transit']:
                            dt = not dt
                        data.append((compress(np.concatenate((st, st_next), axis=0)), at, rt, dt))
                else:
                    for st, at, rt, dt, st_next, inf in zip(self.obs, action, reward, done, obs_next, info):
                        if 'counter' in inf and dt:
                            # print('bad transit')
                            dt = not dt
                        data.append((compress(np.concatenate((st, st_next), axis=0)), at, rt, dt))

            self.obs = obs_next

            for inf in info:
                if not testing and 'best_ep' in inf:
                    best_ep.append(inf['best_ep'])
                if 'real_reward' in inf:
                    rs.append(inf['real_reward'])
                    ep_len.append(inf['steps'])
                    if render:
                        print(rs[-1], ep_len[-1], len(rs), np.mean(rs), np.max(rs), inf)

            if testing and (len(rs) > test_episodes or step > self.cfg.max_record_ep_len):
                break
            if not testing and step > steps:
                break

        toc = time.time()
        return copy.deepcopy(data), rs, qs, self.rank, len(data) / (toc - tic), copy.deepcopy(best_ep)

    def close_envs(self):
        self.envs.close()
