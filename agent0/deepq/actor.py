import copy
import time

import numpy as np
import torch
from agent0.deepq.config import Config
from agent0.deepq.model import DeepQNet
from agent0.common.atari_wrappers import make_atari
from lz4.block import compress
from torch.distributions import Categorical

class Actor:
    def __init__(self, rank, **kwargs):

        self.rank = rank
        self.cfg = Config(**kwargs)
        self.cfg.update_atoms()
        self.envs = make_atari(self.cfg.game, self.cfg.num_envs)

        self.action_dim = self.envs.action_space[0].n
        self.device = torch.device('cuda:0')
        self.model = DeepQNet(self.action_dim, **kwargs).to(self.device)
        self.obs, _ = self.envs.reset()

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

    def sample(self, steps, epsilon, model, testing=False, test_episodes=20, render=False):
        self.model = model
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
            obs_next, reward, terminal, truncated, info = self.envs.step(action)
            if render:
                self.envs.render()
                time.sleep(0.001)

            if not testing:
                self.steps += self.cfg.num_envs
                done = np.logical_and(terminal, np.logical_not(truncated))
                for st, at, rt, dt, st_next in zip(self.obs, action, reward, done, obs_next):
                    data.append((compress(np.concatenate((st, st_next), axis=0)), at, rt, dt))

            self.obs = obs_next

            if 'final_info' in info:
                final_infos = info['final_info'][info['_final_info']]
                for stat in final_infos:
                    rs.append(stat['episode']['r'][0])
                    ep_len.append(stat['episode']['l'][0])


            if testing and (len(rs) > test_episodes or step > self.cfg.max_record_ep_len):
                break
            if not testing and step > steps:
                break

        toc = time.time()
        return copy.deepcopy(data), rs, qs, self.rank, len(data) / (toc - tic), copy.deepcopy(best_ep)

    def close_envs(self):
        self.envs.close()
