import copy
import time
from collections import deque, defaultdict

import numpy as np
import ray
import torch

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
        self.envs = ShmemVecEnv([lambda: make_deepq_env(self.cfg.game, True, True, False, False)
                                 for _ in range(self.cfg.num_envs)], context='fork')
        self.action_dim = self.envs.action_space.n
        self.state_shape = self.envs.observation_space.shape

        self.device = torch.device('cuda:0')
        if self.cfg.algo == 'c51':
            self.atoms = torch.linspace(self.cfg.v_min, self.cfg.v_max, self.cfg.num_atoms).to(self.device)

        self.step = {
            'c51': lambda logits: logits.softmax(dim=-1).mul(self.atoms).sum(-1),
            'qr': lambda logits: logits.mean(-1),
            'dqn': lambda logits: logits.squeeze(-1)
        }

        self.model = NatureCNN(self.cfg.frame_stack, self.action_dim, dueling=self.cfg.dueling,
                               noisy=self.cfg.noisy, num_atoms=self.cfg.num_atoms).to(self.device)
        self.episodic_buffer = defaultdict(list)
        self.obs = self.envs.reset()
        self.st = deque(maxlen=self.cfg.frame_stack)
        for _ in range(self.cfg.frame_stack):
            self.st.append(self.obs)

    def sample(self, steps, epsilon, state_dict, testing=False, test_episodes=10):
        self.model.load_state_dict(state_dict)
        replay = []
        rs, qss = [], []
        tic = time.time()
        step = 0
        while True:
            step += 1
            action_random = np.random.randint(0, self.action_dim, self.cfg.num_envs)
            if self.cfg.noisy and step % self.cfg.reset_noise_freq == 0:
                self.model.reset_noise()

            with torch.no_grad():
                st = np.concatenate(self.st, axis=-1)
                st = torch.from_numpy(st).to(self.device).float().div(255.0).permute(0, 3, 1, 2)
                logits = self.model(st)
                qs = self.step[self.cfg.algo](logits)

            qs_max, qs_arg_max = qs.max(dim=-1)
            action_greedy = qs_arg_max.tolist()
            qss.append(qs_max.mean().item())

            if self.cfg.noisy:
                action = action_greedy
            else:
                action = [act_greedy if p > epsilon else act_random for p, act_random, act_greedy in
                          zip(np.random.rand(self.cfg.num_envs), action_random, action_greedy)]

            obs_next, reward, done, info = self.envs.step(action)
            self.st.append(obs_next)
            for i in range(self.cfg.num_envs):
                self.episodic_buffer[i].append((self.obs[i], action[i], reward[i], done[i]))
                if done[i]:
                    if not testing and len(self.episodic_buffer[i]) > self.cfg.frame_stack:
                        replay.append(
                            dict(transits=copy.deepcopy(self.episodic_buffer[i]),
                                 ep_rew=sum([x[2] for x in self.episodic_buffer[i]]),
                                 ep_len=len(self.episodic_buffer[i]))
                        )
                        del self.episodic_buffer[i]

                    for j in range(self.cfg.frame_stack):
                        self.st[j][i] = self.st[-1][i]

                inf = info[i]
                if 'real_reward' in inf:
                    rs.append(inf['real_reward'])

            self.obs = obs_next

            if testing and len(rs) > test_episodes:
                for i in range(self.cfg.num_envs):
                    if len(self.episodic_buffer[i]) > self.cfg.max_record_ep_len:
                        replay.append(
                            dict(transits=copy.deepcopy(self.episodic_buffer[i]),
                                 ep_rew=sum([x[2] for x in self.episodic_buffer[i]]),
                                 ep_len=len(self.episodic_buffer[i]))
                        )
                        del self.episodic_buffer[i]
                break
            if not testing and step > steps:
                break

        toc = time.time()
        return replay, rs, qss, self.rank, len(replay) / (toc - tic)

    def close_envs(self):
        self.envs.close()
