#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import os
import random
import time
import cv2
import copy
import numpy as np
import collections
import matplotlib.pyplot as plt
import json
import scipy
import argparse
from PIL import Image
from collections import deque
from tqdm import tqdm
import ray
from scipy.signal import savgol_filter
# plt.style.use('')


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision as tv
from torch.utils.data import Dataset
import pickle

from src.common.atari_wrappers import wrap_deepmind, make_atari
from src.common.utils import LinearSchedule, DataLoaderX, DataPrefetcher, ReplayDataset
from src.common.vec_env import ShmemVecEnv, VecEnvWrapper, DummyVecEnv
from src.agents.model import NatureCNN


# In[2]:


num_env = 16
num_actors = 8
total_steps = int(1e7)
epoches = 1000
update_per_data = 8
replay_size = int(1e6)
discount = 0.99
batch_size = 512
lr = 1e-3
agent_train_freq = 20
target_net_update_freq = 250
exploration_ratio = 0.2
steps_per_epoch = total_steps // epoches


# In[3]:


def make_env(game, episode_life=True, clip_rewards=True):
    env = make_atari(f'{game}NoFrameskip-v4')
    env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=clip_rewards, frame_stack=True, scale=False, transpose_image=True)
    return env

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--game", type=str, default="Breakout")
    parser.add_argument("--replay_size", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=int, default=9)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--exploration_steps", type=int, default=20000)
    parser.add_argument("--max_step", type=int, default=int(1e7))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_env", type=int, default=16)

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args


# In[4]:


@ray.remote(num_gpus=0.125)
class Actor:
    def __init__(self, rank, game):
        if rank < num_actors:
            self.envs = ShmemVecEnv([lambda: make_env(game) for _ in range(num_env)], context='fork')
        else:
            self.envs = ShmemVecEnv([lambda: make_env(game, False, False) for _ in range(num_env)], context='fork')
        self.R = np.zeros(num_env)
        self.obs = self.envs.reset()
        self.state_shape, self.action_dim = self.envs.observation_space.shape, self.envs.action_space.n
        self.model = NatureCNN(self.state_shape[0], self.action_dim).cuda()
        self.rank = rank

    def sample(self, epsilon, state_dict):
        self.model.load_state_dict(state_dict)
        steps = steps_per_epoch // (num_env * num_actors)
        Rs, Qs = [], []
        tic = time.time()
        local_replay = deque(maxlen=replay_size)
        for step in range(steps):
            action_random = np.random.randint(0, self.action_dim, num_env)
            st = torch.from_numpy(np.array(self.obs)).float().cuda() / 255.0
            qs = self.model(st)
            qs_max, qs_argmax = qs.max(dim=-1)
            action_greedy = qs_argmax.tolist()
            Qs.append(qs_max.mean().item())
            action = [act_grd if p > epsilon else act_rnd for p, act_rnd, act_grd in zip(np.random.rand(num_env), action_random, action_greedy)]

            obs_next, reward, done, info = self.envs.step(action)
            for entry in zip(self.obs, action, reward, obs_next, done):
                local_replay.append(entry)
            self.obs = obs_next
            self.R += np.array(reward)
            for idx, d in enumerate(done):
                if d:
                    Rs.append(self.R[idx])
                    self.R[idx] = 0
        toc = time.time()
        # print(f"Rank {self.rank}, Data Collection Time: {toc - tic}, Speed {steps_per_epoch / (toc - tic)}")
        return local_replay, Rs, Qs, self.rank, toc - tic




# In[ ]:


class Agent:
    def __init__(self, game):
        test_env = make_env(game)
        self.state_shape, self.action_dim = test_env.observation_space.shape, test_env.action_space.n
        self.model = NatureCNN(self.state_shape[0], self.action_dim).cuda()
        self.model_target = copy.deepcopy(self.model).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.replay = deque(maxlen=replay_size)
        self.update_steps = 0
        self.device = torch.device('cuda:0')

    def get_datafetcher(self):
        dataset = ReplayDataset(self.replay)
        dataloader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        datafetcher = DataPrefetcher(dataloader, self.device)
        return datafetcher

    def append_data(self, data):
        self.replay.extend(data)

    def train_step(self):
        try:
            data = self.prefetcher.next()
        except:
            self.prefetcher = self.get_datafetcher()
            data = self.prefetcher.next()

        states, actions, rewards, next_states, terminals = data
        states = states.float() / 255.0
        next_states = next_states.float() / 255.0
        actions = actions.long()
        terminals = terminals.float()
        rewards = rewards.float()

        with torch.no_grad():
            q_next = self.model_target(next_states)
            q_next_online = self.model(next_states)
            q_next = q_next.gather(1, q_next_online.argmax(dim=-1).unsqueeze(-1)).squeeze(-1)
            q_target = rewards + discount * (1 - terminals) * q_next

        q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1

        if self.update_steps % 250 == 0:
            self.model_target.load_state_dict(self.model.state_dict())
        return loss.detach()


# In[ ]:


def formated_print(var_name, xs):
    if len(xs) > 0:
        print("{0} mean/std/max/min\t {1:12.6f}\t{2:12.6f}\t{3:12.6f}\t{4:12.6}".format(
            var_name, np.mean(xs), np.std(xs), np.max(xs), np.min(xs)))

def train(game):
    ray.init()
    epsilon_schedule = LinearSchedule(1.0, 0.01, int(total_steps * exploration_ratio))
    actors = [Actor.remote(rank, game) for rank in range(num_actors + 1)]
    tester = actors[-1]

    agent = Agent(game)
    sample_ops = [a.sample.remote(1.0, agent.model.state_dict()) for a in actors]

    TRRs, RRs, QQs, LLs, Sfps, Tfps = [], [], [], [], [], []
    for local_replay, Rs, Qs, rank, duration in ray.get(sample_ops):
        if rank < num_actors:
            agent.append_data(local_replay)
            RRs += Rs
            QQs += Qs
            Sfps += [len(local_replay) / duration]
        else:
            TRRs += Rs

    formated_print("Warming up Reward", RRs)
    formated_print("Warming up Qmax", QQs)

    steps = 0
    epoch = 0
    tic = time.time()
    while True:
        done_id, sample_ops = ray.wait(sample_ops)
        data = ray.get(done_id)
        local_replay, Rs, Qs, rank, duration = data[0]

        if rank < num_actors:
            # Actor
            agent.append_data(local_replay)
            steps += len(local_replay)
            epsilon = epsilon_schedule(len(local_replay))

            if epsilon == 0.01:
                epsilon=np.random.choice([0.01, 0.02, 0.05, 0.1], p=[0.7, 0.1, 0.1, 0.1])

            sample_ops.append(actors[rank].sample.remote(epsilon, agent.model.state_dict()))
            RRs += Rs
            QQs += Qs
        else:
            # Tester
            sample_ops.append(tester.sample.remote(0.01, agent.model.state_dict()))
            TRRs += Rs

        # Trainer
        ticc = time.time()
        for _ in range(agent_train_freq):
            loss = agent.train_step()
            LLs.append(loss)
        tocc = time.time()
        Tfps.append(len(replay_size) / (toc - tic))


        # Logging and saving
        if (steps // steps_per_epoch) > epoch:
            if epoch % 10 == 0:
                toc = time.time()
                print("=" * 100)
                print(f"Epoch: {epoch:5d}\t Steps: {steps:10d}\t Average Speed: {steps / (toc - tic):8.2f}\t Epsilon: {epsilon}")
                print('-' * 100)
                formated_print("EP Training Reward", RRs[-1000:])
                formated_print("EP Loss           ", torch.stack(LLs).tolist()[-1000:])
                formated_print("EP Qmax           ", QQs[-1000:])
                if len(TRRs) > 0:
                    formated_print("EP Test Reward    ", TRRs[-1000:])

                print("=" * 100)
                print(" " * 100)

            if epoch % 50 == 0:
                torch.save({
                    'model': agent.model.state_dict(),
                    'optim': agent.optimizer.state_dict(),
                    'epoch': epoch,
                    'epsilon': epsilon,
                    'steps': steps,
                    'Rs': RRs,
                    'TRs': TRRs,
                    'Qs': QQs,
                    'Ls': LLs,
                    'time': toc - tic,
                }, f'ckpt/{game}_e{epoch:04d}.pth')

            epoch += 1
            if epoch == 10:
                sample_ops.append(tester.sample.remote(0.01, agent.model.state_dict()))
            if epoch > epoches:
                ray.shutdown()
                return



# In[ ]:


if __name__ == '__main__':
    args = parse_arguments()
    game = args.game
    train(game)










