import ray
import time
import numpy as np
from collections import deque
import neptune
from tqdm import tqdm
from functools import reduce
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import copy

from src.common.vec_env import ShmemVecEnv
from src.common.utils import LinearSchedule, DataPrefetcher, ReplayDataset, DataLoaderX, pprint, make_env
from src.agents.model import NatureCNN


def default_hyperparams():
    params = dict(
        game='Breakout',

        num_actors=8,
        num_envs=16,
        num_data_workers=4,

        adam_lr=1e-3,

        batch_size=512,
        discount=0.99,
        replay_size=int(1e6),
        exploration_ratio=0.15,

        target_net_update_freq=500,
        agent_train_freq=20,

        total_steps=int(2e7),
        epoches=1000,
        random_seed=1234)
    return params


@ray.remote(num_gpus=0.125)
class Actor:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.rank < self.num_actors:
            self.envs = ShmemVecEnv([lambda: make_env(self.game) for _ in range(self.num_envs)], context='fork')
        else:
            self.envs = ShmemVecEnv([lambda: make_env(self.game, False, False) for _ in range(self.num_envs)], context='fork')
        self.R = np.zeros(self.num_envs)
        self.obs = self.envs.reset()
        self.state_shape, self.action_dim = self.envs.observation_space.shape, self.envs.action_space.n
        self.model = NatureCNN(self.state_shape[0], self.action_dim).cuda()

    def sample(self, epsilon, state_dict):
        self.model.load_state_dict(state_dict)
        steps = self.steps_per_epoch // (self.num_envs * self.num_actors)
        Rs, Qs = [], []
        tic = time.time()
        local_replay = deque(maxlen=self.replay_size)
        for step in range(steps):
            action_random = np.random.randint(0, self.action_dim, self.num_envs)
            st = torch.from_numpy(np.array(self.obs)).float().cuda() / 255.0
            qs = self.model(st)
            qs_max, qs_argmax = qs.max(dim=-1)
            action_greedy = qs_argmax.tolist()
            Qs.append(qs_max.mean().item())
            action = [act_grd if p > epsilon else act_rnd for p, act_rnd, act_grd in zip(np.random.rand(self.num_envs), action_random, action_greedy)]

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
        return local_replay, Rs, Qs, self.rank, len(local_replay) / (toc - tic)

# In[ ]:
class Agent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        test_env = make_env(self.game)
        self.state_shape, self.action_dim = test_env.observation_space.shape, test_env.action_space.n
        self.model = NatureCNN(self.state_shape[0], self.action_dim).cuda()
        self.model_target = copy.deepcopy(self.model).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.adam_lr)
        self.replay = deque(maxlen=self.replay_size)
        self.update_steps = 0
        self.device = torch.device('cuda:0')

    def get_datafetcher(self):
        dataset = ReplayDataset(self.replay)
        dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
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
            q_target = rewards + self.discount * (1 - terminals) * q_next

        q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1

        if self.update_steps % self.target_net_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
        return loss.detach()


# In[ ]:


def formated_print(var_name, xs):
    if len(xs) > 0:
        print("{0} mean/std/max/min\t {1:12.6f}\t{2:12.6f}\t{3:12.6f}\t{4:12.6}".format(
            var_name, np.mean(xs), np.std(xs), np.max(xs), np.min(xs)))

def train(**kwargs):

    ray.init(num_gpus=4)
    agent = Agent(**kwargs)

    epsilon_schedule = LinearSchedule(1.0, 0.01, int(agent.total_steps * agent.exploration_ratio))
    actors = [Actor.remote(rank=rank, **kwargs) for rank in range(agent.num_actors + 1)]
    tester = actors[-1]

    sample_ops = [a.sample.remote(1.0, agent.model.state_dict()) for a in actors]

    TRRs, RRs, QQs, LLs, Sfps, Tfps, Efps, Etime, Ttime = [], [], [], [], [], [], [], [], []
    for local_replay, Rs, Qs, rank, fps in ray.get(sample_ops):
        if rank < agent.num_actors:
            agent.append_data(local_replay)
            RRs += Rs
            QQs += Qs
            Sfps += [fps]
        else:
            TRRs += Rs

    formated_print("Warming up Reward", RRs)
    formated_print("Warming up Qmax", QQs)

    steps = 0
    epoch = 0
    tic = time.time()
    while True:
        ttic = time.time()

        done_id, sample_ops = ray.wait(sample_ops)
        data = ray.get(done_id)
        local_replay, Rs, Qs, rank, duration = data[0]

        if rank < agent.num_actors:
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
        Ls = []
        for _ in range(agent.agent_train_freq):
            loss = agent.train_step()
            Ls.append(loss)
        Ls = torch.stack(Ls).tolist()
        LLs += Ls
        tocc = time.time()
        Tfps.append((agent.batch_size * agent.agent_train_freq) / (tocc - ticc))
        Ttime.append(tocc - ticc)



        # Logging and saving
        if (steps // agent.steps_per_epoch) > epoch:
            if epoch % 10 == 0:
                toc = time.time()
                speed = steps / (toc - tic)

                print("=" * 100)
                print(f"Epoch:{epoch:4d}\t Steps:{steps:8d}\t "
                      f"Updates:{agent.update_steps:4d}\t "
                      f"TimePast(min):{(toc - tic) / 60:5.2}\t "
                      f"EstTimeRem(min):{(agent.total_steps - steps) / speed / 60:8.2f}\n"
                      f"AvgSpeedFPS:{speed:8.2f}\t "
                      f"Epsilon:{epsilon:6.2}")
                print('-' * 100)
                print('-' * 100)
                pprint("Training Reward   ", RRs[-1000:])
                pprint("Loss              ", LLs[-1000:])
                pprint("Qmax              ", QQs[-1000:])
                pprint("Test Reward       ", TRRs[-1000:])
                pprint("Training Speed    ", Tfps[-10:])
                pprint("Training Time     ", Ttime[-10:])
                pprint("Iteration Time    ", Etime[-10:])
                pprint("Iteration FPS     ", Efps[-10:])
                pprint("Actor FPS         ", Sfps[-10:])

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
                }, f'ckptx/{agent.game}_e{epoch:04d}.pth')

            epoch += 1
            if epoch == 10:
                sample_ops.append(tester.sample.remote(0.01, agent.model.state_dict()))


            if epoch > agent.epoches:
                print("Final Testing")
                sample_ops = [tester.sample.remote(0.01, agent.model.state_dict()) for _ in range(100)]
                TRs_final = []
                for local_replay, Rs, Qs, rank, fps in ray.get(sample_ops):
                    TRs_final += Rs

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
                    'FinalTestReward': TRs_final,
                }, f'ckptx/{agent.game}_final.pth')

                ray.shutdown()
                return

        ttoc = time.time()
        Etime.append(ttoc - ttic)
        Efps.append(len(local_replay) / (ttoc - ttic))












