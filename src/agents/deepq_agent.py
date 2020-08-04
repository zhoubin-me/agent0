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

from src.common.vec_env import ShmemVecEnv
from src.common.utils import LinearSchedule, DataPrefetcher, ReplayDataset, DataLoaderX, pprint, make_env
from src.agents.model import NatureCNN


def default_hyperparams():
    params = dict(
        env_id='Breakout',

        num_actors=8,
        num_envs=8,
        num_data_workers=4,

        adam_lr=1e-3,

        batch_size=512,
        discount=0.99,
        replay_size=int(1e6),
        exploration_ratio=0.15,

        target_update_freq=250,
        agent_train_freq=10,

        total_steps=int(2e7),
        epoches=1000,
        random_seed=1234)

    return params

@ray.remote(num_gpus=0.125)
class Actor:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.setup(**kwargs)

    def setup(self, **kwargs):
        if not hasattr(self, 'env_id'):
            kwargs_default = default_hyperparams()
            for k, v in kwargs_default.items():
                if not hasattr(self, k):
                    setattr(self, k, v)

        if self.rank == self.num_actors:
            # Testing
            self.envs = ShmemVecEnv([lambda: make_env(self.env_id, False, False) for _ in range(self.num_envs)])
        else:
            # Training
            self.envs = ShmemVecEnv([lambda: make_env(self.env_id, True, True) for _ in range(self.num_envs)])
        self.action_dim = self.envs.action_space.n
        self.state_shape = self.envs.observation_space.shape

        self.device = torch.device('cuda:0')
        self.model = NatureCNN(self.state_shape[0], self.action_dim).to(self.device)

        self.R = np.zeros(self.num_envs)
        self.obs = self.envs.reset()

    def sample(self, steps, epsilon, state_dict):
        self.model.load_state_dict(state_dict)
        replay = deque(maxlen=self.replay_size)
        Rs, Qs = [], []
        tic = time.time()
        for _ in range(steps):
            action_random = np.random.randint(0, self.action_dim, self.num_envs)
            st = torch.from_numpy(np.array(self.obs)).float().div(255.0).to(self.device)
            qs = self.model(st)
            qs_max, qs_argmax = qs.max(dim=-1)
            action_greedy = qs_argmax.tolist()
            Qs.append(qs_max.mean().item())
            action = [act_greedy if p > epsilon else act_random for p, act_greedy, act_random in
                      zip(np.random.rand(self.num_envs), action_random, action_greedy)]

            obs_next, reward, done, info = self.envs.step(action)
            for entry in zip(self.obs, action, reward, obs_next, done):
                replay.append(entry)
            self.obs = obs_next
            self.R += np.array(reward)
            for idx, d in enumerate(done):
                if d:
                    Rs.append(self.R[idx])
                    self.R[idx] = 0
        toc = time.time()
        return replay, Rs, Qs, self.rank, len(replay) / (toc - tic)

class Agent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.setup(**kwargs)

    def setup(self, **kwargs):
        if not hasattr(self, 'env_id'):
            kwargs_default = default_hyperparams()
            for k, v in kwargs_default.items():
                if not hasattr(self, k):
                    setattr(self, k, v)

        # neptune.init('zhoubinxyz/agentzero')
        # neptune.create_experiment(name=self.env_id, params=vars(self))
        params = vars(self)
        print("input args:\n", json.dumps(params, indent=4, separators=(",", ":")))
        self.vars = params

        self.envs = make_env(self.env_id)
        self.action_dim = self.envs.action_space.n
        self.state_shape = self.envs.observation_space.shape


        self.device = torch.device('cuda:0')
        self.model = NatureCNN(self.state_shape[0], self.action_dim).to(self.device)
        self.model_target = NatureCNN(self.state_shape[0], self.action_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.adam_lr)
        self.update_count = 0
        self.replay = deque(maxlen=self.replay_size)

    def get_datafetcher(self):
        dataset = ReplayDataset(self.replay)
        dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_data_workers, pin_memory=True)
        datafetcher = DataPrefetcher(dataloader, self.device)
        return datafetcher

    def train_step(self):
        try:
            data = self.prefetcher.next()
        except:
            self.prefetcher = self.get_datafetcher()
            data = self.prefetcher.next()

        states, actions, rewards, next_states, terminals = data
        states = states.float().div(255.0)
        next_states = next_states.float().div(255.0)
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

        if self.update_steps % self.target_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
        return loss.detach()

    def load(self):
        pass


    def run(self):
        epsilon_schedule = LinearSchedule(1.0, 0.01, int(self.total_steps * self.exploration_ratio))
        actors = [Actor.remote(rank=rank, **self.vars) for rank in range(self.num_actors + 1)]
        tester = actors[-1]

        steps_per_epoch = self.total_steps // self.epoches
        actor_steps = steps_per_epoch // (self.num_envs * self.num_actors)

        # Warming Up
        sample_ops = [a.sample.remote(actor_steps, 1.0, self.model.state_dict()) for a in actors]
        RRs, QQs, TRRs, LLs = [], [], [], []
        for local_replay, Rs, Qs, rank, fps in ray.get(sample_ops):
            if rank < self.num_actors:
                self.replay.extend(local_replay)
                RRs += Rs
                QQs += Qs
            else:
                TRRs += Rs
        pprint("Warming up Reward", RRs)
        pprint("Warming up Qmax", QQs)

        actor_fps, training_fps, iteration_fps, iteration_time, training_time = [], [], [], []
        epoch, steps = 0, 0
        tic = time.time()
        while True:
            # Sample data
            sampler_tic = time.time()
            done_id, sample_ops = ray.wait(sample_ops)
            data = ray.get(done_id)
            local_replay, Rs, Qs, rank, fps = data[0]
            if rank < self.num_actors:
                # Actors
                self.replay.extend(local_replay)
                epsilon = epsilon_schedule(len(local_replay))
                if epsilon == 0.01:
                    epsilon = np.random.choice([0.01, 0.02, 0.05, 0.1], p=[0.7, 0.1, 0.1, 0.1])
                sample_ops.append(actors[rank].sample.remote(actor_steps, epsilon, self.model.state_dict()))
                RRs += Rs
                QQs += Qs
                steps += len(local_replay)
                actor_fps.append(fps)
            else:
                # Tester
                sample_ops.append(tester.sample.remote(actor_steps, 0.01, self.model.state_dict()))
                TRRs += Rs

            # Trainer
            trainer_tic = time.time()
            Ls = [self.train_step() for _ in range(self.agent_train_freq)]
            Ls = torch.stack(Ls).tolist()
            LLs += Ls


            toc = time.time()
            training_fps += [(self.batch_size * self.agent_train_freq) / (toc - trainer_tic)]
            iteration_fps += [len(local_replay) / (toc - sampler_tic)]
            iteration_time += [toc - sampler_tic]
            training_time += [toc - trainer_tic]
            # Logging and saving
            if (steps // steps_per_epoch) > epoch:
                epoch += 1

                # Start Testing at Epoch 10
                if epoch == 10:
                    sample_ops.append(tester.sample.remote(actor_steps, 0.01, self.model.state_dict()))

                # Logging every 10 epocoh
                if epoch % 10 == 1:
                    speed = steps / (toc - tic)
                    print("=" * 100)
                    print(f"Epoch:{epoch:4d}\t Steps:{steps:8d}\t "
                          f"Updates:{self.update_steps:4d}\t "
                          f"TimePast(min):{(toc - tic) / 60:5.2}\t "
                          f"EstTimeRem(min):{(self.total_steps - steps) / speed / 60:8.2f}\t "
                          f"AvgSpeedFPS:{speed:8.2f}\t "
                          f"Epsilon:{epsilon:6.2}")
                    print('-' * 100)

                    pprint("Training Reward   ", RRs[-1000:])
                    pprint("Loss              ", LLs[-1000:])
                    pprint("Qmax              ", QQs[-1000:])
                    pprint("Test Reward       ", TRRs[-1000:])
                    pprint("Training Speed    ", training_fps[-20:])
                    pprint("Training Time     ", training_time[-20:])
                    pprint("Iteration Time    ", iteration_time[-20:])
                    pprint("Iteration FPS     ", iteration_fps[-20:])
                    pprint("Actor FPS         ", actor_fps[-20:])

                    print("=" * 100)
                    print(" " * 100)

                if epoch % 50 == 1:
                    torch.save({
                        'model': self.model.state_dict(),
                        'optim': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'epsilon': epsilon,
                        'steps': steps,
                        'Rs': RRs,
                        'TRs': TRRs,
                        'Qs': QQs,
                        'Ls': LLs,
                        'time': toc - tic,
                        'vars': self.vars,
                    }, f'ckpt/{self.env_id}_e{epoch:04d}.pth')


                if epoch > self.epoches:
                    print("Final Testing")
                    TRs = [x[1] for x in ray.get(tester.sample.remote(actor_steps * 10, 0.01, self.model.state_dict()))]
                    torch.save({
                        'model': self.model.state_dict(),
                        'optim': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'epsilon': epsilon,
                        'steps': steps,
                        'Rs': RRs,
                        'TRs': TRRs,
                        'Qs': QQs,
                        'Ls': LLs,
                        'time': toc - tic,
                        'FinalTestReward': TRs
                    }, f'ckpt/{self.env_id}_final.pth')
                    ray.shutdown()
                    return
