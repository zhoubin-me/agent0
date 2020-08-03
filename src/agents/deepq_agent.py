import ray
import time
import numpy as np
from collections import deque
import neptune
from tqdm import tqdm
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from src.common.vec_env import ShmemVecEnv, VecEnvWrapper, DummyVecEnv
from src.common.utils import LinearSchedule, DataLoader, DataPrefetcher, ReplayDataset, DataLoaderX
from src.common.atari_wrappers import make_atari, wrap_deepmind
from src.agents.model import NatureCNN


def default_hyperparams():
    params = dict(
        env_id='Breakout',
        num_actors=16,
        num_envs=16,
        num_data_workers=8,
        num_test_envs=8,
        gpu_id=0,
        adam_lr=1e-3,
        replay_size=int(1e6),
        batch_size=512,
        update_per_data=8,
        base_batch_size=32,
        discount=0.99,
        target_update_freq=10000,
        start_update_steps=20000,
        exploration_ratio=0.1,
        total_steps=int(1e7),
        epoches=100,
        random_seed=1234,
    )

    params.update(
        min_epsilons=np.random.choice([0.01, 0.02, 0.05, 0.1], size=params['num_actors'], p=[0.7, 0.1, 0.1, 0.1])
    )

    return params

def make_env(game, episode_life=True, clip_rewards=True):
    env = make_atari(f'{game}NoFrameskip-v4')
    env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=clip_rewards, frame_stack=True, scale=False, transpose_image=True)
    return env

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

        self.envs = ShmemVecEnv([lambda: make_env(self.env_id) for _ in range(self.num_envs)])
        self.action_dim = self.envs.action_space.n
        self.state_shape = self.envs.observation_space.shape

        self.memory_format = torch.channels_last
        self.device = torch.device('cpu')
        self.model = NatureCNN(self.state_shape[0], self.action_dim).to(self.device, memory_format=self.memory_format)

        self.min_epsilon = self.min_epsilons[self.rank]
        self.epsilon_schedule = LinearSchedule(1.0, self.min_epsilon, int(self.epoches * self.exploration_ratio))

        self.steps = 0
        self.R = np.zeros(self.num_envs)
        self.obs = self.envs.reset()

    def load_model(self, model):
        self.model.load_state_dict(model.cpu().state_dict())

    def step_epoch(self, steps):
        replay = deque(maxlen=self.replay_size)
        epsilon = self.epsilon_schedule()
        Rs, Qs = [], []
        tic = time.time()
        for _ in range(steps):
            action_random = np.random.randint(0, self.action_dim, self.num_envs)
            st = torch.from_numpy(np.array(self.obs)).float().div(255.0).to(self.device, memory_format=self.memory_format)
            qs = self.model(st)
            qs_max, qs_argmax = qs.max(dim=-1)
            action_greedy = qs_argmax.tolist()
            Qs.append(qs_max.mean().item())
            action = [act_grd if p > epsilon else act_rnd for p, act_rnd, act_grd in
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
        print(f"Rank {self.rank:2d}: Data Collection Time:\t\t {toc - tic:6.2f}, Speed {len(replay) / (toc - tic):6.1f}")
        print(f"Rank {self.rank:2d}: EP Reward mean/std/max:\t\t {np.mean(Rs):8.3f}, {np.std(Rs):8.3f}, {np.max(Rs):8.3f}")
        print(f"Rank {self.rank:2d}: Qmax mean/std/max:\t\t {np.mean(Qs):8.3f}, {np.std(Qs):8.3f}, {np.max(Qs):8.3f}")
        print(f"Rank {self.rank:2d}: Current Epsilon", epsilon)
        return replay, Rs, Qs

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

        def make_env(env_id):
            env = make_atari(f'{env_id}NoFrameskip-v4')
            env = wrap_deepmind(env, episode_life=True, clip_rewards=False, frame_stack=True, scale=False)
            return env

        self.envs = ShmemVecEnv([lambda: make_env(self.env_id) for _ in range(self.num_test_envs)])
        self.action_dim = self.envs.action_space.n
        self.state_shape = self.envs.observation_space.shape


        self.device = torch.device('cuda:0')
        self.memory_format = torch.channels_last
        self.model = NatureCNN(self.state_shape[0], self.action_dim).to(self.device, memory_format=self.memory_format)
        self.model_target = NatureCNN(self.state_shape[0], self.action_dim).to(self.device, memory_format=self.memory_format)


        # model = NatureCNN(self.state_shape[0], self.action_dim)
        # self.model = torch.nn.DataParallel(model).to(self.device, memory_format=self.memory_format)
        # model_ = NatureCNN(self.state_shape[0], self.action_dim)
        # self.model_target = torch.nn.DataParallel(model_).to(self.device, memory_format=self.memory_format)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.adam_lr, eps=self.adam_eps)
        self.memory_format = torch.channels_last
        self.update_count = 0

        self.actors = [Actor.remote(rank=rank, **kwargs) for rank in range(self.num_actors)]
        self.replay = deque(maxlen=self.replay_size)

        self.obs = self.envs.reset()
        self.Rs = []
        self.Qs = []
        self.Ls = []
        self.RTest = []
        self.R = np.zeros(self.num_test_envs)

    def train_epoch(self, steps):

        dataset = ReplayDataset(self.replay)
        dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_data_workers)
        prefetcher = DataPrefetcher(dataloader, self.device)
        self.model.cuda()

        Ls = []
        for _ in tqdm(range(steps)):
            try:
                data = prefetcher.next()
            except:
                raise StandardError("No data left")

            states, actions, rewards, next_states, terminals = data
            states = states.float().to(memory_format=self.memory_format).div(255.0)
            next_states = next_states.float().to(memory_format=self.memory_format).div(255.0)
            actions = actions.long()
            terminals = terminals.float()
            rewards = rewards.float()

            with torch.no_grad():
                q_next = self.model_target(next_states)
                q_next_online = self.model(next_states)
                q_next = q_next.gather(1, q_next_online.argmax(dim=-1).unsqueeze(-1)).squeeze(-1)
                q_target = rewards + self.discount * (1 - terminals) * q_next

            q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            # loss = F.mse_loss(q, q_target)
            loss= F.smooth_l1_loss(q, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_count += 1
            Ls.append(loss.item())
            if self.update_count % self.target_sync_freq == 0:
                self.model_target.load_state_dict(self.model.state_dict())
        return Ls

    def test(self):
        Rs = []
        E = 0
        while True:
            obs = torch.from_numpy(np.array(self.obs)).to(self.device).float().div(255.0)
            action = self.model(obs).argmax(dim=-1).tolist()
            obs_next, reward, done, info = self.envs.step(action)
            self.obs = obs_next
            self.R += np.array(reward)
            for idx, d in enumerate(done):
                if d:
                    Rs.append(self.R[idx])
                    self.R[idx] = 0
                    E += 1
            if E > 300:
                return Rs



    def run(self):

        frames_per_epoch = self.total_steps // self.epoches
        steps_per_actor = int(frames_per_epoch / self.num_envs / self.num_actors)
        steps_per_epoch_update = int(frames_per_epoch * self.update_per_data / self.batch_size)
        self.target_sync_freq = int(self.target_update_freq / (self.batch_size / self.base_batch_size))
        self.steps_per_actor = steps_per_actor

        for epoch in range(self.epoches):
            ticc = time.time()


            tic = time.time()
            datas = ray.get([a.step_epoch.remote(steps_per_actor) for a in self.actors])
            Rs, Qs = [], []
            for replay, rs, qs in datas:
                self.replay.extend(replay)
                Rs += rs
                Qs += qs
                self.Qs += qs
                self.Rs += rs
            toc = time.time()

            print(f"Eopch {epoch:3d}: Data Collection Time:\t\t{toc - tic:6.2f}, Speed {frames_per_epoch / (toc - tic):6.1f}")
            print(f"Epoch {epoch:3d}: EP Reward mean/std/max:\t\t {np.mean(Rs):8.3f}, {np.std(Rs):8.3f}, {np.max(Rs):8.3f}")
            print(f"Epoch {epoch:3d}: Qmax mean/std/max:\t\t {np.mean(Qs):8.3f}, {np.std(Qs):8.3f}, {np.max(Qs):8.3f}")

            tic = time.time()
            Ls = self.train_epoch(steps_per_epoch_update)
            toc = time.time()
            print(f"Epoch {epoch:3d}: Model Training Time: {toc - tic:6.2f}, Speed {steps_per_epoch_update / (toc - tic):6.1f}")
            print(f"Epoch {epoch:3d}: Epoch Loss mean/std/max {np.mean(Ls):8.5f}, {np.std(Ls):8.5f}, {np.max(Ls):8.5f}")
            self.Ls += Ls

            tic = time.time()
            ray.get([a.load_model.remote(self.model) for a in self.actors])
            toc = time.time()
            print(f"Epoch {epoch:3d}: Model Sync Time: {toc - tic:6.2f}")

            tic = time.time()
            RTest = self.test()
            toc = time.time()
            print(f"Epoch {epoch:3d}: Model Test Time: {toc - tic}")
            print(f"Epoch {epoch:3d}: EP Test Reward mean/std/max {np.mean(RTest):8.3f}, {np.std(RTest):8.3f}, {np.max(RTest):8.3f}")
            self.RTest += RTest

            torch.save({'model': self.model.state_dict(), 'Ls': self.Ls, 'Rs': self.Rs, 'Qs': self.Qs, 'RTs': self.RTest}, f'ckpt/deepq_e{epoch}.pth')

            print("=" * 50)
            print(f"Total Epoch Time : {toc - ticc}")
            print("=" * 50)

