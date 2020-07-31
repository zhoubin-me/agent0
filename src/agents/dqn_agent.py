import ray
import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import ShmemVecEnv, DummyVecEnv
from src.common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from src.common.model import NatureCNN
from src.common.utils import LinearSchedule




@ray.remote(num_gpus=0.2)
class Actor:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        def make_env(env_id):
            env = make_atari(f'{env_id}NoFrameskip-v4')
            env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=False)
            return env

        self.device = torch.device(f'cuda:{self.gpu_id}' if self.gpu_id >= 0 else 'cpu')
        self.train_envs = ShmemVecEnv([lambda: make_env(self.env_id) for _ in range(self.num_envs)])
        self.obs = self.train_envs.reset()
        self.action_dim = self.train_envs.action_space.n
        self.state_shape = self.train_envs.observation_space.shape
        self.network = NatureCNN(self.state_shape[-1], self.action_dim).to(self.device)
        self.min_epsilon = self.min_epsilons[self.rank]
        self.epsilon_schedule = LinearSchedule(1.0, self.min_epsilon, int(1e6))

        self.R = np.zeros(self.num_envs)
        self.Rs = []

    def get_inp(self, obs):
        return torch.from_numpy(obs).float().permute(0, 3, 1, 2).to(self.device) / 255.0

    def get_actions(self):
        with torch.no_grad():
            inp = self.get_inp(self.obs)
            qs = self.network(inp)
            actions_greedy = qs.argmax(dim=-1).tolist()
            actions_random = np.random.randint(0, self.action_dim, self.num_envs)
            # epsilon = self.epsilon_schedule(4)
            epsilon = self.min_epsilon
        actions = [act_greedy if rnd > epsilon else act_random
                   for rnd, act_greedy, act_random in zip(np.random.rand(4), actions_greedy, actions_random)]
        return actions

    def get_random_actions(self):
        return np.random.randint(0, self.train_envs.action_space.n, self.num_envs)

    def step(self, steps):
        data = []
        for step in range(steps):
            actions = self.get_actions()
            obs_next, rewards, dones, infos = self.train_envs.step(actions)
            self.R += np.array(rewards)
            for n, entry in enumerate(zip(self.obs, actions, rewards, dones, obs_next)):
                data.append(entry)
                if entry[-2]:
                    self.Rs.append(self.R[n])
                    self.R[n] = 0
            self.obs = obs_next
        print(self.rank, self.min_epsilon, np.mean(self.Rs), np.std(self.Rs), np.max(self.Rs))
        return data

    def set_network(self, network):
        self.network = copy.deepcopy(network)


@ray.remote(num_gpus=0.5)
class Learner:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        def make_env(env_id):
            env = make_atari(f'{env_id}NoFrameskip-v4')
            env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=False)
            return env

        self.device = torch.device(f'cuda:{self.gpu_id}' if self.gpu_id >= 0 else 'cpu')
        self.test_env = make_env(self.env_id)
        self.network = NatureCNN(4, self.test_env.action_space.n).to(self.device)
        self.network_target = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        self.update_steps = 0
        self.Ls = []


    def train(self, data):
        obs, actions, rewards, dones, obs_next = data
        obs = obs.permute(0, 3, 1, 2) / 255.0
        obs_next = obs_next.permute(0, 3, 1, 2) / 255.0
        actions = actions.long()
        rewards = rewards.float()
        dones = dones.float()

        with torch.no_grad():
            q_next = self.network_target(obs_next)
            q_next_online = self.network(obs_next)
            q_next = q_next.gather(1, q_next_online.argmax(dim=-1).unsqueeze(-1)).squeeze(-1)
            q_target = rewards + self.discount * (1 - dones) * q_next

        q = self.network(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(q, q_target)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        self.Ls.append(loss.item())

        if self.update_steps % self.target_update_freq == 0:
            self.network_target.load_state_dict(self.network.state_dict())

        return np.mean(self.Ls[-100:])

    def get_latest_network(self):
        return self.network

    def test(self):
        Rs, R = [], 0
        for e in range(100):
            obs = self.test_env.reset()
            while True:
                with torch.no_grad():
                    inp = self.get_inp(obs)
                    qs = self.network(inp)
                    action = qs.argmax(dim=-1).item()

                obs_next, reward, done, info = self.test_env.step(action)
                obs = obs_next
                R += reward
                if done:
                    Rs.append(R)
                    R = 0
                    break
        print(np.mean(Rs), np.std(Rs), np.max(Rs))

    def get_inp(self, obs):
        return torch.from_numpy(np.array(obs)).unsqueeze(0).float().permute(0, 3, 1, 2).to(self.device) / 255.0




@ray.remote(num_gpus=0.1)
class Sampler:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.device = torch.device(f'cuda:{self.gpu_id}' if self.gpu_id >= 0 else 'cpu')
        self.buffer = ReplayBuffer(self.replay_size)
        self.stream = torch.cuda.Stream()
        self.data = None
        self.count = 0

    def preload(self):
        try:
            data = self.buffer.sample(self.batch_size)
            self.data = map(lambda x: torch.from_numpy(x), data)
        except:
            print("Cannot preload")
            return

        with torch.cuda.stream(self.stream):
            self.data = list(x.to(self.device, non_blocking=True) for x in self.data)

    def sample(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        self.preload()
        return data

    def add(self, entry):
        self.buffer.add(entry)
        self.count += 1

    def add_entries(self, entries):
        for entry in entries:
            self.add(entry)

        if self.data is None and self.count > self.batch_size:
            self.preload()

    def get_count(self):
        return self.count




class Agent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.setup()

    def default_params(self):
        return dict(
            env_id='Breakout',
            num_envs=4,
            gpu_id=0,
            lr=1e-4,
            replay_size=int(1e6),
            num_actors=4,
            actor_steps=32,
            batch_size=128,
            discount=0.99,
            target_update_freq=100,
            min_epsilons=[0.99, 0.8, 0.2, 0.01],
        )

    def setup(self):
        if not hasattr(self, 'env_id'):
            kwargs = self.default_params()
            for k, v in kwargs.items():
                setattr(self, k, v)

        self.learner = Learner.remote(**kwargs)
        self.sampler = Sampler.remote(**kwargs)
        self.actors = [Actor.remote(**kwargs.update(rank=n)) for n in range(self.num_actors)]


    def warmup(self):
        tic = time.time()
        # Filling Sampler
        for step in range(10):
            sampler_ops = [self.sampler.add_entries.remote(a.step.remote(self.actor_steps)) for a in self.actors]
            ray.get(sampler_ops)
            print(ray.get(self.sampler.get_count.remote()))

        toc = time.time()
        print("Warmming up:", ray.get(self.sampler.get_count.remote()) / (toc - tic))

    def train(self):
        # Start Training
        tic = time.time()
        for step in range(1000):
            sampler_ops = [self.sampler.add_entries.remote(a.step.remote(self.actor_steps)) for a in self.actors]

            learner_train_op = self.learner.train.remote(self.sampler.sample.remote())
            actor_sync_op = [a.set_network.remote(self.learner.get_latest_network.remote()) for a in self.actors]
            sampler_count_op = self.sampler.get_count.remote()
            # agent_test_op = self.agent.test.remote()
            loss, count, *_ = ray.get([learner_train_op, sampler_count_op] + sampler_ops + actor_sync_op)
            print(step, loss, count)
        toc = time.time()
        print(ray.get(self.sampler.get_count.remote()) / (toc - tic))

    def run(self):
        self.warmup()
        for epoch in range(10):
            print("Epoch ", epoch)
            self.train()

