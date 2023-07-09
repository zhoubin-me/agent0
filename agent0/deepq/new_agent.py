from agent0.deepq.new_config import ExpConfig
from agent0.common.atari_wrappers import make_atari
from agent0.deepq.new_model import DeepQNet
from agent0.common.utils import DataLoaderX, DataPrefetcher
from agent0.deepq.replay import ReplayDataset

import torch
import torch.nn.functional as F
import numpy as np
from lz4.block import compress
from copy import deepcopy

class Actor:
    def __init__(self, cfg: ExpConfig):

        self.cfg = cfg
        self.envs = make_atari(self.cfg.game, self.cfg.actor.num_envs)
        self.obs, _ = self.envs.reset()

        self.action_dim = self.envs.action_space[0].n
        self.obs_shape = self.envs.observation_space.shape
        self.device = self.cfg.actor.device.value

        self.model = DeepQNet(self.action_dim, self.obs_shape[1]).to(self.device)
        
    def act(self, st, epsilon):
        qt = self.model(st)
        action_random = np.random.randint(0, self.action_dim, self.cfg.actor.num_envs)
        qt_max, qt_arg_max = qt.max(dim=-1)
        action_greedy = qt_arg_max.cpu().numpy()
        action = np.where(np.random.rand(self.cfg.actor.num_envs) > epsilon, action_greedy, action_random)
        return action, qt_max.mean().item()

    def sample(self, steps, epsilon, model=None):
        if model is not None:
                self.model = model
        rs, qs, data = [], [], []
        step = 0
        while True:
            step += 1
            with torch.no_grad():
                st = torch.from_numpy(self.obs).to(self.device).float().div(255.0)
                action, qt_max = self.act(st, epsilon)

            qs.append(qt_max)
            obs_next, reward, terminal, truncated, info = self.envs.step(action)

            done = np.logical_and(terminal, np.logical_not(truncated))
            for st, at, rt, dt, st_next in zip(self.obs, action, reward, done, obs_next):
                data.append((compress(np.concatenate((st, st_next), axis=0)), at, rt, dt))

            self.obs = obs_next

            if 'final_info' in info:
                final_infos = info['final_info'][info['_final_info']]
                for stat in final_infos:
                    rs.append(stat['episode']['r'][0])

            if step > steps:
                break

        return data, rs, qs

    def close(self):
        self.envs.close()

class Learner:
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.device = self.cfg.trainer.device.value

        dummy_env = make_atari(self.cfg.game, 1)
        self.action_dim = dummy_env.action_space[0].n
        self.obs_shape = dummy_env.observation_space.shape
        dummy_env.close()
        del dummy_env

        self.model = DeepQNet(self.action_dim, self.obs_shape[1]).to(self.device)
        self.model_target = deepcopy(self.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            self.cfg.trainer.learning_rate, 
            eps=1e-2 / self.cfg.trainer.batch_size)
        
        self.update_steps = 0
        self.replay = ReplayDataset(cfg)
        self.batch_indices = torch.arange(self.cfg.trainer.batch_size).to(self.device)

    def get_data_fetcher(self):

        data_loader = DataLoaderX(self.replay, batch_size=self.cfg.trainer.batch_size, 
                                    shuffle=True, num_workers=2, pin_memory=True)
        data_fetcher = DataPrefetcher(data_loader, self.device)
        return data_fetcher

    def train_step_dqn(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next = self.model_target(next_states)
            a_next = q_next.argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next]
            q_target = rewards + self.cfg.algo.discount * (1 - terminals) * q_next

        q = self.model(states)[self.batch_indices, actions]
        loss = F.smooth_l1_loss(q, q_target)
        return loss
    
    def train_step(self, data=None):
        if data is None:
            try:
                data = self.data_fetcher.next()
            except (StopIteration, AttributeError):
                self.data_fetcher = self.get_data_fetcher()
                data = self.data_fetcher.next()

        frames, actions, rewards, terminals = data
        frames = frames.reshape(self.cfg.trainer.batch_size, -1, *self.obs_shape[2:]).float().div(255.0)
        states = frames[:, :4, :, :]
        next_states = frames[:, -4:, :, :]
        actions = actions.long()
        terminals = terminals.float()
        rewards = rewards.float()

        loss = self.train_step_dqn(states, next_states, actions, terminals, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1

        if self.update_steps % self.cfg.trainer.target_update_freq == 0:
            self.model_target = deepcopy(self.model)

        return {'loss' : loss.item()}