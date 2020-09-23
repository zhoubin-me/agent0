import copy

import numpy as np
import torch
import torch.nn.functional as fx
from agent0.common.mujoco_wrappers import make_bullet_env
from agent0.common.replay import ReplayDataset
from agent0.common.utils import DataLoaderX, DataPrefetcher
from agent0.ddpg.config import Config
from agent0.ddpg.model import DDPGMLP


class Agent:
    def __init__(self, **kwargs):
        self.cfg = Config(**kwargs)
        self.env = make_bullet_env(self.cfg.game, seed=None)
        self.obs_shape = self.env.observation_space.shape
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_high = self.env.action_space.high[0]
        self.device = torch.device('cuda:0')
        self.network = DDPGMLP(self.state_dim, self.action_dim, self.action_high).to(self.device)
        self.target_network = copy.deepcopy(self.network)

        self.actor_optimizer = torch.optim.Adam(self.network.get_policy_params(), lr=self.cfg.p_lr)
        self.critic_optimizer = torch.optim.Adam(self.network.get_value_params(), lr=self.cfg.v_lr)
        self.total_steps = 0
        self.noise_std = self.cfg.action_noise_level * self.action_high
        self.replay = ReplayDataset(self.obs_shape, **kwargs)
        self.data_fetcher = None
        self.state = self.env.reset()
        self.steps = 0

    def get_data_fetcher(self):
        data_loader = DataLoaderX(self.replay, batch_size=self.cfg.batch_size, shuffle=True,
                                  num_workers=self.cfg.num_data_workers, pin_memory=self.cfg.pin_memory)
        data_fetcher = DataPrefetcher(data_loader, self.device)
        return data_fetcher

    def train_ddpg_step(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            target_q = self.target_network.action_value(next_states, self.target_network.p(next_states))
            target_q = rewards + (1.0 - terminals) * self.cfg.gamma * target_q.detach()

        current_q = self.network.action_value(states, actions)
        value_loss = fx.mse_loss(current_q, target_q)
        policy_loss = self.network.action_value(states, self.network.act(states)).mean().neg()
        return value_loss, policy_loss

    def act_eval(self):
        state = torch.from_numpy(self.state).to(self.device).float()
        action = self.network.act(state)
        return action.squeeze(0).cpu().numpy()

    def act_explore(self):
        with torch.no_grad():
            state = torch.from_numpy(self.state).to(self.device).float()
            action = self.network.act(state)
        return action.squeeze(0).cpu().numpy()

    def act_random(self):
        return self.env.action_space.sample()

    def sample(self, steps, act_random=False, testing=False, test_episodes=10):
        data, rs = [], []
        step = 0
        while True:
            if testing:
                action = self.act_eval()
            elif act_random:
                action = self.act_random()
            else:
                action = self.act_explore()
            next_state, reward, done, info = self.env.step(action)
            if 'real_reward' in info:
                rs.append(info['real_reward'])
            self.total_steps += 1
            if not testing:
                data.append((np.concatenate((self.state.reshape(1, -1), next_state.reshape(1, -1)), axis=0),
                             action, reward, done))
            self.state = next_state
            if done:
                self.state = self.env.reset()
            step += 1

            if not testing and step > steps:
                break
            if testing and len(rs) > test_episodes:
                break

        return data, rs

    def train_step(self):
        try:
            data = self.data_fetcher.next()
        except (StopIteration, AttributeError):
            self.data_fetcher = self.get_data_fetcher()
            data = self.data_fetcher.next()

        frames, actions, rewards, terminals, _, _ = data
        states = frames[:, 0, :].float().view(-1, self.state_dim)
        next_states = frames[:, 1, :].float().view(-1, self.state_dim)
        actions = actions.float().view(-1, self.action_dim)
        terminals = terminals.float().view(-1, 1)
        rewards = rewards.float().view(-1, 1)

        value_loss, policy_loss = self.train_ddpg_step(states, next_states, actions, terminals, rewards)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

        return {'v_loss': value_loss.item(), 'p_loss': policy_loss.item()}
