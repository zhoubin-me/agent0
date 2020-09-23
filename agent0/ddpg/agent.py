import copy

import numpy as np
import torch
import torch.nn.functional as fx
from agent0.common.mujoco_wrappers import make_bullet_env
from agent0.ddpg.config import Config
from agent0.ddpg.model import DDPGMLP, SACMLP, TD3MLP
from agent0.ddpg.replay_buffer import ReplayBuffer
from torch.distributions import Normal


class Agent:
    def __init__(self, **kwargs):
        self.cfg = Config(**kwargs)
        cfg = self.cfg
        self.device = torch.device('cuda:0')
        self.env = make_bullet_env(cfg.game, seed=cfg.seed)
        self.action_high = self.env.action_space.high[0]

        self.replay = ReplayBuffer(size=cfg.buffer_size)

        self.net = {
            'ddpg': DDPGMLP,
            'sac': SACMLP,
            'td3': TD3MLP,
        }

        self.step_fn = {
            'ddpg': self.train_step_ddpg,
            'sac': self.train_step_sac,
            'td3': self.train_step_td3,
        }

        assert self.cfg.algo in self.net
        self.network = self.net[self.cfg.algo](
            self.env.observation_space.shape[0], self.env.action_space.shape[0],
            self.action_high, cfg.hidden_size).to(self.device)
        self.network.train()
        self.target_network = copy.deepcopy(self.network)

        self.actor_optimizer = torch.optim.Adam(self.network.get_policy_params(), lr=cfg.p_lr)
        self.critic_optimizer = torch.optim.Adam(self.network.get_value_params(), lr=cfg.v_lr)

        self.total_steps = 0
        self.noise_std = torch.tensor(self.cfg.action_noise_level * self.action_high).to(self.device)
        self.state = self.env.reset()

        if self.cfg.algo == 'sac':
            self.target_entropy = torch.tensor(np.prod(self.env.action_space.shape)).to(self.device).float().neg()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.p_lr)

    def act(self, st, random=False, testing=False):
        if random:
            return self.env.action_space.sample()

        if self.cfg.algo == 'ddpg':
            action_mean = self.network.act(st)
        elif self.cfg.algo == 'sac':
            _, _, action_mean = self.network.act(st)
        else:
            raise ValueError("no such policy")

        if testing:
            return action_mean.squeeze(0).cpu().numpy()
        else:
            dist = Normal(action_mean, self.noise_std.expand_as(action_mean))
            return dist.sample().clamp(-self.action_high, self.action_high).squeeze(0).cpu().numpy()

    def step(self, testing=False):

        st = torch.tensor(self.state).float().to(self.device).unsqueeze(0)
        action = self.act(st, random=self.total_steps < self.cfg.exploration_steps, testing=testing)
        next_state, reward, done, info = self.env.step(action)

        if not testing:
            self.total_steps += 1
            self.replay.add(self.state, action, reward, next_state, int(done))

        self.state = next_state
        if done:
            self.state = self.env.reset()

        if not testing and self.total_steps > self.cfg.exploration_steps:
            loss = self.train_step()
        else:
            loss = dict()

        if 'real_reward' in info:
            loss.update(rs=info['real_reward'])

        return loss

    def train_step_gmm(self, states, actions, rewards, next_states, terminals):
        pass

    def train_step_td3(self, states, actions, rewards, next_states, terminals):
        with torch.no_grad():
            next_actions_mean = self.target_network.act(next_states)
            dist = Normal(next_actions_mean, self.noise_std.expand_as(next_actions_mean))
            next_actions = dist.sample().clamp(-self.action_high, self.action_high)
            target_q1, target_q2 = self.target_network.action_value(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - terminals) * self.cfg.gamma * target_q.detach()

        current_q1, current_q2 = self.network.action_value(states, actions)
        value_loss = fx.mse_loss(current_q1, target_q) + fx.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        loss = dict(vloss=value_loss.item())

        if self.total_steps % self.cfg.policy_update_freq == 0:
            policy_loss = self.network.v(torch.cat([states, self.network.p(states)], dim=1)).mean().neg()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            loss.update(ploss=policy_loss.item())
        return loss

    def train_step_sac(self, states, actions, rewards, next_states, terminals):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.network.act(next_states)
            target_q1, target_q2 = self.target_network.action_value(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) + self.log_alpha.exp() * next_entropies
            target_q = rewards + (1.0 - terminals) * self.cfg.gamma * target_q.detach()

        current_q1, current_q2 = self.network.action_value(states, actions)
        value_loss = fx.mse_loss(current_q1, target_q) + fx.mse_loss(current_q2, target_q)

        sampled_action, entropy, _ = self.network.act(states)
        q1, q2 = self.network.action_value(states, sampled_action)
        q = torch.min(q1, q2)
        policy_loss = (q + self.log_alpha.exp() * entropy).mean().neg()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        entropy_loss = (self.log_alpha * (self.target_entropy - entropy).detach()).mean().neg()
        self.alpha_optim.zero_grad()
        entropy_loss.backward()
        self.alpha_optim.step()

        return {'vloss': value_loss.item(), 'ploss': policy_loss.item(), 'ent_loss': entropy_loss.item()}

    def train_step_ddpg(self, states, actions, rewards, next_states, terminals):
        with torch.no_grad():
            target_q = self.target_network.action_value(next_states, self.target_network.p(next_states))
            target_q = rewards + (1.0 - terminals) * self.cfg.gamma * target_q.detach()

        current_q = self.network.action_value(states, actions)
        value_loss = fx.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        policy_loss = self.network.action_value(states, self.network.act(states)).mean().neg()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return {'vloss': value_loss.item(), 'ploss': policy_loss.item()}

    def train_step(self):
        states, actions, rewards, next_states, terminals = map(
            lambda x: torch.tensor(x).to(self.device).float(), self.replay.sample(self.cfg.batch_size))

        terminals = terminals.float().view(-1, 1)
        rewards = rewards.float().view(-1, 1)

        loss = self.step_fn[self.cfg.algo](states, actions, rewards, next_states, terminals)

        for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

        return loss
