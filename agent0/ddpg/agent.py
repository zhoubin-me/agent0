import copy

import numpy as np
import torch
import torch.nn.functional as fx
from agent0.common.mujoco_wrappers import make_bullet_env
from agent0.ddpg.config import Config
from agent0.ddpg.model import DDPGMLP
from agent0.ddpg.replay_buffer import ReplayBuffer
from torch.distributions import Normal


class DDPGAgent:
    def __init__(self, **kwargs):
        self.cfg = Config(**kwargs)
        cfg = self.cfg
        self.device = torch.device('cuda:0')
        self.env = make_bullet_env(cfg.game, seed=cfg.seed)
        self.action_high = self.env.action_space.high[0]

        self.replay = ReplayBuffer(size=cfg.buffer_size)

        self.network = DDPGMLP(self.env.observation_space.shape[0], self.env.action_space.shape[0],
                               self.action_high, cfg.hidden_size).to(self.device)
        self.network.train()
        self.target_network = copy.deepcopy(self.network)

        self.actor_optimizer = torch.optim.Adam(self.network.get_policy_params(), lr=cfg.p_lr)
        self.critic_optimizer = torch.optim.Adam(self.network.get_value_params(), lr=cfg.v_lr)

        self.total_steps = 0
        self.noise_std = torch.tensor(self.cfg.action_noise_level * self.action_high).to(self.device)

    def step(self, testing=False):
        if self.total_steps == 0:
            self.state = self.env.reset()

        if self.total_steps < self.cfg.exploration_steps:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(self.state).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                action_mean = self.network.act(state)
            if testing:
                action = action_mean.squeeze(0).cpu().numpy()
            else:
                dist = Normal(action_mean, self.noise_std.expand_as(action_mean))
                action = dist.sample().clamp(-self.action_high, self.action_high).squeeze(0).cpu().numpy()

        next_state, reward, done, info = self.env.step(action)

        if not testing:
            self.total_steps += 1
            self.replay.add(self.state, action, reward, next_state, int(done))

        if 'real_reward' in info:
            rs = info['real_reward']
        else:
            rs = None

        self.state = next_state
        if done:
            self.state = self.env.reset()

        if not testing and self.total_steps > self.cfg.exploration_steps:
            vloss, ploss = self.train_step()
        else:
            vloss, ploss = None, None
        return dict(rs=rs, ploss=ploss, vloss=vloss)

    def train_step(self):
        experiences = self.replay.sample(self.cfg.batch_size)
        states, actions, rewards, next_states, terminals = map(lambda x: torch.tensor(x).to(self.device).float(),
                                                               experiences)

        terminals = terminals.float().view(-1, 1)
        rewards = rewards.float().view(-1, 1)
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

        for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

        return value_loss.item(), policy_loss.item()


if __name__ == '__main__':
    agent = DDPGAgent()
    rs, vloss, ploss = [], [], []
    while True:
        info = agent.step()
        if info['rs'] is not None:
            rs.append(info['rs'])
        if info['vloss'] is not None:
            vloss.append(info['vloss'])
        if info['ploss'] is not None:
            ploss.append(info['ploss'])

        if agent.total_steps % 5000 == 0:
            stream = ""
            if len(rs) > 10:
                stream += f"Rs: {np.mean(rs[-100:])}\t"
            if len(vloss) > 10:
                stream += f"VLoss: {np.mean(vloss[-100:])}\t"
            if len(ploss) > 10:
                stream += f"PLoss: {np.mean(ploss[-100:])}"
            print(stream)
