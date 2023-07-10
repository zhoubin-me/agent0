from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lz4.block import compress

from agent0.common.atari_wrappers import make_atari
from agent0.deepq.new_config import ExpConfig
from agent0.deepq.new_model import DeepQNet

class Actor:
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg

        self.envs = make_atari(cfg.env_id, cfg.actor.num_envs)
        self.act_dim = self.envs.action_space[0].n
        self.obs_shape = self.envs.observation_space.shape[1:]
        self.obs, _ = self.envs.reset()
        self.model = DeepQNet(self.act_dim, self.obs_shape[0]).to(cfg.device.value)

    def act(self, st, epsilon):
        qt = self.model(st)
        action_random = np.random.randint(0, self.act_dim, self.cfg.actor.num_envs)
        qt_max, qt_arg_max = qt.max(dim=-1)
        action_greedy = qt_arg_max.cpu().numpy()
        action = np.where(
            np.random.rand(self.cfg.actor.num_envs) > epsilon,
            action_greedy,
            action_random,
        )
        return action, qt_max.mean().item()

    def sample(self, epsilon, state_dict=None):
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        rs, qs, data = [], [], []
        for _ in range(self.cfg.actor.actor_steps):
            with torch.no_grad():
                st = (
                    torch.from_numpy(self.obs)
                    .to(self.cfg.device.value)
                    .float()
                    .div(255.0)
                )
                action, qt_max = self.act(st, epsilon)

            qs.append(qt_max)
            obs_next, reward, terminal, truncated, info = self.envs.step(action)

            done = np.logical_and(terminal, np.logical_not(truncated))
            for st, at, rt, dt, st_next in zip(
                self.obs, action, reward, done, obs_next
            ):
                data.append(
                    (compress(np.concatenate((st, st_next), axis=0)), at, rt, dt)
                )

            self.obs = obs_next

            if "final_info" in info:
                final_infos = info["final_info"][info["_final_info"]]
                for stat in final_infos:
                    rs.append(stat["episode"]["r"][0])
        return data, rs, qs

    def close(self):
        self.envs.close()


class Learner:
    def __init__(self, cfg: ExpConfig, model: nn.Module):
        self.cfg = cfg
        self.model = model
        self.model_target = deepcopy(self.model)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            cfg.learner.learning_rate,
            eps=1e-2 / cfg.learner.batch_size,
        )

        self.update_steps = 0
        self.batch_indices = torch.arange(cfg.learner.batch_size).to(cfg.device.value)

    def train_step_dqn(self, obs, actions, rewards, terminals, next_obs):
        with torch.no_grad():
            q_next = self.model_target(next_obs)
            a_next = q_next.argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next]
            q_target = rewards + self.cfg.learner.discount * (1 - terminals) * q_next

        q = self.model(obs)[self.batch_indices, actions]
        loss = F.smooth_l1_loss(q, q_target)
        return loss

    def train_step(self, obs, actions, rewards, terminals, next_obs):
        loss = self.train_step_dqn(obs, actions, rewards, terminals, next_obs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1

        if self.update_steps % self.cfg.learner.target_update_freq == 0:
            self.model_target = deepcopy(self.model)

        return {"loss": loss.item()}
