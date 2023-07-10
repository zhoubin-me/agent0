from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lz4.block import compress

from agent0.common.atari_wrappers import make_atari
from agent0.deepq.new_config import ExpConfig, AlgoEnum
from agent0.deepq.new_model import DeepQNet


class Actor:
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.envs = make_atari(cfg.env_id, cfg.actor.num_envs)
        self.obs, _ = self.envs.reset()
        self.model = DeepQNet(cfg).to(cfg.device.value)

    def act(self, st, epsilon):
        qt = self.model.qval(st)
        action_random = np.random.randint(0, self.cfg.action_dim, self.cfg.actor.num_envs)
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
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.model = DeepQNet(cfg).to(cfg.device.value)
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
            if self.cfg.learner.double_q:
                a_next = self.model(next_obs).argmax(dim=-1)
            else:
                a_next = q_next.argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next]
            q_target = rewards + self.cfg.learner.discount * (1 - terminals) * q_next

        q = self.model(obs)[self.batch_indices, actions]
        loss = F.smooth_l1_loss(q, q_target, reduction='none').view(-1)
        return loss

    def train_step_c51(self, obs, actions, rewards, terminals, next_obs):
        with torch.no_grad():
            prob_next = self.model_target(next_obs).softmax(dim=-1)

            if self.cfg.learner.double_q:
                a_next = self.model.qval(next_obs).argmax(dim=-1)
            else:
                a_next = prob_next.mul(self.model.head.atoms).sum(dim=-1).argmax(dim=-1)
            
            prob_next = prob_next[self.batch_indices, a_next, :]
        
            atoms_next = rewards.view(-1, 1) + self.cfg.learner.discount * (1 - terminals.view(-1, 1)) \
                * self.model.head.atoms.view(1, -1)
            
            cfg = self.cfg.learner.c51
            atoms_next.clamp_(cfg.vmin, cfg.vmax)
            base = (atoms_next - cfg.vmin) / self.model.head.delta

            lo, up = base.floor().long(), base.ceil().long()
            lo[(up > 0) * (lo == up)] -= 1
            up[(lo < (cfg.atoms - 1)) * (lo == up)] += 1

            target_prob = torch.zeros_like(prob_next)
            offset = torch.linspace(
                0, ((self.cfg.learner.batch_size - 1) * cfg.atoms), self.cfg.learner.batch_size
            ).to(self.cfg.device.value)
            offset = offset.view(-1, 1).expand(self.cfg.learner.batch_size, cfg.atoms).long()
            
            target_prob.view(-1).index_add_(
                0, (lo + offset).view(-1), (prob_next * (up.float() - base)).view(-1)
            )

            target_prob.view(-1).index_add_(
                0, (up + offset).view(-1), (prob_next * (base - lo.float())).view(-1)
            )

        log_prob = self.model(obs).log_softmax(dim=-1)
        log_prob = log_prob[self.batch_indices, actions, :]
        loss = target_prob.mul(log_prob).sum(-1).neg()
        return loss.view(-1)


    def train_step(self, obs, actions, rewards, terminals, next_obs):
        algo = self.cfg.learner.algo.value

        if algo == AlgoEnum.dqn.value:
            loss_fn = self.train_step_dqn
        elif algo == AlgoEnum.c51.value:
            loss_fn = self.train_step_c51
        else:
            raise NotImplementedError(algo)
        
        loss = loss_fn(obs, actions, rewards, terminals, next_obs)


        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1

        if self.update_steps % self.cfg.learner.target_update_freq == 0:
            self.model_target = deepcopy(self.model)

        return {"loss": loss.item()}
