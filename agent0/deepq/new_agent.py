from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lz4.block import compress
from collections import deque

from agent0.common.atari_wrappers import make_atari
from agent0.deepq.new_config import AlgoEnum, ExpConfig
from agent0.deepq.new_model import DeepQNet


class Actor:
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.envs = make_atari(cfg.env_id, cfg.actor.num_envs)
        self.obs, _ = self.envs.reset()
        self.model = DeepQNet(cfg).to(cfg.device.value)
        self.tracker = deque(maxlen=cfg.learner.n_step_q)

    
    def act(self, st, epsilon):
        with torch.no_grad():
            qt = self.model.qval(st)
        action_random = np.random.randint(
            0, self.cfg.action_dim, self.cfg.actor.num_envs
        )
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
            done = np.logical_or(terminal, info['life_loss']) if 'life_loss' in info else terminal
            done = np.logical_and(done, np.logical_not(truncated))

            if self.cfg.learner.n_step_q > 1:
                self.tracker.append((self.obs, action, reward, done))
                r_nstep = np.zeros_like(reward)
                d_nstep = np.zeros_like(reward, dtype=np.bool_)
                for _, _, rt, dt in reversed(self.tracker):
                    d_nstep = np.logical_or(d_nstep, dt)
                    r_nstep = r_nstep * self.cfg.learner.discount * (1 - dt) + rt
                obs = self.tracker[0][0]
                action = self.tracker[0][1]
                reward = r_nstep
                done = d_nstep

            for st, at, rt, dt, st_next in zip(
                obs, action, reward, done, obs_next
            ):
                data.append(
                    (compress(np.concatenate((st, st_next), axis=0)), at, rt, dt))


            self.obs = obs_next
            if "final_info" in info:
                final_infos = info["final_info"][info["_final_info"]]
                for stat in final_infos:
                    rs.append(stat["episode"]["r"][0])

        return data, rs, qs

    def close(self):
        self.envs.close()


def huber_qr_loss(q, q_target, taus):
    huber_loss = F.smooth_l1_loss(q, q_target, reduction="none")
    loss = huber_loss * (taus - q_target.lt(q).detach().float()).abs()
    return loss.sum(-1).mean(-1).view(-1)


class Learner:
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.model = DeepQNet(cfg).to(cfg.device.value)
        self.model_target = deepcopy(self.model)

        self.optimizer = torch.optim.Adam(
            self.model.params(),
            cfg.learner.learning_rate,
            eps=1e-2 / cfg.learner.batch_size,
        )

        if self.cfg.learner.algo == AlgoEnum.fqf:
            self.fqf_optimizer = torch.optim.RMSprop(
                self.model.head.fraction_net.parameters(),
                lr=cfg.learner.learning_rate / 2e4,
                alpha=0.95,
                eps=0.00001,
            )

        self.update_steps = 0
        self.batch_indices = torch.arange(cfg.learner.batch_size).to(cfg.device.value)

    def train_step_dqn(self, obs, actions, rewards, terminals, next_obs):
        with torch.no_grad():
            q_next = self.model_target(next_obs)
            if self.cfg.learner.double_q:
                a_next = self.model.qval(next_obs).argmax(dim=-1)
            else:
                a_next = q_next.argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next]
            q_target = rewards + self.cfg.learner.discount * (1 - terminals) * q_next

        q = self.model(obs)[self.batch_indices, actions]
        loss = F.smooth_l1_loss(q, q_target, reduction="none").view(-1)
        return loss

    def train_step_c51(self, obs, actions, rewards, terminals, next_obs):
        with torch.no_grad():
            prob_next = self.model_target(next_obs).softmax(dim=-1)

            if self.cfg.learner.double_q:
                a_next = self.model.qval(next_obs).argmax(dim=-1)
            else:
                a_next = prob_next.mul(self.model.head.atoms).sum(dim=-1).argmax(dim=-1)

            prob_next = prob_next[self.batch_indices, a_next, :]

            atoms_next = rewards.view(-1, 1) + self.cfg.learner.discount * (
                1 - terminals.view(-1, 1)
            ) * self.model.head.atoms.view(1, -1)

            cfg = self.cfg.learner.c51
            atoms_next.clamp_(cfg.vmin, cfg.vmax)
            base = (atoms_next - cfg.vmin) / self.model.head.delta

            lo, up = base.floor().long(), base.ceil().long()
            lo[(up > 0) * (lo == up)] -= 1
            up[(lo < (cfg.num_atoms - 1)) * (lo == up)] += 1

            target_prob = torch.zeros_like(prob_next)
            offset = torch.linspace(
                0,
                ((self.cfg.learner.batch_size - 1) * cfg.num_atoms),
                self.cfg.learner.batch_size,
            ).to(self.cfg.device.value)
            offset = (
                offset.view(-1, 1)
                .expand(self.cfg.learner.batch_size, cfg.num_atoms)
                .long()
            )

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

    def train_step_qr(self, obs, actions, rewards, terminals, next_obs):
        with torch.no_grad():
            q_next = self.model_target(next_obs)
            if self.cfg.learner.double_q:
                a_next = self.model.qval(next_obs).argmax(dim=-1)
            else:
                a_next = q_next.mean(dim=-1).argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next, :]
            q_target = (
                rewards.view(-1, 1)
                + self.cfg.learner.discount * (1 - terminals.view(-1, 1)) * q_next
            )

        q = self.model(obs)[self.batch_indices, actions, :]
        q = rearrange(q, "b n -> b 1 n")
        q_target = rearrange(q_target, "b n -> b n 1")
        taus = self.model.head.cumulative_density.view(1, 1, -1)
        loss = huber_qr_loss(q, q_target, taus)
        return loss

    def train_step_iqn(self, obs, actions, rewards, terminals, next_obs):
        cfg = self.cfg.learner.iqn
        with torch.no_grad():
            q_next_convs = self.model_target.encoder(next_obs)
            if self.cfg.learner.double_q:
                q_next_online = self.model.head.qval(
                    self.model.encoder(next_obs), n=cfg.K
                )
                a_next = q_next_online.argmax(dim=-1)
            else:
                q_next_dummy = self.model_target.head.qval(q_next_convs, n=cfg.K)
                a_next = q_next_dummy.argmax(dim=-1)

            q_next, _ = self.model_target.head(q_next_convs, n=cfg.N_dash)
            q_next = q_next[self.batch_indices, :, a_next]

            q_target = (
                rewards.view(-1, 1)
                + self.cfg.learner.discount * (1 - terminals.view(-1, 1)) * q_next
            )

        q, taus = self.model.head(self.model.encoder(obs), n=cfg.N)
        q = q[self.batch_indices, :, actions]

        q = rearrange(q, "b n -> b 1 n")
        q_target = rearrange(q_target, "b n -> b n 1")
        taus = rearrange(taus, "b n 1 -> b 1 n")
        loss = huber_qr_loss(q, q_target, taus)
        return loss

    def train_step_fqf(self, obs, actions, rewards, terminals, next_obs):
        q_convs = self.model.encoder(obs)
        # taus: B X (N+1) X 1, taus_hats: B X N X 1
        taus, taus_hat, _ = self.model.head.prop_taus(q_convs.detach())
        # q_hat: B X N X A
        q_hat, _ = self.model.head(q_convs, taus=taus_hat)
        q_hat = q_hat[self.batch_indices, :, actions]

        with torch.no_grad():
            q_next_convs = self.model_target.encoder(next_obs)
            if self.cfg.learner.double_q:
                q_next_online = self.model.head.qval(self.model.encoder(next_obs))
                a_next = q_next_online.argmax(dim=-1)
            else:
                q_next_dummy = self.model_target.head.qval(q_next_convs)
                a_next = q_next_dummy.argmax(dim=-1)

            q_next, _ = self.model_target.head(q_next_convs, taus=taus_hat)
            q_next = q_next[self.batch_indices, :, a_next]
            q_target = (
                rewards.view(-1, 1)
                + self.cfg.learner.discount * (1 - terminals.view(-1, 1)) * q_next
            )

        q_hat = rearrange(q_hat, "b n -> b 1 n")
        q_target = rearrange(q_target, "b n -> b n 1")
        tau_hats = rearrange(taus_hat, "b n 1 -> b 1 n")
        loss = huber_qr_loss(q_hat, q_target, tau_hats)

        q_hat = rearrange(q_hat, "b 1 n -> b n")
        with torch.no_grad():
            # q: B X (N-1) X A
            q, _ = self.model.head(q_convs, taus=taus[:, 1:-1])
            q = q[self.batch_indices, :, actions]
            values_1 = q - q_hat[:, :-1]
            signs_1 = q.gt(torch.cat((q_hat[:, :1], q[:, :-1]), dim=1))

            values_2 = q - q_hat[:, 1:]
            signs_2 = q.lt(torch.cat((q[:, 1:], q_hat[:, -1:]), dim=1))

        # gradients: B X (N-1)
        gradients_of_taus = torch.where(signs_1, values_1, -values_1) + torch.where(
            signs_2, values_2, -values_2
        )
        gradients_of_taus = gradients_of_taus.view(-1, self.cfg.learner.iqn.F - 1)
        fraction_loss = (gradients_of_taus * taus[:, 1:-1, 0]).sum(dim=1).view(-1)
        return loss, fraction_loss

    def train_step(self, obs, actions, rewards, terminals, next_obs):
        algo = self.cfg.learner.algo
        if algo == AlgoEnum.dqn:
            loss_fn = self.train_step_dqn
        elif algo == AlgoEnum.c51:
            loss_fn = self.train_step_c51
        elif algo == AlgoEnum.qr:
            loss_fn = self.train_step_qr
        elif algo == AlgoEnum.iqn:
            loss_fn = self.train_step_iqn
        elif algo == AlgoEnum.fqf:
            loss_fn = self.train_step_fqf
        else:
            raise NotImplementedError(algo)
        loss = loss_fn(obs, actions, rewards, terminals, next_obs)

        result = {}

        if algo == AlgoEnum.fqf:
            loss, fraction_loss = loss
            fraction_loss = fraction_loss.mean()
            self.fqf_optimizer.zero_grad()
            fraction_loss.backward(retain_graph=True)
            if self.cfg.learner.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.head.fraction_net.parameters(),
                    self.cfg.learner.max_grad_norm,
                )
            self.fqf_optimizer.step()

            result["fraction_loss"] = fraction_loss.item()

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1
        result["loss"] = loss.item()

        if self.update_steps % self.cfg.learner.target_update_freq == 0:
            self.model_target = deepcopy(self.model)

        return result
