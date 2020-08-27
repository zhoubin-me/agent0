import copy

import torch
import torch.nn.functional as fx

from src.common.atari_wrappers import make_deepq_env
from src.common.utils import DataLoaderX, DataPrefetcher
from src.deepq.config import Config
from src.deepq.model import NatureCNN
from src.deepq.replay import ReplayDataset


class Agent:
    def __init__(self, **kwargs):

        self.cfg = Config(**kwargs)
        self.cfg.update()

        env = make_deepq_env(self.cfg.game)
        self.obs_shape = env.observation_space.shape
        self.action_dim = env.action_space.n
        self.device = torch.device('cuda:0')
        # noinspection PyArgumentList
        self.batch_indices = torch.arange(self.cfg.batch_size).to(self.device)
        if self.cfg.algo == 'c51':
            self.atoms = torch.linspace(self.cfg.v_min, self.cfg.v_max, self.cfg.num_atoms).to(self.device)
            self.delta_atom = (self.cfg.v_max - self.cfg.v_min) / (self.cfg.num_atoms - 1)
        elif self.cfg.algo == 'qr':
            # noinspection PyArgumentList
            self.cumulative_density = ((2 * torch.arange(self.cfg.num_atoms) + 1) /
                                       (2.0 * self.cfg.num_atoms)).to(self.device)

        self.model = NatureCNN(self.cfg.frame_stack, self.action_dim, dueling=self.cfg.dueling,
                               noisy=self.cfg.noisy, num_atoms=self.cfg.num_atoms).to(self.device)
        self.model_target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.cfg.adam_lr)

        self.update_steps = 0
        self.replay = ReplayDataset(self.obs_shape, **kwargs)
        self.data_fetcher = None

        self.step = {
            'qr': self.train_step_qr,
            'c51': self.train_step_c51,
            'dqn': self.train_step_dqn,
            'mdqn': self.train_step_mdqn,
            'kl': self.train_step_kl_dqn,
        }

        assert self.cfg.algo in self.step

    def get_data_fetcher(self):
        if self.cfg.prioritize:
            data_loader = DataLoaderX(self.replay, batch_sampler=self.replay,
                                      num_workers=self.cfg.num_data_workers, pin_memory=self.cfg.pin_memory)
        else:
            data_loader = DataLoaderX(self.replay, batch_size=self.cfg.batch_size, shuffle=True,
                                      num_workers=self.cfg.num_data_workers, pin_memory=self.cfg.pin_memory)
        data_fetcher = DataPrefetcher(data_loader, self.device)
        return data_fetcher

    def train_step_qr(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next = self.model_target(next_states)
            if self.cfg.double_q:
                q_next_online = self.model(next_states).mean(dim=-1)
                a_next = q_next_online.argmax(dim=-1)
            else:
                a_next = q_next.mean(dim=-1).argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next, :]
            q_target = rewards.unsqueeze(-1) + self.cfg.discount * (1 - terminals.unsqueeze(-1)) * q_next

        q = self.model(states)[self.batch_indices, actions, :]
        q = q.view(self.cfg.batch_size, -1).unsqueeze(1)
        q_target = q_target.view(self.cfg.batch_size, -1).unsqueeze(-1)

        loss = fx.smooth_l1_loss(q, q_target, reduction='none')
        weights = torch.abs(self.cumulative_density.view(1, 1, -1) - (q - q_target).detach().sign().float())
        loss = loss * weights
        loss = loss.sum(-1).mean(-1)
        return loss.view(-1)

    def train_step_c51(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            prob_next = self.model_target(next_states).softmax(dim=-1)
            if self.cfg.double_q:
                prob_next_online = self.model(next_states).softmax(dim=-1)
                actions_next = prob_next_online.mul(self.atoms).sum(dim=-1).argmax(dim=-1)
            else:
                actions_next = prob_next.mul(self.atoms).sum(dim=-1).argmax(dim=-1)
            prob_next = prob_next[self.batch_indices, actions_next, :]

            atoms_next = rewards.unsqueeze(-1) + \
                         self.cfg.discount * (1 - terminals.unsqueeze(-1)) * self.atoms.view(1, -1)

            atoms_next.clamp_(self.cfg.v_min, self.cfg.v_max)
            base = (atoms_next - self.cfg.v_min) / self.delta_atom

            lo, up = base.floor().long(), base.ceil().long()
            lo[(up > 0) * (lo == up)] -= 1
            up[(lo < (self.cfg.num_atoms - 1)) * (lo == up)] += 1

            target_prob = torch.zeros_like(prob_next)
            offset = torch.linspace(0, ((self.cfg.batch_size - 1) * self.cfg.num_atoms), self.cfg.batch_size)
            offset = offset.unsqueeze(1).expand(self.cfg.batch_size, self.cfg.num_atoms).long().to(self.device)

            target_prob.view(-1).index_add_(0, (lo + offset).view(-1), (prob_next * (up.float() - base)).view(-1))
            target_prob.view(-1).index_add_(0, (up + offset).view(-1), (prob_next * (base - lo.float())).view(-1))

        log_prob = self.model(states).log_softmax(dim=-1)
        log_prob = log_prob[self.batch_indices, actions, :]
        loss = target_prob.mul(log_prob).sum(dim=-1).neg()
        return loss.view(-1)

    @staticmethod
    def log_softmax_stable(logits, tau=0.01):
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        return logits - tau * torch.logsumexp(logits / tau, dim=-1, keepdim=True)

    def train_step_mdqn(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next_logits = self.model_target(next_states)
            q_next = q_next_logits - self.log_softmax_stable(q_next_logits, self.cfg.mdqn_tau)
            q_next = q_next_logits.softmax(dim=-1).mul(q_next).sum(dim=-1)

            add_on = self.model_target(states)
            add_on = self.log_softmax_stable(add_on, self.cfg.mdqn_tau)
            add_on = add_on[self.batch_indices, actions].clamp(self.cfg.mdqn_lo, 0)

            q_target = rewards + self.cfg.mdqn_alpha * add_on + self.cfg.discount * (1 - terminals) * q_next

        q = self.model(states)[self.batch_indices, actions]
        loss = fx.smooth_l1_loss(q, q_target, reduction='none')
        return loss.view(-1)

    def train_step_kl_dqn(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next_logits = self.model_target(next_states)
            q_next = q_next_logits - self.log_softmax_stable(q_next_logits, self.cfg.mdqn_tau)
            q_next = q_next_logits.softmax(dim=-1).mul(q_next).sum(dim=-1)

            add_on = rewards + self.cfg.discount * (1 - terminals) * q_next
            q_target = self.model_target(states)
            q_target[self.batch_indices, actions] += add_on
            q_target = q_target.softmax(dim=-1)

        q = self.model(states).softmax(dim=-1)
        loss = fx.kl_div(q_target.log(), q, reduction='none').sum(dim=-1)
        return loss.view(-1)

    def train_step_dqn(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next = self.model_target(next_states)
            if self.cfg.double_q:
                a_next = self.model(next_states).argmax(dim=-1)
            else:
                a_next = q_next.argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next]
            q_target = rewards + self.cfg.discount * (1 - terminals) * q_next

        q = self.model(states)[self.batch_indices, actions]
        loss = fx.smooth_l1_loss(q, q_target, reduction='none')
        return loss.view(-1)

    def train_step(self):
        try:
            data = self.data_fetcher.next()
        except (StopIteration, AttributeError):
            self.data_fetcher = self.get_data_fetcher()
            data = self.data_fetcher.next()

        states, actions, rewards, terminals, next_states, weights, indices = data
        states = states.float().div(255.0)
        next_states = next_states.float().div(255.0)
        actions = actions.long()
        terminals = terminals.float()
        rewards = rewards.float()
        weights = weights.float()

        if self.cfg.noisy:
            self.model.reset_noise()
            self.model_target.reset_noise()

        loss = self.step[self.cfg.algo](states, next_states, actions, terminals, rewards)

        if self.cfg.prioritize:
            self.replay.update_priorities(indices.cpu(), loss.detach().cpu())
            weights /= weights.sum().add(1e-8)
            loss = loss.mul(weights).sum()
        else:
            loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()

        if self.cfg.algo == 'qr' and self.cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)

        self.optimizer.step()
        self.update_steps += 1

        if self.update_steps % self.cfg.target_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
        return loss.detach()
