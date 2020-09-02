import copy

import torch
import torch.nn.functional as fx

from agent0.common.atari_wrappers import make_deepq_env
from agent0.common.utils import DataLoaderX, DataPrefetcher
from agent0.deepq.config import Config
from agent0.deepq.model import DeepQNet
from agent0.deepq.replay import ReplayDataset


class Agent:
    def __init__(self, **kwargs):

        self.cfg = Config(**kwargs)
        self.cfg.update_atoms()

        env = make_deepq_env(self.cfg.game, frame_stack=self.cfg.frame_stack, transpose_image=True)
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

        self.model = DeepQNet(self.action_dim, **kwargs).to(self.device)
        self.model_target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.params(), self.cfg.adam_lr, eps=1e-2 / self.cfg.batch_size)

        if self.cfg.algo == 'fqf':
            self.fraction_optimizer = torch.optim.RMSprop(
                self.model.fraction_net.parameters(),
                lr=self.cfg.adam_lr / 2e4,
                alpha=0.95, eps=0.00001)

        self.update_steps = 0
        self.replay = ReplayDataset(self.obs_shape, **kwargs)
        self.data_fetcher = None

        self.step = {
            'qr': self.train_step_qr,
            'iqr': self.train_step_iqr,
            'c51': self.train_step_c51,
            'dqn': self.train_step_dqn,
            'mdqn': self.train_step_mdqn,
            'fqf': self.train_step_fqf,
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

    @staticmethod
    def calc_huber_qr_loss(q, q_target, taus):
        huber_loss = fx.smooth_l1_loss(q, q_target, reduction='none')
        loss = huber_loss * (taus - q_target.lt(q).detach().float()).abs()
        return loss.sum(-1).mean(-1).view(-1)

    @staticmethod
    def calc_fqf_q(st, model):
        if st.ndim == 4:
            convs = model.convs(st)
        else:
            convs = st

        # taus: B X (N+1) X 1, taus_hats: B X N X 1
        taus, tau_hats, _ = model.taus_prop(convs.detach())
        q_hats, _ = model(convs, iqr=True, taus=tau_hats)
        q = ((taus[:, 1:, :] - taus[:, :-1, :]) * q_hats).sum(dim=1)
        return q

    def train_step_fqf(self, states, next_states, actions, terminals, rewards):
        q_convs = self.model.convs(states)
        # taus: B X (N+1) X 1, taus_hats: B X N X 1
        taus, tau_hats, _ = self.model.taus_prop(q_convs.detach())
        # q_hat: B X N X A
        q_hat, _ = self.model(q_convs, iqr=True, taus=tau_hats)
        q_hat = q_hat[self.batch_indices, :, actions]

        # q_hat: B X 1 X N
        q_hat = q_hat.unsqueeze(1)
        # taus: B X 1 X (N+1)
        taus = taus.squeeze(-1).unsqeeze(1)

        with torch.no_grad():
            q_next_convs = self.model_target.convs(next_states)
            if self.cfg.double_q:
                q_next_online = self.calc_fqf_q(next_states, self.model)
                a_next = q_next_online.argmax(dim=-1)
            else:
                q_next_ = self.calc_fqf_q(q_next_convs, self.model_target)
                a_next = q_next_.argmax(dim=-1)

            q_next, _ = self.model_target(q_next_convs, taus=tau_hats, iqr=True)
            q_next = q_next[self.batch_indices, :, a_next]
            q_target = rewards.unsqueeze(-1) + \
                       self.cfg.discount ** self.cfg.n_step * (1 - terminals.unsqueeze(-1)) * q_next
            q_target = q_target.unsqueeze(-1)

        huber_loss = self.calc_huber_qr_loss(q_hat, q_target, taus)

        ###
        with torch.no_grad():
            # q: B X (N-1) X A
            q, _ = self.model(q_convs, iqr=True, taus=taus[1:-1])
            q = q[self.batch_indices, :, actions]
            values_1 = q - q_hat[:, :-1]
            signs_1 = q.gt(torch.cat((q_hat[:, :1], q[:, :-1]), dim=1))

            values_2 = q - q_hat[:, 1:]
            signs_2 = q.lt(torch.cat((q[:, 1:], q_hat[:, -1:]), dim=1))

        # gradients: B X (N-1)
        gradients_of_taus = (torch.where(signs_1, values_1, -values_1)
                             + torch.where(signs_2, values_2, -values_2)).view(self.cfg.batch_size, self.cfg.N_fqf - 1)

        fraction_loss = (gradients_of_taus * taus[:, 1:-1]).sum(dim=1).view(-1)
        return huber_loss, fraction_loss

    def train_step_iqr(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next_convs = self.model_target.convs(next_states)
            if self.cfg.double_q:
                q_next_online, _ = self.model(next_states, iqr=True, n=self.cfg.K_iqr)
                a_next = q_next_online.mean(dim=1).argmax(dim=-1)
            else:
                q_next_, _ = self.model_target(q_next_convs, iqr=True, n=self.cfg.K_iqr)
                a_next = q_next_.mean(dim=1).argmax(dim=-1)

            q_next, taus_dash = self.model_target(q_next_convs, iqr=True, n=self.cfg.N_iqr_dash)
            q_next = q_next[self.batch_indices, :, a_next]
            q_target = rewards.unsqueeze(-1) + \
                       self.cfg.discount ** self.cfg.n_step * (1 - terminals.unsqueeze(-1)) * q_next

        q, taus = self.model(states, iqr=True, n=self.cfg.N_iqr)
        q = q[self.batch_indices, :, actions]
        q = q.unsqueeze(1)
        taus = taus.squeeze(-1).unsqueeze(1)
        q_target = q_target.unsqueeze(-1)
        loss = self.calc_huber_qr_loss(q, q_target, taus)
        return loss.view(-1)

    def train_step_qr(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next = self.model_target(next_states)
            if self.cfg.double_q:
                q_next_online = self.model(next_states).mean(dim=-1)
                a_next = q_next_online.argmax(dim=-1)
            else:
                a_next = q_next.mean(dim=-1).argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next, :]
            q_target = rewards.unsqueeze(-1) + \
                       self.cfg.discount ** self.cfg.n_step * (1 - terminals.unsqueeze(-1)) * q_next

        q = self.model(states)[self.batch_indices, actions, :]
        q = q.unsqueeze(1)
        q_target = q_target.unsqueeze(-1)

        loss = self.calc_huber_qr_loss(q, q_target, self.cumulative_density.view(1, 1, -1))
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
                         self.cfg.discount ** self.cfg.n_step * (1 - terminals.unsqueeze(-1)) * self.atoms.view(1, -1)

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

            q_target = rewards + self.cfg.mdqn_alpha * add_on + \
                       self.cfg.discount ** self.cfg.n_step * (1 - terminals) * q_next

        q = self.model(states)[self.batch_indices, actions]
        loss = fx.smooth_l1_loss(q, q_target, reduction='none')
        return loss.view(-1)

    def train_step_dqn(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next = self.model_target(next_states)
            if self.cfg.double_q:
                a_next = self.model(next_states).argmax(dim=-1)
            else:
                a_next = q_next.argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next]
            q_target = rewards + self.cfg.discount ** self.cfg.n_step * (1 - terminals) * q_next

        q = self.model(states)[self.batch_indices, actions]
        loss = fx.smooth_l1_loss(q, q_target, reduction='none')
        return loss.view(-1)

    def train_step(self):
        try:
            data = self.data_fetcher.next()
        except (StopIteration, AttributeError):
            self.data_fetcher = self.get_data_fetcher()
            data = self.data_fetcher.next()

        frames, actions, rewards, terminals, weights, indices = data
        states = frames[:, :self.cfg.frame_stack, :, :].float().div(255.0)
        next_states = frames[:, -self.cfg.frame_stack:, :, :].float().div(255.0)
        actions = actions.long()
        terminals = terminals.float()
        rewards = rewards.float()
        weights = weights.float()

        if self.cfg.noisy:
            self.model.reset_noise()
            self.model_target.reset_noise()

        loss = self.step[self.cfg.algo](states, next_states, actions, terminals, rewards)
        if self.cfg.algo == 'fqf':
            loss, fraction_loss = loss
        else:
            fraction_loss = None

        if self.cfg.prioritize:
            self.replay.update_priorities(indices.cpu(), loss.detach().cpu())
            weights /= weights.sum().add(1e-8)
            loss = loss.mul(weights).sum()
            fraction_loss = fraction_loss.mul(weights).sum() if fraction_loss is not None else None
        else:
            loss = loss.mean()
            fraction_loss = fraction_loss.mean() if fraction_loss is not None else None

        if fraction_loss is not None:
            self.fraction_optimizer.zero_grad()
            fraction_loss.backward(retain_graph=True)
            self.fraction_optimizer.step()

        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.algo == 'qr' and self.cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
        self.optimizer.step()
        self.update_steps += 1

        if self.update_steps % self.cfg.target_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
        return loss.detach()
