from collections import deque

import torch
import torch.nn.functional as fx

from src.common.atari_wrappers import make_deepq_env
from src.common.utils import ReplayDataset, DataLoaderX, DataPrefetcher
from src.deepq.config import Config
from src.deepq.model import NatureCNN


class Agent:
    def __init__(self, **kwargs):

        self.cfg = Config(**kwargs)

        env = make_deepq_env(self.cfg.game)
        self.action_dim = env.action_space.n
        self.state_shape = env.observation_space.shape
        self.device = torch.device('cuda:0')
        self.batch_indices = torch.arange(self.cfg.batch_size).to(self.device)
        if self.cfg.distributional:
            self.atoms = torch.linspace(self.cfg.v_min, self.cfg.v_max, self.cfg.num_atoms).to(self.device)
            self.delta_atom = (self.cfg.v_max - self.cfg.v_min) / (self.cfg.num_atoms - 1)
        if self.cfg.qr:
            self.cumulative_density = ((2 * torch.arange(self.cfg.num_atoms) + 1) /
                                       (2.0 * self.cfg.num_atoms)).to(self.device)

        self.model = NatureCNN(self.state_shape[0], self.action_dim, dueling=self.cfg.dueling,
                               noisy=self.cfg.noisy, num_atoms=self.cfg.num_atoms).to(self.device)
        self.model_target = copy.deepcopy(self.model)

        # self.models = [NatureCNN(self.state_shape[0], self.action_dim, dueling=self.cfg.dueling,
        #                          noisy=self.cfg.noisy, num_atoms=self.cfg.num_atoms).to(self.device)]
        #
        # self.optimizers = [torch.optim.Adam(model.parameters(), self.cfg.adam_lr) for model in self.models]

        self.update_steps = 0
        self.replay = deque(maxlen=self.cfg.replay_size)
        self.data_fetcher = None

    def get_data_fetcher(self):
        dataset = ReplayDataset(self.replay)
        data_loader = DataLoaderX(dataset, batch_size=self.cfg.batch_size, shuffle=True,
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
        loss = loss.sum(-1).mean()
        return loss

    def train_step_c51(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            prob_next = self.model_target(next_states).softmax(dim=-1)
            if self.cfg.double_q:
                prob_next_online = self.model(next_states).softmax(dim=-1)
                actions_next = prob_next_online.mul(self.atoms).sum(dim=-1).argmax(dim=-1)
            else:
                actions_next = prob_next.mul(self.atoms).sum(dim=-1).argmax(dim=-1)
            prob_next = prob_next[self.batch_indices, actions_next, :]

            atoms_next = rewards.unsqueeze(-1) + self.cfg.discount * (1 - terminals.unsqueeze(-1)) * self.atoms.view(1,
                                                                                                                     -1)
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
        loss = target_prob.mul(log_prob).sum(dim=-1).neg().mean()
        return loss

    def train_step_dqn(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next = self.model_target(next_states).squeeze(-1)
            if self.cfg.double_q:
                a_next = self.model(next_states).squeeze(-1).argmax(dim=-1)
            else:
                a_next = q_next.argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next]
            q_target = rewards + self.cfg.discount * (1 - terminals) * q_next

        q = self.model(states).squeeze(dim=-1)[self.batch_indices, actions]
        loss = torch.nn.functional.smooth_l1_loss(q, q_target)
        return loss

    def train_step(self):
        assert not (self.cfg.distributional and self.cfg.qr)
        try:
            data = self.data_fetcher.next()
        except Exception as e:
            self.data_fetcher = self.get_data_fetcher()
            data = self.data_fetcher.next()
            # print(e)

        frames, actions, rewards, terminals = data
        states = frames[:, :-1, :, :].float().div(255.0)
        next_states = frames[:, 1:, :, :].float().div(255.0)
        actions = actions.long()
        terminals = terminals.float()
        rewards = rewards.float()

        if self.cfg.noisy:
            self.model.reset_noise()
            self.model_target.reset_noise()

        if self.cfg.distributional:
            loss = self.train_step_c51(states, next_states, actions, terminals, rewards)
        elif self.cfg.qr:
            loss = self.train_step_qr(states, next_states, actions, terminals, rewards)
        else:
            loss = self.train_step_dqn(states, next_states, actions, terminals, rewards, model_x, model_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1

        if self.update_steps % self.cfg.target_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
        return loss.detach()
