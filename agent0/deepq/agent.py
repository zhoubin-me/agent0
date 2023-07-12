import copy

import torch
import torch.nn.functional as fx
from torch.distributions import Categorical, Normal

from agent0.common.atari_wrappers import make_atari
from agent0.common.MixtureSameFamily import MixtureSameFamily
from agent0.common.utils import DataLoaderX, DataPrefetcher
from agent0.deepq.config import Config
from agent0.deepq.model import DeepQNet
from agent0.deepq.new_config import ExpConfig
from agent0.deepq.replay import ReplayDataset


class Agent:
    def __init__(self, **kwargs):
        self.cfg = Config(**kwargs)
        self.cfg.update_atoms()

        dummy_env = make_atari(self.cfg.game, 1)
        self.obs_shape = dummy_env.observation_space.shape[1:]
        self.action_dim = dummy_env.action_space[0].n
        dummy_env.close()

        self.device = torch.device("cuda:0")
        # noinspection PyArgumentList
        self.batch_indices = torch.arange(self.cfg.batch_size).to(self.device)

        self.model = DeepQNet(self.action_dim, **kwargs).to(self.device)
        self.model_target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(
            self.model.params(), self.cfg.adam_lr, eps=1e-2 / self.cfg.batch_size
        )

        if self.cfg.algo in ["fqf"]:
            self.fraction_optimizer = torch.optim.RMSprop(
                self.model.fraction_net.parameters(),
                lr=self.cfg.adam_lr / 2e4,
                alpha=0.95,
                eps=0.00001,
            )

        self.update_steps = 0
        self.replay = ReplayDataset(ExpConfig())
        self.data_fetcher = None

        self.step = {
            "qr": self.train_step_qr,
            "iqr": self.train_step_iqr,
            "c51": self.train_step_c51,
            "dqn": self.train_step_dqn,
            "mdqn": self.train_step_mdqn,
            "fqf": self.train_step_fqf,
            "gmm": self.train_step_gmm,
        }

        assert self.cfg.algo in self.step

    def get_data_fetcher(self):
        if self.cfg.prioritize:
            data_loader = DataLoaderX(
                self.replay,
                batch_sampler=self.replay,
                num_workers=self.cfg.num_data_workers,
                pin_memory=self.cfg.pin_memory,
            )
        else:
            data_loader = DataLoaderX(
                self.replay,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_data_workers,
                pin_memory=self.cfg.pin_memory,
            )
        data_fetcher = DataPrefetcher(data_loader, self.device)
        return data_fetcher

    @staticmethod
    def calc_huber_qr_loss(q, q_target, taus):
        huber_loss = fx.smooth_l1_loss(q, q_target, reduction="none")
        loss = huber_loss * (taus - q_target.lt(q).detach().float()).abs()
        return loss.sum(-1).mean(-1).view(-1)

    def train_step_fqf(self, states, next_states, actions, terminals, rewards):
        q_convs = self.model.convs(states)
        # taus: B X (N+1) X 1, taus_hats: B X N X 1
        taus, tau_hats, _ = self.model.taus_prop(q_convs.detach())
        # q_hat: B X N X A
        q_hat, _ = self.model.forward_iqr(q_convs, taus=tau_hats)
        q_hat = q_hat[self.batch_indices, :, actions]

        with torch.no_grad():
            q_next_convs = self.model_target.convs(next_states)
            if self.cfg.double_q:
                q_next_online = self.model.calc_fqf_q(next_states)
                a_next = q_next_online.argmax(dim=-1)
            else:
                q_next_ = self.model_target.calc_fqf_q(q_next_convs)
                a_next = q_next_.argmax(dim=-1)

            q_next, _ = self.model_target.forward_iqr(q_next_convs, taus=tau_hats)
            q_next = q_next[self.batch_indices, :, a_next]
            q_target = rewards.unsqueeze(-1).add(
                self.cfg.discount**self.cfg.n_step
                * (1 - terminals.unsqueeze(-1))
                * q_next
            )

        # q_hat: B X 1 X N
        # tau_hats: B X 1 X N
        # q_target: B X N X 1
        huber_loss = self.calc_huber_qr_loss(
            q_hat.unsqueeze(1),
            q_target.unsqueeze(-1),
            tau_hats.squeeze(-1).unsqueeze(1),
        )

        ###
        with torch.no_grad():
            # q: B X (N-1) X A
            q, _ = self.model.forward_iqr(q_convs, taus=taus[:, 1:-1])
            q = q[self.batch_indices, :, actions]
            values_1 = q - q_hat[:, :-1]
            signs_1 = q.gt(torch.cat((q_hat[:, :1], q[:, :-1]), dim=1))

            values_2 = q - q_hat[:, 1:]
            signs_2 = q.lt(torch.cat((q[:, 1:], q_hat[:, -1:]), dim=1))

        # gradients: B X (N-1)
        gradients_of_taus = (
            torch.where(signs_1, values_1, -values_1)
            + torch.where(signs_2, values_2, -values_2)
        ).view(self.cfg.batch_size, self.cfg.N_fqf - 1)
        # import pdb
        # pdb.set_trace()
        fraction_loss = (gradients_of_taus * taus[:, 1:-1, 0]).sum(dim=1).view(-1)
        return huber_loss, fraction_loss

    def train_step_iqr(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next_convs = self.model_target.convs(next_states)
            if self.cfg.double_q:
                q_next_online, _ = self.model.forward_iqr(next_states, n=self.cfg.K_iqr)
                a_next = q_next_online.mean(dim=1).argmax(dim=-1)
            else:
                q_next_, _ = self.model_target.forward_iqr(
                    q_next_convs, n=self.cfg.K_iqr
                )
                a_next = q_next_.mean(dim=1).argmax(dim=-1)

            q_next, taus_dash = self.model_target.forward_iqr(
                q_next_convs, n=self.cfg.N_iqr_dash
            )
            q_next = q_next[self.batch_indices, :, a_next]
            q_target = rewards.unsqueeze(-1).add(
                self.cfg.discount**self.cfg.n_step
                * (1 - terminals.unsqueeze(-1))
                * q_next
            )

        q, taus = self.model.forward_iqr(states, n=self.cfg.N_iqr)
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
            q_target = rewards.unsqueeze(-1).add(
                self.cfg.discount**self.cfg.n_step
                * (1 - terminals.unsqueeze(-1))
                * q_next
            )

        q = self.model(states)[self.batch_indices, actions, :]
        q = q.unsqueeze(1)
        q_target = q_target.unsqueeze(-1)

        loss = self.calc_huber_qr_loss(
            q, q_target, self.model.cumulative_density.view(1, 1, -1)
        )
        return loss.view(-1)

    def train_step_c51(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            prob_next = self.model_target(next_states).softmax(dim=-1)
            if self.cfg.double_q:
                prob_next_online = self.model(next_states).softmax(dim=-1)
                actions_next = (
                    prob_next_online.mul(self.model.atoms).sum(dim=-1).argmax(dim=-1)
                )
            else:
                actions_next = (
                    prob_next.mul(self.model.atoms).sum(dim=-1).argmax(dim=-1)
                )
            prob_next = prob_next[self.batch_indices, actions_next, :]

            atoms_next = rewards.unsqueeze(-1).add(
                self.cfg.discount**self.cfg.n_step
                * (1 - terminals.unsqueeze(-1))
                * self.model.atoms.view(1, -1)
            )

            atoms_next.clamp_(self.cfg.v_min, self.cfg.v_max)
            base = (atoms_next - self.cfg.v_min) / self.model.delta_atom

            lo, up = base.floor().long(), base.ceil().long()
            lo[(up > 0) * (lo == up)] -= 1
            up[(lo < (self.cfg.num_atoms - 1)) * (lo == up)] += 1

            target_prob = torch.zeros_like(prob_next)
            offset = torch.linspace(
                0, ((self.cfg.batch_size - 1) * self.cfg.num_atoms), self.cfg.batch_size
            )
            offset = (
                offset.unsqueeze(1)
                .expand(self.cfg.batch_size, self.cfg.num_atoms)
                .long()
                .to(self.device)
            )

            target_prob.view(-1).index_add_(
                0, (lo + offset).view(-1), (prob_next * (up.float() - base)).view(-1)
            )
            target_prob.view(-1).index_add_(
                0, (up + offset).view(-1), (prob_next * (base - lo.float())).view(-1)
            )

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
            q_next = q_next_logits - self.log_softmax_stable(
                q_next_logits, self.cfg.mdqn_tau
            )
            q_next = q_next_logits.softmax(dim=-1).mul(q_next).sum(dim=-1)

            add_on = self.model_target(states)
            add_on = self.log_softmax_stable(add_on, self.cfg.mdqn_tau)
            add_on = add_on[self.batch_indices, actions].clamp(self.cfg.mdqn_lo, 0)

            q_target = rewards.add(self.cfg.mdqn_alpha * add_on).add(
                self.cfg.discount**self.cfg.n_step * (1 - terminals) * q_next
            )

        q = self.model(states)[self.batch_indices, actions]
        loss = fx.smooth_l1_loss(q, q_target, reduction="none")
        return loss.view(-1)

    def train_step_gmm(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next_mean, q_next_std, q_next_weight = self.model_target.forward_gmm(
                next_states
            )
            if self.cfg.double_q:
                a_next = self.model.calc_gmm_q(next_states).argmax(dim=-1)
            else:
                a_next = q_next_mean.mul(q_next_weight).sum(-1).argmax(dim=-1)
            q_next_mean = q_next_mean[self.batch_indices, a_next, :]
            q_next_std = q_next_std[self.batch_indices, a_next, :]
            q_next_weight = q_next_weight[self.batch_indices, a_next, :]
            q_target_mean = rewards.view(-1, 1).add(
                self.cfg.discount**self.cfg.n_step
                * (1 - terminals.view(-1, 1))
                * q_next_mean
            )

            comp = Normal(q_target_mean.squeeze(), q_next_std)
            mix = Categorical(q_next_weight.squeeze())
            target_gmm = MixtureSameFamily(mix, comp)
            q_target_sample = target_gmm.sample_n(self.cfg.gmm_num_samples)

        q_mean, q_std, q_weight = map(
            lambda x: x[self.batch_indices, actions, :], self.model.forward_gmm(states)
        )

        comp = Normal(q_mean.squeeze(), q_std.squeeze())
        mix = Categorical(q_weight.squeeze())
        q_gmm = MixtureSameFamily(mix, comp)
        loss = q_gmm.log_prob(q_target_sample).neg().mean(0)
        return loss.view(-1)

    def train_step_dqn(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next = self.model_target(next_states)
            if self.cfg.double_q:
                a_next = self.model(next_states).argmax(dim=-1)
            else:
                a_next = q_next.argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next]
            q_target = (
                rewards
                + self.cfg.discount**self.cfg.n_step * (1 - terminals) * q_next
            )

        q = self.model(states)[self.batch_indices, actions]
        loss = fx.smooth_l1_loss(q, q_target, reduction="none").view(-1)
        return loss

    def train_best_ep(self, st, at):
        qs = self.model(st)
        best_ep_loss = fx.cross_entropy(qs, at)
        return best_ep_loss

    def train_step(self, data=None):
        if data is None:
            try:
                data = self.data_fetcher.next()
            except (StopIteration, AttributeError):
                self.data_fetcher = self.get_data_fetcher()
                data = self.data_fetcher.next()

        frames, actions, rewards, terminals = data
        frames = frames.reshape(self.cfg.batch_size, -1, *self.obs_shape[1:])
        states = frames[:, : self.cfg.frame_stack, :, :].float().div(255.0)
        next_states = frames[:, -self.cfg.frame_stack :, :, :].float().div(255.0)
        actions = actions.long()
        terminals = terminals.float()
        rewards = rewards.float()
        # weights = weights.float()
        # indices = indices.long()

        if self.cfg.noisy:
            self.model.reset_noise()
            self.model_target.reset_noise()

        loss = self.step[self.cfg.algo](
            states, next_states, actions, terminals, rewards
        )
        if self.cfg.algo in ["fqf"]:
            loss, fraction_loss = loss
        else:
            fraction_loss = None

        if self.cfg.prioritize:
            self.replay.update_priorities(indices.cpu(), loss.detach().cpu())
            weights /= weights.sum().add(1e-8)
            loss = loss.mul(weights).sum()
            fraction_loss = (
                fraction_loss.mul(weights).sum() if fraction_loss is not None else None
            )
        else:
            loss = loss.mean()
            fraction_loss = fraction_loss.mean() if fraction_loss is not None else None

        if self.cfg.cor_loss:
            phi_norm = self.model.phi - self.model.phi.mean(dim=0, keepdim=True)
            cor_loss = (
                (phi_norm.unsqueeze(1) * phi_norm.unsqueeze(-1))
                .pow(2)
                .triu(diagonal=1)
                .mean()
            )
            loss += cor_loss * self.cfg.cor_reg
        elif self.cfg.cor_loss2:
            phi_norm = self.model.phi / (self.model.phi.max(dim=0, keepdim=True) + 1e-5)
            cor_loss = (
                (phi_norm.unsqueeze(1) * phi_norm.unsqueeze(-1))
                .pow(2)
                .triu(diagonal=1)
                .mean()
            )
            loss += cor_loss * self.cfg.cor_reg
        elif self.cfg.cor_loss3:
            _, s, _ = torch.svd(self.model.phi)
            cor_loss = s[:256].sum().div(s.sum())
            loss += cor_loss
        else:
            cor_loss = None

        if fraction_loss is not None:
            self.fraction_optimizer.zero_grad()
            fraction_loss.backward(retain_graph=True)
            if self.cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.fraction_net.parameters(), self.cfg.clip_grad_norm
                )
            self.fraction_optimizer.step()

        if not torch.isnan(loss).any():
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.params(), self.cfg.clip_grad_norm
                )
            self.optimizer.step()
            self.update_steps += 1
        else:
            loss = torch.zeros_like(loss).to(loss)
            self.update_steps += 1

        if self.update_steps % self.cfg.target_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
        return {
            "loss": loss.detach(),
            "cor_loss": cor_loss.detach() if cor_loss is not None else 0,
            "fraction_loss": fraction_loss.detach() if fraction_loss is not None else 0,
        }


if __name__ == "__main__":
    agent = Agent(game="Breakout")
    st = torch.randn(512, 4, 84, 84).to(0)
    at = torch.randint(0, 4, (512,)).to(0)
    rt = torch.randn(512).to(0)
    dt = torch.zeros(512).to(0)
    # data = frames, actions, rewards, terminals, weights, indices, best_frames, best_actions
    data = torch.cat((st, st), dim=1), at, rt, dt, rt, dt, st, at
    loss = agent.train_step(data)
    print(loss)
