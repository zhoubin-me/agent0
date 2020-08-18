import json
import os
import time
from collections import deque

import numpy as np
import ray
import torch
import torch.nn.functional as F
from ray import tune
from ray.tune.trial import ExportFormat

from src.common.utils import LinearSchedule, DataPrefetcher, ReplayDataset, DataLoaderX, pprint, make_env
from src.common.vec_env import ShmemVecEnv
from src.deepq.model import NatureCNN


def default_hyperparams():
    params = dict(
        game='Breakout',
        double_q=True,
        dueling=True,
        noisy=False,
        distributional=False,
        qr=False,

        v_max=10,
        v_min=-10,
        num_atoms=1,

        reset_noise_freq=5,
        exp_name='atari_deepq',
        save_prefix="ckpt",
        pin_memory=True,

        num_actors=8,
        num_envs=16,
        num_data_workers=4,

        adam_lr=5e-4,
        adamw=True,

        batch_size=512,
        discount=0.99,
        replay_size=int(1e6),
        exploration_ratio=0.1,
        min_eps=0.01,

        target_update_freq=500,
        agent_train_freq=10,

        total_steps=int(2.5e7),
        start_training_step=int(2e4),
        epoches=2500,
        random_seed=1234)

    return params


@ray.remote(num_gpus=0.1)
class Actor:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.rank == self.num_actors:
            # Testing
            self.envs = ShmemVecEnv([lambda: make_env(self.game, False, False)
                                     for _ in range(self.num_envs)], context='fork')
        else:
            # Training
            self.envs = ShmemVecEnv([lambda: make_env(self.game, True, True)
                                     for _ in range(self.num_envs)], context='fork')
        self.action_dim = self.envs.action_space.n
        self.state_shape = self.envs.observation_space.shape

        self.device = torch.device('cuda:0')
        if self.distributional:
            self.atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.model = NatureCNN(self.state_shape[0], self.action_dim,
                               dueling=self.dueling, noisy=self.noisy, num_atoms=self.num_atoms).to(self.device)

        self.obs = self.envs.reset()


    def sample(self, steps, epsilon, state_dict, testing=False, test_episodes=100):
        self.model.load_state_dict(state_dict)
        if testing:
            self.obs = self.envs.reset()

        replay = deque(maxlen=self.replay_size)
        Rs, Qs = [], []
        tic = time.time()
        step = 0
        while True:
            step += 1
            action_random = np.random.randint(0, self.action_dim, self.num_envs)
            if self.noisy and step % self.reset_noise_freq == 0:
                self.model.reset_noise()

            with torch.no_grad():
                st = torch.from_numpy(np.array(self.obs)).to(self.device).float().div(255.0)

                if self.distributional:
                    qs_prob = self.model(st).softmax(dim=-1)
                    qs = qs_prob.mul(self.atoms).sum(dim=-1)
                elif self.qr:
                    qs = self.model(st).mean(dim=-1)
                else:
                    qs = self.model(st).squeeze(-1)

            qs_max, qs_argmax = qs.max(dim=-1)
            action_greedy = qs_argmax.tolist()
            Qs.append(qs_max.mean().item())
            action = [act_greedy if p > epsilon else act_random for p, act_random, act_greedy in
                      zip(np.random.rand(self.num_envs), action_random, action_greedy)]

            obs_next, reward, done, info = self.envs.step(action)
            frames = np.zeros((self.num_envs, self.state_shape[0] + 1, *self.state_shape[1:]), dtype=np.uint8)
            frames[:, :-1, :, :] = self.obs
            frames[:, -1, :, :] = obs_next[:, -1, :, :]
            if not testing:
                for entry in zip(frames, action, reward, done):
                    replay.append(entry)
            self.obs = obs_next

            for inf in info:
                if 'real_reward' in inf:
                    Rs.append(inf['real_reward'])

            if testing and len(Rs) > test_episodes:
                break
            if not testing and step > steps:
                break

        toc = time.time()
        return replay, Rs, Qs, self.rank, len(replay) / (toc - tic)

    def get_state_dict(self):
        return self.model.state_dict()

    def close_envs(self):
        self.envs.close()


class Agent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.envs = make_env(self.game)
        self.action_dim = self.envs.action_space.n
        self.state_shape = self.envs.observation_space.shape

        self.device = torch.device('cuda:0')
        self.batch_indices = torch.arange(self.batch_size).to(self.device)
        if self.distributional:
            self.atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
            self.delta_atom = (self.v_max - self.v_min) / (self.num_atoms - 1)
        if self.qr:
            self.cumulative_density = ((2 * torch.arange(self.num_atoms) + 1) / (2.0 * self.num_atoms)).to(self.device)

        self.model = NatureCNN(self.state_shape[0], self.action_dim, dueling=self.dueling,
                               noisy=self.noisy, num_atoms=self.num_atoms).to(self.device)
        self.model_target = NatureCNN(self.state_shape[0], self.action_dim, dueling=self.dueling,
                                      noisy=self.noisy, num_atoms=self.num_atoms).to(self.device)

        if self.adamw:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), self.adam_lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.adam_lr)
        self.update_steps = 0
        self.replay = deque(maxlen=self.replay_size)

    def get_datafetcher(self):
        dataset = ReplayDataset(self.replay)
        self.dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_data_workers, pin_memory=self.pin_memory)
        datafetcher = DataPrefetcher(self.dataloader, self.device)
        return datafetcher

    def train_step_qr(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next = self.model_target(next_states)
            if self.double_q:
                q_next_online = self.model(next_states).mean(dim=-1)
                a_next = q_next_online.argmax(dim=-1)
            else:
                a_next = q_next.mean(dim=-1).argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next, :]
            q_target = rewards.unsqueeze(-1) + self.discount * (1 - terminals.unsqueeze(-1)) * q_next

        q = self.model(states)[self.batch_indices, actions, :]
        q = q.view(self.batch_size, -1).unsqueeze(1)
        q_target = q_target.view(self.batch_size, -1).unsqueeze(-1)

        loss = F.smooth_l1_loss(q, q_target, reduction='none')
        weights = torch.abs(self.cumulative_density.view(1, 1, -1) - (q - q_target).detach().sign().float())
        loss = loss * weights
        loss = loss.sum(-1).mean()
        return loss

    def train_step_c51(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            prob_next = self.model_target(next_states).softmax(dim=-1)
            if self.double_q:
                prob_next_online = self.model(next_states).softmax(dim=-1)
                actions_next = prob_next_online.mul(self.atoms).sum(dim=-1).argmax(dim=-1)
            else:
                actions_next = prob_next.mul(self.atoms).sum(dim=-1).argmax(dim=-1)
            prob_next = prob_next[self.batch_indices, actions_next, :]

            atoms_next = rewards.unsqueeze(-1) + self.discount * (1 - terminals.unsqueeze(-1)) * self.atoms.view(1, -1)
            atoms_next.clamp_(self.v_min, self.v_max)
            b = (atoms_next - self.v_min) / self.delta_atom

            l, u = b.floor().long(), b.ceil().long()
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1

            target_prob = torch.zeros_like(prob_next)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.num_atoms), self.batch_size)
            offset = offset.unsqueeze(1).expand(self.batch_size, self.num_atoms).long().to(self.device)

            target_prob.view(-1).index_add_(0, (l + offset).view(-1), (prob_next * (u.float() - b)).view(-1))
            target_prob.view(-1).index_add_(0, (u + offset).view(-1), (prob_next * (b - l.float())).view(-1))

        log_prob = self.model(states).log_softmax(dim=-1)
        log_prob = log_prob[self.batch_indices, actions, :]
        loss = target_prob.mul(log_prob).sum(dim=-1).neg().mean()
        return loss

    def train_step_dqn(self, states, next_states, actions, terminals, rewards):
        with torch.no_grad():
            q_next = self.model_target(next_states).squeeze(-1)
            if self.double_q:
                a_next = self.model(next_states).squeeze(-1).argmax(dim=-1)
            else:
                a_next = q_next.argmax(dim=-1)
            q_next = q_next[self.batch_indices, a_next]
            q_target = rewards + self.discount * (1 - terminals) * q_next

        q = self.model(states).squeeze(dim=-1)[self.batch_indices, actions]
        loss = F.smooth_l1_loss(q, q_target)
        return loss


    def train_step(self):
        assert (self.distributional and self.qr) == False

        try:
            data = self.prefetcher.next()
        except:
            self.prefetcher = self.get_datafetcher()
            data = self.prefetcher.next()

        frames, actions, rewards, terminals = data
        states = frames[:, :-1, :, :].float().div(255.0)
        next_states = frames[:, 1:, :, :].float().div(255.0)
        actions = actions.long()
        terminals = terminals.float()
        rewards = rewards.float()

        if self.noisy:
            self.model.reset_noise()
            self.model_target.reset_noise()

        if self.distributional:
            loss = self.train_step_c51(states, next_states, actions, terminals, rewards)
        elif self.qr:
            loss = self.train_step_qr(states, next_states, actions, terminals, rewards)
        else:
            loss = self.train_step_dqn(states, next_states, actions, terminals, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1

        if self.update_steps % self.target_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
        return loss.detach()


class Trainer(tune.Trainable):
    def _setup(self, config):
        kwargs = default_hyperparams()
        for k, v in config.items():
            kwargs[k] = v

        for k, v in kwargs.items():
            setattr(self, k, v)

        print("input args:\n", json.dumps(kwargs, indent=4, separators=(",", ":")))

        self.agent = Agent(**kwargs)
        self.epsilon_schedule = LinearSchedule(1.0, self.min_eps, int(self.total_steps * self.exploration_ratio))
        self.epsilon = 1.0
        self.actors = [Actor.remote(rank=rank, **kwargs) for rank in range(self.num_actors + 1)]
        self.tester = self.actors[-1]

        self.steps_per_epoch = self.total_steps // self.epoches
        self.actor_steps = self.total_steps // (self.epoches * self.num_envs * self.num_actors)

        self.sample_ops = [a.sample.remote(self.actor_steps, 1.0, self.agent.model.state_dict())
                           for a in self.actors[:-1]]

        self.sample_ops.append(
            self.tester.sample.remote(
                self.actor_steps,
                self.min_eps,
                self.agent.model.state_dict(),
                testing=True))

        self.frame_count = 0
        self.lr_updated = False
        self.Rs, self.Qs, self.TRs, self.Ls = [], [], [], []
        self.best = float('-inf')

    def _train(self):
        done_id, self.sample_ops = ray.wait(self.sample_ops)
        data = ray.get(done_id)
        local_replay, Rs, Qs, rank, fps = data[0]
        if rank < self.num_actors:
            # Actors
            self.agent.replay.extend(local_replay)
            self.epsilon = self.epsilon_schedule(len(local_replay))

            self.sample_ops.append(
                self.actors[rank].sample.remote(self.actor_steps, self.epsilon, self.agent.model.state_dict()))
            self.frame_count += len(local_replay)
            self.Rs += Rs
            self.Qs += Qs
            # Start training at
            if self.frame_count > self.start_training_step:
                loss = [self.agent.train_step() for _ in range(self.agent_train_freq)]
                loss = torch.stack(loss)
                self.Ls += loss.tolist()
        else:
            # Tester
            self.sample_ops.append(
                self.tester.sample.remote(
                    self.actor_steps,
                    self.min_eps,
                    self.agent.model.state_dict(),
                    testing=True))

            if len(Rs) > 0 and np.mean(Rs) > self.best:
                self.best = np.mean(Rs)
                print(f"{self.game} updated Best Ep Reward: {self.best}")
                torch.save({
                    'model': ray.get(self.tester.get_state_dict.remote()),
                    'Ls': self.Ls,
                    'Rs': self.Rs,
                    'Qs': self.Qs,
                    'TRs': self.TRs
                }, './best.pth')

            self.TRs += Rs




        result = dict(
            game=self.game,
            time_past=self._time_total,
            epsilon=self.epsilon,
            adam_lr=self.adam_lr,
            frames=self.frame_count,
            speed=self.frame_count / (self._time_total + 1),
            time_remain=(self.total_steps - self.frame_count) / (self.frame_count / (self._time_total + 1)),
            loss=np.mean(self.Ls[-20:]) if len(self.Ls) > 0 else 0,
            ep_reward_test=np.mean(self.TRs[-20:]) if len(self.TRs) > 0 else 0,
            ep_reward_train=np.mean(self.Rs[-20:]) if len(self.Rs) > 0 else 0,
            ep_reward_train_max=np.max(self.Rs) if len(self.Rs) > 0 else 0,
            ep_reward_test_max=np.max(self.TRs) if len(self.TRs) > 0 else 0,
            qmax=np.mean(self.Qs[-100:]) if len(self.Qs) > 0 else 0
        )
        return result

    def _save(self, checkpoint_dir):
        return {
            'model': self.agent.model.state_dict(),
            'optim': self.agent.optimizer.state_dict(),
            'model_target': self.agent.model_target.state_dict(),
            'Ls': self.Ls,
            'Rs': self.Rs,
            'Qs': self.Qs,
            'TRs': self.TRs,
            'frame_count': self.frame_count,
        }

    def _restore(self, checkpoint):
        self.agent.model.load_state_dict(checkpoint['model'])
        self.agent.model_target.load_state_dict(checkpoint['model_target'])
        self.agent.optimizer.load_state_dict(checkpoint['optim'])
        self.Ls = checkpoint['Ls']
        self.Qs = checkpoint['Qs']
        self.Rs = checkpoint['Rs']
        self.TRs = checkpoint['TRs']
        self.frame_count = checkpoint['frame_count']
        self.epsilon_schedule(self.frame_count)

    def _export_model(self, export_formats, export_dir):
        if export_formats == [ExportFormat.MODEL]:
            path = os.path.join(export_dir, "exported_models")
            torch.save({
                "model": self.agent.model.state_dict(),
                "optim": self.agent.optimizer.state_dict()
            }, path)
            return {ExportFormat.MODEL: path}
        else:
            raise ValueError("unexpected formats: " + str(export_formats))

    def reset_config(self, new_config):
        if "adam_lr" in new_config:
            self.adam_lr = new_config['adam_lr']
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = new_config['adam_lr']

        self.config = new_config
        return True

    def _stop(self):
        if self.frame_count > self.total_steps:
            print("Final Testing")
            output = ray.get([a.sample.remote(self.actor_steps,
                                              self.agent.model.state_dict(),
                                              testing=True,
                                              test_episodes=10) for a in self.actors])

            FTRs = []
            for _, Rs, Qs, rank, fps in output:
                FTRs += Rs

            print(f"Final Test Result: {np.mean(FTRs)}\t{np.std(FTRs)}\t{np.max(FTRs)}\t{len(FTRs)}")
            torch.save({
                'model': self.agent.model.state_dict(),
                'FTRs': FTRs,
                'Ls': self.Ls,
                'Rs': self.Rs,
                'Qs': self.Qs,
                'TRs': self.TRs
            }, './final.pth')
        try:
            ray.get([a.close_envs.remote() for a in self.actors])
            self.agent.dataloader._shutdown_workers()
        except:
            pass


    def logstat(self):
        rem_time = (self.total_steps - self.frame_count) / (self.frame_count / self._time_total)
        print("=" * 105)
        print(f"Epoch:[{self.frame_count // self.steps_per_epoch:4d}/{self.epoches}]\t"
              f"Game: {self.game:<10s}\t"
              f"TimeElapsed: {self._time_total:6.0f}\t\t"
              f"TimeRemEst: {rem_time:6.0f}\t\n"
              f"FrameCount:{self.frame_count:8d}\t"
              f"UpdateCount:{self.iteration:6d}\t"
              f"Epsilon:{self.epsilon:6.4}")
        print('-' * 105)
        pprint("Training EP Reward ", self.Rs[-1000:])
        pprint("Loss               ", self.Ls[-1000:])
        pprint("Qmax               ", self.Qs[-1000:])
        pprint("Test EP Reward     ", self.TRs[-1000:])
        print("=" * 105)
        print(" " * 105)
        print(" " * 105)
