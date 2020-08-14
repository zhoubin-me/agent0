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
        prioritize=True,
        distributional=True,
        noisy=True,
        exp_name='atari_deepq',
        save_prefix="ckpt",
        pin_memory=False,

        num_actors=8,
        num_envs=16,
        num_data_workers=4,

        v_max=10,
        v_min=-10,
        num_atoms=51,

        adam_lr=5e-4,
        adam_eps=1.5e-4,

        batch_size=512,
        discount=0.99,
        replay_size=int(1e6),
        exploration_ratio=0.1,

        target_update_freq=500,
        agent_train_freq=10,

        start_training_step=int(2e4),
        total_steps=int(1e7),
        epoches=1000,
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
        self.atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.model = NatureCNN(self.state_shape[0], self.action_dim, self.dueling).to(self.device)

        self.R = np.zeros(self.num_envs)
        self.obs = self.envs.reset()

    def sample(self, steps, epsilon, state_dict):
        self.model.load_state_dict(state_dict)
        replay = deque(maxlen=self.replay_size)
        Rs, Qs = [], []
        tic = time.time()
        for _ in range(steps):
            action_random = np.random.randint(0, self.action_dim, self.num_envs)

            with torch.no_grad():
                st = torch.from_numpy(np.array(self.obs)).to(self.device).float().div(255.0)
                qs_prob, _ = self.model(st)
                qs = qs_prob.mul(self.atoms).sum(dim=-1)

            qs_max, qs_argmax = qs.max(dim=-1)
            action_greedy = qs_argmax.tolist()
            Qs.append(qs_max.mean().item())
            action = [act_greedy if p > epsilon else act_random for p, act_random, act_greedy in
                      zip(np.random.rand(self.num_envs), action_random, action_greedy)]

            obs_next, reward, done, info = self.envs.step(action)
            frames = np.zeros((self.num_envs, self.state_shape[0] + 1, *self.state_shape[1:]), dtype=np.uint8)
            frames[:, :-1, :, :] = self.obs
            frames[:, -1, :, :] = obs_next[:, -1, :, :]
            for entry in zip(frames, action, reward, done):
                replay.append(entry)
            self.obs = obs_next
            self.R += np.array(reward)
            for idx, d in enumerate(done):
                if d:
                    Rs.append(self.R[idx])
                    self.R[idx] = 0
        toc = time.time()
        return replay, Rs, Qs, self.rank, len(replay) / (toc - tic)

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
        self.atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_atom = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.model = NatureCNN(self.state_shape[0], self.action_dim, self.dueling).to(self.device)
        self.model_target = NatureCNN(self.state_shape[0], self.action_dim, self.dueling).to(self.device)

        # self.optimizer = torch.optim.AdamW(self.model.parameters(), self.adam_lr, eps=self.adam_eps)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.adam_lr) # , eps=self.adam_eps)
        self.update_steps = 0
        self.replay = deque(maxlen=self.replay_size)

    def get_datafetcher(self):
        dataset = ReplayDataset(self.replay)
        self.dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_data_workers, pin_memory=self.pin_memory)
        datafetcher = DataPrefetcher(self.dataloader, self.device)
        return datafetcher

    def train_step(self):
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

        with torch.no_grad():
            prob_next, _ = self.model_target(next_states)
            if self.double_q:
                prob_next_online, _ = self.model(next_states)
                actions_next = prob_next_online.mul(self.atoms).sum(dim=-1).argmax(dim=-1)
            else:
                actions_next = prob_next.mul(self.atoms).sum(dim=-1).argmax(dim=-1)
            prob_next = prob_next[self.batch_indices, actions_next, :]

            rewards = rewards.float().unsqueeze(-1)
            terminals = terminals.float().unsqueeze(-1)
            atoms_next = rewards + self.discount * (1 - terminals) * self.atoms.view(1, -1)

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

        _, log_prob = self.model(states)
        log_prob = log_prob[self.batch_indices, actions, :]
        loss = target_prob.mul_(log_prob).sum(dim=-1).neg().mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1

        if self.update_steps % self.target_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
        return loss.detach()

    def train_step_(self):
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

        with torch.no_grad():

            if self.double_q:
                q_next = self.model_target(next_states)
                q_next_online = self.model(next_states)
                q_next = q_next.gather(1, q_next_online.argmax(dim=-1, keepdim=True)).squeeze(-1)
            else:
                q_next, _ = self.model_target(next_states).max(dim=-1)

            q_target = rewards + self.discount * (1 - terminals) * q_next

        q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1

        if self.update_steps % self.target_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
        return loss.detach()

    def adjust_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class Trainer(tune.Trainable):
    def _setup(self, config):
        kwargs = default_hyperparams()
        for k, v in config.items():
            kwargs[k] = v

        for k, v in kwargs.items():
            setattr(self, k, v)

        print("input args:\n", json.dumps(kwargs, indent=4, separators=(",", ":")))

        self.agent = Agent(**kwargs)
        self.epsilon_schedule = LinearSchedule(1.0, 0.01, int(self.total_steps * self.exploration_ratio))
        self.epsilon = 1.0
        self.actors = [Actor.remote(rank=rank, **kwargs) for rank in range(self.num_actors + 1)]
        self.tester = self.actors[-1]

        self.steps_per_epoch = self.total_steps // self.epoches
        self.actor_steps = self.total_steps // (self.epoches * self.num_envs * self.num_actors)

        self.sample_ops = [a.sample.remote(self.actor_steps, 1.0, self.agent.model.state_dict()) for a in
                           self.actors[:-1]]
        self.frame_count = 0
        self.lr_updated = False
        self.Rs, self.Qs, self.TRs, self.Ls = [], [], [], []

    def _train(self):
        done_id, self.sample_ops = ray.wait(self.sample_ops)
        data = ray.get(done_id)
        local_replay, Rs, Qs, rank, fps = data[0]
        if rank < self.num_actors:
            # Actors
            self.agent.replay.extend(local_replay)
            self.epsilon = self.epsilon_schedule(len(local_replay))

            # if self.epsilon == 0.01:
            #    self.epsilon = np.random.choice([0.01, 0.02, 0.05, 0.1], p=[0.7, 0.1, 0.1, 0.1])

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
            self.sample_ops.append(self.tester.sample.remote(self.actor_steps, 0.01, self.agent.model.state_dict()))
            self.TRs += Rs

        # Start testing at itr > 100
        if self.iteration == 100:
            print("Testing Started ... ")
            self.sample_ops.append(self.tester.sample.remote(self.actor_steps, 0.01, self.agent.model.state_dict()))

        if self.frame_count > (self.total_steps * self.exploration_ratio) and not self.lr_updated:
            # self.agent.adjust_lr(self.adam_lr * 0.1)
            # self.agent_train_freq *= 2
            self.lr_updated = True

        result = dict(
            game=self.game,
            time_past=self._time_total,
            epsilon=self.epsilon,
            adam_lr=self.adam_lr,
            frames=self.frame_count,
            speed=self.frame_count / (self._time_total + 1),
            time_remain=(self.total_steps - self.frame_count) / (self.frame_count / (self._time_total + 1)),
            loss=np.mean(self.Ls[-100:]) if len(self.Ls) > 0 else 0,
            ep_reward_test=np.mean(self.TRs[-100:]) if len(self.TRs) > 0 else 0,
            ep_reward_train=np.mean(self.Rs[-100:]) if len(self.Rs) > 0 else 0,
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
        print("Final Testing")
        if self.frame_count > self.total_steps:
            local_replay, Rs, Qs, rank, fps = ray.get(
                self.tester.sample.remote(self.actor_steps * self.num_envs * self.num_actors, self.epsilon,
                                          self.agent.model.state_dict()))
            print(f"Final Test Result: {np.mean(Rs)}\t{np.std(Rs)}\t{np.max(Rs)}\t{len(Rs)}")
            torch.save({
                'model': self.agent.model.state_dict(),
                'optim': self.agent.optimizer.state_dict(),
                'FTRs': Rs,
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
