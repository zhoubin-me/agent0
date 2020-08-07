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

        num_actors=8,
        num_envs=16,
        num_data_workers=4,

        adam_lr=2e-4,

        batch_size=512,
        discount=0.99,
        replay_size=int(1e6),
        exploration_ratio=0.15,

        target_update_freq=500,
        agent_train_freq=16,

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
                qs = self.model(st)

            qs_max, qs_argmax = qs.max(dim=-1)
            action_greedy = qs_argmax.tolist()
            Qs.append(qs_max.mean().item())
            action = [act_greedy if p > epsilon else act_random for p, act_random, act_greedy in
                      zip(np.random.rand(self.num_envs), action_random, action_greedy)]

            obs_next, reward, done, info = self.envs.step(action)
            for entry in zip(self.obs, action, reward, obs_next, done):
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
        self.model = NatureCNN(self.state_shape[0], self.action_dim, self.dueling).to(self.device)
        self.model_target = NatureCNN(self.state_shape[0], self.action_dim, self.dueling).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.adam_lr)
        self.update_steps = 0
        self.replay = deque(maxlen=self.replay_size)

    def get_datafetcher(self):
        dataset = ReplayDataset(self.replay)
        dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_data_workers,
                                 pin_memory=True)
        datafetcher = DataPrefetcher(dataloader, self.device)
        return datafetcher

    def train_step(self):
        try:
            data = self.prefetcher.next()
        except:
            self.prefetcher = self.get_datafetcher()
            data = self.prefetcher.next()

        states, actions, rewards, next_states, terminals = data
        states = states.float().div(255.0)
        next_states = next_states.float().div(255.0)
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


class Trainer(tune.Trainable):
    def _setup(self, config):
        kwargs = default_hyperparams()
        for k, v in config.items():
            kwargs[k] = v

        for k, v in kwargs.items():
            setattr(self, k, v)

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
        self.Rs, self.Qs, self.TRs, self.Ls = [], [], [], []

    def _train(self):
        done_id, self.sample_ops = ray.wait(self.sample_ops)
        data = ray.get(done_id)
        local_replay, Rs, Qs, rank, fps = data[0]
        if rank < self.num_actors:
            # Actors
            self.agent.replay.extend(local_replay)
            self.epsilon = self.epsilon_schedule(len(local_replay))
            if self.epsilon == 0.01:
                epsilon = np.random.choice([0.01, 0.02, 0.05, 0.1], p=[0.7, 0.1, 0.1, 0.1])
            self.sample_ops.append(
                self.actors[rank].sample.remote(self.actor_steps, self.epsilon, self.agent.model.state_dict()))
            self.frame_count += len(local_replay)
            result = dict(ep_reward_train=np.mean(Rs)) if len(Rs) > 0 else dict()
            self.Rs += Rs
            self.Qs += Qs
        else:
            # Tester
            self.sample_ops.append(self.tester.sample.remote(self.actor_steps, 0.01, self.agent.model.state_dict()))
            result = dict(ep_reward_test=np.mean(Rs)) if len(Rs) > 0 else dict()
            self.TRs += Rs

        # Start testing at itr > 100
        if self.iteration == 100:
            print("Testing Started ... ")
            self.sample_ops.append(self.tester.sample.remote(self.actor_steps, 0.01, self.agent.model.state_dict()))

        # Start training at
        if self.frame_count > self.start_training_step:
            train_tic = time.time()
            loss = [self.agent.train_step() for _ in range(self.agent_train_freq)]
            loss = torch.stack(loss)
            self.Ls += loss.tolist()
            loss = loss.mean().item()
            train_toc = time.time()
            result.update(loss=loss, train_time=train_toc - train_tic)

        result.update(frames=self.frame_count)

        # if self.iteration % 100 == 10:
        #     self.logstat()

        return result

    def _save(self, checkpoint_dir):
        return {
            'model': self.agent.model.state_dict(),
            'optim': self.agent.optimizer.state_dict(),
            'model_target': self.agent.model_target.state_dict(),
            'Ls': self.Ls,
            'Rs': self.Rs,
            'Qs': self.Qs,
            'TRs': self.TRs
        }

    def _restore(self, checkpoint):
        self.agent.model.load_state_dict(checkpoint['model'])
        self.agent.model_target.load_state_dict(checkpoint['model_target'])
        self.agent.optimizer.load_state_dict(checkpoint['optim'])
        self.Ls = checkpoint['Ls']
        self.Qs = checkpoint['Qs']
        self.Rs = checkpoint['Rs']
        self.TRs = checkpoint['TRs']

    def _export_model(self, export_formats, export_dir):
        if export_formats == [ExportFormat.MODEL]:
            path = os.path.join(export_dir, "exported_models")
            torch.save({
                "model": self.agent.model.state_dict(),
            }, path)
            return {ExportFormat.MODEL: path}
        else:
            raise ValueError("unexpected formats: " + str(export_formats))

    def _stop(self):
        print("Final Testing")
        local_replay, Rs, Qs, rank, fps  = ray.get(self.tester.sample.remote(self.actor_steps * self.num_envs * self.num_actors, self.epsilon, self.agent.model.state_dict()))
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
        ray.get([a.close_envs.remote() for a in self.actors])


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
