from new_config import ExpConfig, ActorEnum
from agent import Agent
from actor import Actor
from replay import ReplayDataset
import torch
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
import os
import numpy as np
import logging
import time

from new_config import ExpConfig, ActorEnum


class Trainer:
    def __init__(self, cfg: ExpConfig) -> None:
        self.cfg = cfg
        self.actor = Actor(cfg)
        self.evaluator = Actor(cfg)
        self.agent = Agent(cfg)
        self.replay = ReplayDataset(cfg.replay.size, cfg.trainer.batch_size)
        self.sampler = RandomSampler(self.replay, replacement=True, num_samples=cfg.replay.size)
        self.loader = DataLoader(self.replay, cfg.trainer.batch_size, sampler=self.sampler,
                                pin_memory=True, pin_memory_device='cuda', drop_last=True)
        self.writer = SummaryWriter('output')
        self.steps = 0


    def step(self):
        tic = time.time()
        transitions, rs, qs, rank, fps, best_ep = self.actor.sample()
        # Actors
        if len(transitions) > 0:
            self.agent.replay.extend(transitions)


        self.epsilon = self.epsilon_schedule(self.cfg.actor_steps * self.cfg.num_envs)
        self.frame_count += self.cfg.actor_steps * self.cfg.num_envs

        self.sample_ops.append(
            self.actors[rank].sample.remote(self.cfg.actor_steps, self.epsilon, self.agent.model.state_dict()))
        self.Rs += rs
        self.Qs += qs
        # Start training at
        if len(self.agent.replay) > self.cfg.start_training_step:
            data = [self.agent.train_step() for _ in range(self.cfg.agent_train_steps)]
            if self.cfg.algo in ['fqf']:
                fraction_loss = torch.stack([x['fraction_loss'] for x in data]).mean().item()
            if self.cfg.best_ep:
                ce_loss = torch.stack([x['ce_loss'] for x in data]).mean().item()

            loss = [x['loss'] for x in data]
            loss = torch.stack(loss)
            self.Ls += loss.tolist()
        toc = time.time()
        self.velocity.append(self.cfg.actor_steps * self.cfg.num_envs / (toc - tic))

        result = dict(
            game=self.cfg.game,
            time_past=self._time_total,
            epsilon=self.epsilon,
            adam_lr=self.cfg.adam_lr,
            frames=self.frame_count,
            fraction_loss=fraction_loss if fraction_loss is not None else 0,
            ce_loss=ce_loss if ce_loss is not None else 0,
            velocity=np.mean(self.velocity[-20:]) if len(self.velocity) > 0 else 0,
            speed=self.frame_count / (self._time_total + 1),
            time_remain=(self.cfg.total_steps - self.frame_count) / ((self.frame_count + 1) / (self._time_total + 1)),
            loss=np.mean(self.Ls[-20:]) if len(self.Ls) > 0 else 0,
            ep_reward_test=np.mean(self.ITRs) if len(self.ITRs) > 0 else 0,
            ep_reward_train=np.mean(self.Rs[-20:]) if len(self.Rs) > 0 else 0,
            ep_reward_train_max=np.max(self.Rs) if len(self.Rs) > 0 else 0,
            ep_reward_test_max=np.max(self.TRs) if len(self.TRs) > 0 else 0,
            qmax=np.mean(self.Qs[-100:]) if len(self.Qs) > 0 else 0
        )
        return result



