import json
import time
from abc import ABC

import numpy as np
import torch
from agent0.common.utils import LinearSchedule, set_random_seed
from agent0.deepq.actor import Actor
from agent0.deepq.agent import Agent
from agent0.deepq.config import Config

from tensorboardX import SummaryWriter
import logging
from tqdm import tqdm

class Trainer:
    def __init__(self):

        self.Rs, self.Qs, self.TRs, self.Ls, self.ITRs, self.velocity = [], [], [], [], [], []
        self.cfg = None
        self.agent = None
        self.epsilon = None
        self.epsilon_schedule = None
        self.actors = None
        self.frame_count = None
        self.Rs, self.Qs, self.TRs, self.Ls, self.ITRs = [], [], [], [], []
        self.best = float('-inf')
        self.sample_ops = None

    def setup(self, config):
        self.cfg = Config(**config)
        self.cfg.update_atoms()
        set_random_seed(self.cfg.random_seed)
        print("input args:\n", json.dumps(vars(self.cfg), indent=4, separators=(",", ":")))

        self.agent = Agent(**config)
        self.epsilon_schedule = LinearSchedule(1.0, self.cfg.min_eps, self.cfg.exploration_steps)
        self.actor = Actor(rank=0, **config)

        self.frame_count = 0
        self.best = float('-inf')
        self.epsilon = 1.0
        self.writer = SummaryWriter('output')

    def step(self):
        tic = time.time()

        transitions, rs, qs, rank, fps, best_ep = self.actor.sample(self.cfg.actor_steps, self.epsilon, self.agent.model)
        # Actors
        if len(transitions) > 0:
            self.agent.replay.extend(transitions)
        if len(best_ep) > 0:
            self.agent.replay.extend_ep_best(best_ep)

        self.epsilon = self.epsilon_schedule(self.cfg.actor_steps * self.cfg.num_envs)
        self.frame_count += self.cfg.actor_steps * self.cfg.num_envs

        self.Rs += rs
        self.Qs += qs
        # Start training at
        if len(self.agent.replay) > self.cfg.start_training_step:
            data = [self.agent.train_step() for _ in range(self.cfg.agent_train_steps)]
            loss = [x['loss'] for x in data]
            loss = torch.stack(loss)
            self.Ls += loss.tolist()
        toc = time.time()
        self.velocity.append(self.cfg.actor_steps * self.cfg.num_envs / (toc - tic))

        result = dict(
            epsilon=self.epsilon,
            frames=self.frame_count,
            velocity=np.mean(self.velocity[-20:]) if len(self.velocity) > 0 else None,
            loss=np.mean(self.Ls[-20:]) if len(self.Ls) > 0 else None,
            return_test=np.mean(self.ITRs) if len(self.ITRs) > 0 else None,
            return_train=np.mean(self.Rs[-20:]) if len(self.Rs) > 0 else None,
            return_train_max=np.max(self.Rs) if len(self.Rs) > 0 else None,
            return_test_max=np.max(self.TRs) if len(self.TRs) > 0 else None,
            qmax=np.mean(self.Qs[-100:]) if len(self.Qs) > 0 else None
        )
        return result

    def run(self):
        trainer_steps = self.cfg.total_steps // (self.cfg.num_envs * self.cfg.actor_steps) + 1
        with tqdm(range(trainer_steps)) as t:
            for _ in t:
                result = self.step()
                msg = ""
                for k, v in result.items():
                    if v is None:
                        continue
                    self.writer.add_scalar(k, v, self.frame_count)
                    if k in ['frames', 'loss', 'qmax'] or 'return' in k:
                        msg += f"{k}: {v:.2f} | "
                t.set_description(msg)