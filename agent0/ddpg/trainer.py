import json
import os
import time
from abc import ABC

import numpy as np
import torch
from agent0.common.utils import set_random_seed
from agent0.ddpg.agent import Agent
from agent0.ddpg.config import Config
from ray import tune
from ray.tune.trial import ExportFormat


class Trainer(tune.Trainable, ABC):
    def __init__(self, config=None, logger_creator=None):
        self.Rs, self.Qs, self.TRs, self.Ls, self.ITRs, self.velocity = [], [], [], [], [], []
        self.cfg = None
        self.agent = None
        self.epsilon = None
        self.epsilon_schedule = None
        self.actors = None
        self.frame_count = None
        self.Rs, self.Qs, self.TRs, self.VLoss, self.PLoss, self.ITRs = [], [], [], [], [], []
        self.best = float('-inf')
        self.sample_ops = None
        super(Trainer, self).__init__(config, logger_creator)

    def setup(self, config):

        self.cfg = Config(**config)
        set_random_seed(self.cfg.seed)
        print("input args:\n", json.dumps(vars(self.cfg), indent=4, separators=(",", ":")))

        self.agent = Agent(**config)
        self.frame_count = 0
        self.best = float('-inf')

    def step(self):
        tic = time.time()
        rs, vloss, ploss = self.agent.step()

        if rs is not None:
            self.Rs.append(rs)
        if ploss is not None:
            self.PLoss.append(ploss)
        if vloss is not None:
            self.VLoss.append(vloss)

        toc = time.time()
        self.velocity.append(self.cfg.actor_steps * self.cfg.num_envs / (toc - tic))

        result = dict(
            game=self.cfg.game,
            time_past=self._time_total,
            frames=self._iteration,
            velocity=np.mean(self.velocity[-20:]) if len(self.velocity) > 0 else 0,
            speed=self.frame_count / (self._time_total + 1),
            time_remain=(self.cfg.total_steps - self.frame_count) / ((self.frame_count + 1) / (self._time_total + 1)),
            v_loss=np.mean(self.VLoss[-20:]) if len(self.VLoss) > 0 else 0,
            p_loss=np.mean(self.PLoss[-20:]) if len(self.PLoss) > 0 else 0,
            ep_reward_test=np.mean(self.ITRs) if len(self.ITRs) > 0 else 0,
            ep_reward_train=np.mean(self.Rs[-20:]) if len(self.Rs) > 0 else 0,
            ep_reward_train_max=np.max(self.Rs) if len(self.Rs) > 0 else 0,
            ep_reward_test_max=np.max(self.TRs) if len(self.TRs) > 0 else 0,
        )
        return result

    def save_checkpoint(self, checkpoint_dir):
        rs = []
        while True:
            rs += self.agent.sample(testing=True)
            if len(rs) > self.cfg.test_episodes:
                break

        self.ITRs = rs
        self.TRs += rs
        print(f"Iteration {self.training_iteration} test Result(mean|std|max|min|len):"
              f" {np.mean(rs)}\t{np.std(rs)}\t{np.max(rs)}\t{np.min(rs)}\t{len(rs)}")

        data_to_save = {
            'model': self.agent.network.state_dict(),
            'model_target': self.agent.target_network.state_dict(),
            'actor_optim': self.agent.actor_optimizer.state_dict(),
            'critic_optim': self.agent.critic_optimizer.state_dict(),
            'VLoss': self.VLoss,
            'PLoss': self.PLoss,
            'Rs': self.Rs,
            'TRs': self.TRs,
            'frame_count': self.frame_count,
            'ITRs': rs,
            'best': self.best,
        }

        if np.mean(rs) > self.best:
            self.best = np.mean(rs)
            torch.save(data_to_save, './best.pth')

        return data_to_save

    def load_checkpoint(self, checkpoint):
        self.agent.network.load_state_dict(checkpoint['model'])
        self.agent.target_network.load_state_dict(checkpoint['model_target'])
        self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optim'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optim'])
        self.VLoss = checkpoint['VLoss']
        self.PLoss = checkpoint['PLoss']
        self.Rs = checkpoint['Rs']
        self.TRs = checkpoint['TRs']
        self.frame_count = checkpoint['frame_count']
        self.best = checkpoint['best']

    def _export_model(self, export_formats, export_dir):
        if export_formats == [ExportFormat.MODEL]:
            path = os.path.join(export_dir, "exported_models")
            torch.save({
                "model": self.agent.network.state_dict(),
            }, path)
            return {ExportFormat.MODEL: path}
        else:
            raise ValueError("unexpected formats: " + str(export_formats))
