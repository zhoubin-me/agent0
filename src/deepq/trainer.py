import json
import os
from abc import ABC

import numpy as np
import ray
import torch
from ray import tune
from ray.tune.trial import ExportFormat

from src.common.utils import LinearSchedule, set_random_seed
from src.deepq.actor import Actor
from src.deepq.agent import Agent
from src.deepq.config import Config


class Trainer(tune.Trainable, ABC):
    def __init__(self, config=None, logger_creator=None):

        self.cfg = None
        self.agent = None
        self.epsilon = None
        self.epsilon_schedule = None
        self.actors = None
        self.frame_count = None
        self.Rs, self.Qs, self.TRs, self.Ls, self.ITRs = [], [], [], [], []
        self.best = float('-inf')
        self.sample_ops = None
        super(Trainer, self).__init__(config, logger_creator)

    def setup(self, config):
        self.cfg = Config(**config)
        set_random_seed(self.cfg.random_seed)
        print("input args:\n", json.dumps(vars(self.cfg), indent=4, separators=(",", ":")))

        self.agent = Agent(**config)
        self.epsilon_schedule = LinearSchedule(1.0, self.cfg.min_eps,
                                               int(self.cfg.total_steps * self.cfg.exploration_ratio))
        self.actors = [Actor.remote(rank=rank, **config) for rank in range(self.cfg.num_actors)]

        self.frame_count = 0
        self.Rs, self.Qs, self.TRs, self.Ls, self.ITRs = [], [], [], [], []
        self.best = float('-inf')
        self.epsilon = 1.0

        self.sample_ops = [a.sample.remote(self.cfg.actor_steps, 1.0, self.agent.model.state_dict()) for a in
                           self.actors]

    def step(self):
        done_id, self.sample_ops = ray.wait(self.sample_ops)
        data = ray.get(done_id)
        transitions, rs, qs, rank, fps = data[0]
        # Actors
        self.agent.replay.extend(transitions)
        self.epsilon = self.epsilon_schedule(self.cfg.actor_steps * self.cfg.num_envs)
        self.frame_count += self.cfg.actor_steps * self.cfg.num_envs

        self.sample_ops.append(
            self.actors[rank].sample.remote(self.cfg.actor_steps, self.epsilon, self.agent.model.state_dict()))
        self.Rs += rs
        self.Qs += qs
        # Start training at
        if self.frame_count > self.cfg.start_training_step:
            loss = [self.agent.train_step() for _ in range(self.cfg.agent_train_steps)]
            loss = torch.stack(loss)
            self.Ls += loss.tolist()

        result = dict(
            game=self.cfg.game,
            time_past=self._time_total,
            epsilon=self.epsilon,
            adam_lr=self.cfg.adam_lr,
            frames=self.frame_count,
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

    def save_checkpoint(self, checkpoint_dir):
        output = ray.get([a.sample.remote(self.cfg.actor_steps,
                                          self.cfg.min_eps,
                                          self.agent.model.state_dict(),
                                          testing=True,
                                          test_episodes=10) for a in self.actors])

        ckpt_rs = []
        for _, rs, qs, rank, fps in output:
            ckpt_rs += rs

        self.ITRs = ckpt_rs
        self.TRs += ckpt_rs
        print(f"Iteration {self.training_iteration} test Result(mean|std|max|min|len):"
              f" {np.mean(ckpt_rs)}\t{np.std(ckpt_rs)}\t{np.max(ckpt_rs)}\t{np.min(ckpt_rs)}\t{len(ckpt_rs)}")

        data_to_save = {
            'model': self.agent.model.state_dict(),
            'optim': self.agent.optimizer.state_dict(),
            'model_target': self.agent.model_target.state_dict(),
            'Ls': self.Ls,
            'Rs': self.Rs,
            'Qs': self.Qs,
            'TRs': self.TRs,
            'frame_count': self.frame_count,
            'ITRs': ckpt_rs,
            'best': self.best,
        }

        if np.mean(ckpt_rs) > self.best:
            self.best = np.mean(ckpt_rs)
            torch.save(data_to_save, './best.pth')

        return data_to_save

    def load_checkpoint(self, checkpoint):
        self.agent.model.load_state_dict(checkpoint['model'])
        self.agent.model_target.load_state_dict(checkpoint['model_target'])
        self.agent.optimizer.load_state_dict(checkpoint['optim'])
        self.Ls = checkpoint['Ls']
        self.Qs = checkpoint['Qs']
        self.Rs = checkpoint['Rs']
        self.TRs = checkpoint['TRs']
        self.frame_count = checkpoint['frame_count']
        self.best = checkpoint['best']
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
            self.cfg.adam_lr = new_config['adam_lr']
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = new_config['adam_lr']

        self.config = new_config
        return True

    def cleanup(self):
        ray.get([a.close_envs.remote() for a in self.actors])
