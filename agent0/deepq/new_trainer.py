import time

import numpy as np
from agent0.common.utils import LinearSchedule, set_random_seed
from agent0.deepq.new_agent import Learner, Actor
from agent0.deepq.new_config import ExpConfig

from tensorboardX import SummaryWriter
from tqdm import tqdm

class Trainer:
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        print(cfg)

        set_random_seed(cfg.seed)
        self.learner = Learner(cfg)
        self.actor = Actor(cfg)
        self.epsilon_schedule = LinearSchedule(1.0, cfg.actor.min_eps, cfg.trainer.exploration_steps)

        self.frame_count = 0
        self.epsilon = 1.0
        self.writer = SummaryWriter('output2')
        self.num_transitions = self.cfg.actor.actor_steps * self.cfg.actor.num_envs
        self.Ls, self.Rs, self.Qs = [], [], []

    def step(self):
        tic = time.time()
        transitions, returns, qmax = self.actor.sample(self.cfg.actor.actor_steps, self.epsilon, self.learner.model)
        self.Qs.extend(qmax)
        self.Rs.extend(returns)
        # Actors
        self.learner.replay.extend(transitions)

        self.epsilon = self.epsilon_schedule(self.num_transitions)
        self.frame_count += self.num_transitions

        # Start training at
        if len(self.learner.replay) > self.cfg.trainer.training_start_steps:
            data = [self.learner.train_step() for _ in range(self.cfg.trainer.learner_steps)]
            loss = [x['loss'] for x in data]
            self.Ls.extend(loss)

        toc = time.time()

        result = dict(
            epsilon=self.epsilon,
            frames=self.frame_count,
            velocity=self.num_transitions / (toc - tic),
            loss=np.mean(self.Ls[-20:]) if len(self.Ls) > 0 else None,
            return_train=np.mean(self.Rs[-20:]) if len(self.Rs) > 0 else None,
            return_train_max=np.max(self.Rs) if len(self.Rs) > 0 else None,
            qmax=np.mean(self.Qs[-100:]) if len(self.Qs) > 0 else None
        )
        return result

    def run(self):
        trainer_steps = self.cfg.trainer.total_steps // self.num_transitions + 1
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