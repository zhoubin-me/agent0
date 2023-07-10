# from tensorboardX import SummaryWriter
import time
from typing import List

import hydra
import launchpad as lp
import numpy as np
from absl import logging
from dacite import from_dict
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import tqdm

from agent0.common.utils import LinearSchedule, set_random_seed
from agent0.deepq.new_agent import Actor, Learner
from agent0.deepq.new_config import ActorConfig, ExpConfig


class TrainerNode:
    def __init__(self, actors, cfg: ExpConfig) -> None:
        self.cfg = cfg
        set_random_seed(cfg.seed)

        self._actors = actors
        self.learner = Learner(cfg)
        self.epsilon_schedule = LinearSchedule(
            1.0, cfg.actor.min_eps, cfg.trainer.exploration_steps
        )

        self.frame_count = 0
        self.epsilon = 1.0
        # self.writer = SummaryWriter('tblog')
        self.num_transitions = self.cfg.actor.actor_steps * self.cfg.actor.num_envs
        self.Ls, self.Rs, self.Qs = [], [], []

    def run(self):
        trainer_steps = self.cfg.trainer.total_steps // self.num_transitions + 1
        with tqdm(range(trainer_steps)) as t:
            for _ in t:
                result = self.step()
                msg = ""
                for k, v in result.items():
                    if v is None:
                        continue
                    # self.writer.add_scalar(k, v, self.frame_count)
                    if k in ["frames", "loss", "qmax"] or "return" in k:
                        msg += f"{k}: {v:.2f} | "
                t.set_description(msg)

        for actor in self._actors:
            actor.close()

        lp.stop()

    def step(self):
        tic = time.time()
        futures = [actor.futures.sample(self.epsilon) for actor in self._actors]
        results = [future.result() for future in futures]
        transitions, returns, qmax = results[0]
        self.Qs.extend(qmax)
        self.Rs.extend(returns)
        # Actors
        self.learner.replay.extend(transitions)

        self.epsilon = self.epsilon_schedule(self.num_transitions)
        self.frame_count += self.num_transitions

        # Start training at
        if len(self.learner.replay) > self.cfg.trainer.training_start_steps:
            data = [
                self.learner.train_step() for _ in range(self.cfg.trainer.learner_steps)
            ]
            loss = [x["loss"] for x in data]
            self.Ls.extend(loss)

        toc = time.time()

        result = dict(
            epsilon=self.epsilon,
            frames=self.frame_count,
            velocity=self.num_transitions / (toc - tic),
            loss=np.mean(self.Ls[-20:]) if len(self.Ls) > 0 else None,
            return_train=np.mean(self.Rs[-20:]) if len(self.Rs) > 0 else None,
            return_train_max=np.max(self.Rs) if len(self.Rs) > 0 else None,
            qmax=np.mean(self.Qs[-100:]) if len(self.Qs) > 0 else None,
        )
        return result


class ActorNode:
    def __init__(self, cfg: ExpConfig) -> None:
        self.cfg = cfg
        self.actor = Actor(cfg)

    def sample(self, epsilon):
        print("haha")
        return self.actor.sample(self.cfg.actor.actor_steps, epsilon)

    def close(self):
        self.actor.close()


def make_program(cfg: ExpConfig):
    program = lp.Program("dqn")
    with program.group("actors"):
        actors = [program.add_node(lp.CourierNode(ActorNode, cfg)) for _ in range(2)]

    node = lp.CourierNode(TrainerNode, actors=actors, cfg=cfg)
    program.add_node(node, label="trainer")

    return program


@hydra.main(version_base=None, config_name="config")
def main(cfg: ExpConfig):
    cfg = OmegaConf.to_container(cfg)
    cfg = from_dict(ExpConfig, cfg)
    program = make_program(cfg)
    resources = {"trainer": {"cpu": 2, "gpu": 1}, "actors": {"cpu": 2}}
    lp.launch(program, local_resources=resources)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=ExpConfig)
    main()
