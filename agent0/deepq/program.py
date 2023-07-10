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
import torch

from agent0.deepq.new_agent import Actor, Learner
from agent0.deepq.new_model import DeepQNet
from agent0.deepq.new_config import ExpConfig
from agent0.deepq.replay import ReplayDataset
from agent0.common.atari_wrappers import make_atari
from agent0.common.utils import set_random_seed, DataLoaderX, DataPrefetcher

class TrainerNode:
    def __init__(self, cfg: ExpConfig, actors):
        self.cfg = cfg
        self.actors = actors

        set_random_seed(cfg.seed)

        dummy_env = make_atari(cfg.env_id, 1)
        self.obs_shape = dummy_env.observation_space.shape[1:]
        self.act_dim = dummy_env.action_space[0].n
        dummy_env.close()

        self.model = DeepQNet(self.act_dim, self.obs_shape[0]).to(cfg.device.value)
        self.replay = ReplayDataset(cfg)
        self.learner = Learner(cfg)


        self.epsilon_fn = (
            lambda step: cfg.actor.min_eps
            if step > cfg.trainer.exploration_steps
            else (1.0 - step / cfg.trainer.exploration_steps) + cfg.actor.min_eps
        )

        self.frame_count = 0
        self.num_transitions = cfg.actor.actor_steps * cfg.actor.num_envs
        self.Ls, self.Rs, self.Qs = [], [], []
        self.data_fetcher = None

    def get_data_fetcher(self):
        data_loader = DataLoaderX(
            self.replay,
            batch_size=self.cfg.learner.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        data_fetcher = DataPrefetcher(data_loader, self.cfg.device.value)
        return data_fetcher

    def step(self):
        tic = time.time()
        epsilon = self.epsilon_fn(self.frame_count)
        futures = [actor.futures.sample(epsilon) for actor in self.actors]
        results = [future.result() for future in futures]
        transitions, returns, qmax = results[0]
        # transitions, returns, qmax = self.actor.sample(epsilon)
        self.Qs.extend(qmax)
        self.Rs.extend(returns)
        # Actors
        self.replay.extend(transitions)
        self.frame_count += self.num_transitions

        # Start training at
        if len(self.replay) > self.cfg.trainer.training_start_steps:
            for _ in range(self.cfg.learner.learner_steps):
                try:
                    data = self.data_fetcher.next()
                except (StopIteration, AttributeError):
                    self.data_fetcher = self.get_data_fetcher()
                    data = self.data_fetcher.next()
                frames, actions, rewards, terminals = map(lambda x: x.float(), data)
                frames = frames.reshape(
                    self.cfg.learner.batch_size, -1, *self.obs_shape[1:]
                ).div(255.0)
                obs, next_obs = torch.split(frames, self.obs_shape[0], 1)
                actions = actions.long()
                loss = self.learner.train_step(
                    obs, actions, rewards, terminals, next_obs
                )
                self.Ls.append(loss["loss"])

        toc = time.time()

        result = dict(
            epsilon=epsilon,
            frames=self.frame_count,
            velocity=self.num_transitions / (toc - tic),
            loss=np.mean(self.Ls[-20:]) if len(self.Ls) > 0 else None,
            return_train=np.mean(self.Rs[-20:]) if len(self.Rs) > 0 else None,
            return_train_max=np.max(self.Rs) if len(self.Rs) > 0 else None,
            qmax=np.mean(self.Qs[-100:]) if len(self.Qs) > 0 else None,
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
                    if k in ["frames", "loss", "qmax"] or "return" in k:
                        msg += f"{k}: {v:.2f} | "
                t.set_description(msg)

        self.actor.close()


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

    node = lp.CourierNode(TrainerNode,cfg=cfg,  actors=actors)
    program.add_node(node, label="trainer")

    return program


@hydra.main(version_base=None, config_name="config")
def main(cfg: ExpConfig):
    cfg = OmegaConf.to_container(cfg)
    cfg = from_dict(ExpConfig, cfg)
    program = make_program(cfg)
    resources = {"trainer": {"cpu": 2, "gpu": 1}, "actors": {"cpu": 2}}
    lp.launch(program)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=ExpConfig)
    main()
