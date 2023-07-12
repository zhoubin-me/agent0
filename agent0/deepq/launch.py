# from tensorboardX import SummaryWriter
import time
from concurrent import futures
from typing import List

import hydra
import launchpad as lp
import numpy as np
import torch
from absl import logging
from dacite import from_dict
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from tqdm import tqdm

from agent0.common.atari_wrappers import make_atari
from agent0.common.utils import DataLoaderX, DataPrefetcher, set_random_seed
from agent0.deepq.new_agent import Actor, Learner
from agent0.deepq.new_config import ExpConfig
from agent0.deepq.replay import ReplayDataset


class TrainerNode:
    def __init__(self, cfg: ExpConfig, actors):
        self.cfg = cfg
        self.actors = actors

        set_random_seed(cfg.seed)
        self.replay = ReplayDataset(cfg)
        self.learner = Learner(cfg)
        self.model = self.learner.model

        self.epsilon_fn = (
            lambda step: cfg.actor.min_eps
            if step > cfg.trainer.exploration_steps
            else (1.0 - step / cfg.trainer.exploration_steps) + cfg.actor.min_eps
        )

        self.frame_count = 0
        self.num_transitions = cfg.actor.actor_steps * cfg.actor.num_envs
        self.Ls, self.Rs, self.Qs, self.RTs = [], [], [], []
        self.data_fetcher = None
        self.writer = SummaryWriter(cfg.logdir)

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

    def step(self, transitions, returns, qmax):
        self.Qs.extend(qmax)
        self.Rs.extend(returns)
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
                    self.cfg.learner.batch_size, -1, *self.cfg.obs_shape[1:]
                ).div(255.0)
                obs, next_obs = torch.split(frames, self.cfg.obs_shape[0], 1)
                actions = actions.long()
                loss = self.learner.train_step(
                    obs, actions, rewards, terminals, next_obs
                )
                self.Ls.append(loss["loss"])

        result = dict(
            frames=self.frame_count,
            loss=np.mean(self.Ls[-20:]) if len(self.Ls) > 0 else None,
            return_train=np.mean(self.Rs[-20:]) if len(self.Rs) > 0 else None,
            return_train_max=np.max(self.Rs) if len(self.Rs) > 0 else None,
            qmax=np.mean(self.Qs[-100:]) if len(self.Qs) > 0 else None,
        )
        return result

    def run(self):
        sample_eps = self.epsilon_fn(self.frame_count)
        tasks = [
            actor.futures.sample(sample_eps, self.model.state_dict())
            for actor in self.actors
        ]
        num_steps = self.cfg.trainer.total_steps // self.num_transitions + 1
        training_start_step = (
            self.cfg.trainer.training_start_steps // self.num_transitions + 1
        )
        start_time = None
        
        for step in range(num_steps):
            if step == training_start_step:
                start_time = time.time()
                start_frames = self.frame_count

            dones, not_dones = futures.wait(tasks, return_when=futures.FIRST_COMPLETED)
            tasks = list(dones) + list(not_dones)
            rank, (transitions, returns, qmax) = tasks.pop(0).result()
            result = self.step(transitions, returns, qmax)

            msg = f"step {step:6d} / {num_steps:6d} | "
            for k, v in result.items():
                if v is None:
                    continue
                self.writer.add_scalar(f"train/{k}", v, self.frame_count)
                if k in ["frames", "loss", "qmax"] or "return" in k:
                    if type(v) in [float, np.float32, np.float64]:
                        msg += f"{k}: {v:7.2f} | "
                    elif type(v) == int:
                        msg += f"{k}: {v:7d} | "

            msg += f"epsilon: {sample_eps:.4f} | "
            self.writer.add_scalar("train/epsilon", sample_eps, self.frame_count)

            if start_time is not None:
                avg_speed = (
                    self.frame_count - start_frames
                ) / (time.time() - start_time)
                msg += f"avg speed: {avg_speed:.2f}"
                self.writer.add_scalar("train/avg_speed", avg_speed, self.frame_count)

            logging.info(msg)

            sample_eps = self.epsilon_fn(self.frame_count)
            tasks.append(
                self.actors[rank].futures.sample(sample_eps, self.model.state_dict())
            )


class ActorNode:
    def __init__(self, rank: int, cfg: ExpConfig) -> None:
        self.cfg = cfg
        self.rank = rank
        self.step_count = 0
        self.actor = Actor(cfg)

    def sample(self, epsilon, model_dict=None):
        tic = time.time()
        transition, returns, qmax = self.actor.sample(epsilon, model_dict)
        toc = time.time()
        fps = len(transition) / (toc - tic)
        self.step_count += 1
        logging.info(
            f"Rank {self.rank} -- Step: {self.step_count:7d} | FPS: {fps:.2f} | Avg Return: {np.mean(returns):.2f}"
        )
        return self.rank, (transition, returns, qmax)

    def close(self):
        self.actor.close()


def make_program(cfg: ExpConfig):
    program = lp.Program("dqn")
    with program.group("actors"):
        actors = [
            program.add_node(lp.CourierNode(ActorNode, rank, cfg))
            for rank in range(cfg.num_actors)
        ]

    node = lp.CourierNode(TrainerNode, cfg=cfg, actors=actors)
    program.add_node(node, label="trainer")
    return program


@hydra.main(version_base=None, config_name="config")
def main(cfg: ExpConfig):
    print(cfg)
    cfg = OmegaConf.to_container(cfg)
    cfg = from_dict(ExpConfig, cfg)
    dummy_env = make_atari(cfg.env_id, num_envs=1)
    cfg.obs_shape = dummy_env.observation_space.shape[1:]
    cfg.action_dim = dummy_env.action_space[0].n
    dummy_env.close()
    program = make_program(cfg)
    lp.launch(program, launch_type="local_mp", terminal="tmux_session")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=ExpConfig)
    main()
