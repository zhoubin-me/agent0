# from tensorboardX import SummaryWriter
import os
import time
from concurrent import futures
from time import localtime, strftime

import git
import hydra
import launchpad as lp
import numpy as np
import shortuuid
from absl import logging
from dacite import from_dict
from einops import repeat
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import agent0.deepq.agent as agents
import wandb
from agent0.common.atari_wrappers import make_atari
from agent0.deepq.config import ExpConfig
from agent0.deepq.trainer import Trainer


class TrainerNode(Trainer):
    def __init__(self, cfg: ExpConfig, actors):
        super().__init__(cfg, use_lp=True)
        self.actors = actors

    def run(self):
        trainer_steps = self.cfg.trainer.total_steps // self.num_transitions + 1
        sample_eps = self.epsilon_fn(self.frame_count)
        tasks = [
            actor.futures.sample(sample_eps, self.learner.model.state_dict())
            for actor in self.actors[1:]
        ]

        tasks.append(
            self.actors[0].futures.test(
                self.frame_count, self.learner.model.state_dict()
            )
        )

        step = 0
        while step < trainer_steps:
            tic = time.time()
            dones, not_dones = futures.wait(tasks, return_when=futures.FIRST_COMPLETED)
            tasks = list(dones) + list(not_dones)
            rank, (transitions_or_video, returns, qmax_or_frames) = tasks.pop(
                0
            ).result()
            if rank > 0:
                qmax = qmax_or_frames
                transitions = transitions_or_video
                sample_eps = self.epsilon_fn(self.frame_count)
                tasks.append(
                    self.actors[rank].futures.sample(
                        sample_eps, self.learner.model.state_dict()
                    )
                )
                result = self.step(transitions, returns, qmax)
                step += 1
            else:
                tasks.append(
                    self.actors[rank].futures.test(
                        self.frame_count, self.learner.model.state_dict()
                    )
                )

                test_frames = qmax_or_frames
                video = transitions_or_video
                video = np.stack(video, axis=1)
                video = repeat(video, "n t c h w -> n t (3 c) h w")
                self.RTs.extend(returns)

                if self.cfg.tb:
                    self.writer.add_scalar("return_test", np.mean(returns), test_frames)
                    self.writer.add_scalar(
                        "return_test_max", np.max(self.RTs), test_frames
                    )
                    self.writer.add_video("test_video", video, test_frames, fps=60)
                if self.cfg.wandb:
                    wandb.log({"return_test": np.mean(returns), "frame": test_frames})
                    wandb.log(
                        {"return_test_max": np.max(self.RTs), "frame": test_frames}
                    )
                    wandb.log({"test_video": wandb.Video(video, fps=60, format="mp4"), "frame": test_frames})

                continue

            if rank > 0 and self.frame_count > self.cfg.trainer.training_start_steps:
                fps = self.num_transitions / (time.time() - tic)
                result.update(fps=fps)
            self.logging(result)

        self.final()

    def final(self):
        self.logger.info("Final Testing ... ")
        dones = futures.wait(
            [
                actor.futures.test(self.frame_count, self.learner.model.state_dict())
                for actor in self.actors
            ],
            return_when=futures.ALL_COMPLETED,
        )
        test_returns = []
        for done in dones:
            _, (_, returns, _) = done.result()
            test_returns.extend(returns)

        self.logger.info(
            f"TEST ---> Frames: {self.frame_count} | Return Avg: {np.mean(test_returns):.2f} Max: {np.max(test_returns)}"
        )
        self.writer.add_scalar("return_test", np.mean(test_returns), self.frame_count)
        self.writer.add_scalar("return_test_max", np.max(self.RTs), self.frame_count)
        futures.wait(
            [actor.close() for actor in self.actors], return_when=futures.ALL_COMPLETED
        )


class ActorNode:
    def __init__(self, rank: int, cfg: ExpConfig) -> None:
        self.cfg = cfg
        self.rank = rank
        self.step_count = 0
        self.actor = agents.Actor(cfg)

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

    def test(self, frame_count, model_dict=None):
        rs = []
        tic = time.time()
        frames = 0
        video = []
        while len(rs) < self.cfg.trainer.test_episodes:
            images, returns, _ = self.actor.sample(
                self.cfg.actor.test_eps, model_dict, test=True
            )
            frames += sum([x.shape[0] for x in images])
            rs.extend(returns)
            if len(video) < 3600:
                video.extend(images)

        fps = frames / (time.time() - tic)
        logging.info(
            f"Rank {self.rank} -- Test Frames: {frame_count} FPS: {fps:.2f} | Avg Return {np.mean(rs):.2f}"
        )
        return self.rank, (video, rs, frame_count)

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
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:8]
    uuid = shortuuid.uuid()[:4]
    timestr = strftime("%Y%m%d-%H%M%S", localtime())
    subdir = (
        f"{cfg.name}-{cfg.env_id}-{cfg.learner.algo.name}-{cfg.seed}-{sha}-{timestr}-{uuid}"
    )
    dummy_env = make_atari(cfg.env_id, num_envs=1)
    dummy_env.close()

    cfg.logdir = os.path.join(cfg.logdir, subdir)
    cfg.obs_shape = dummy_env.observation_space.shape[1:]
    cfg.action_dim = int(dummy_env.action_space[0].n)

    cfg = OmegaConf.to_container(cfg)
    cfg = from_dict(ExpConfig, cfg)

    program = make_program(cfg)
    lp.launch(program, launch_type="local_mp", terminal="tmux_session")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=ExpConfig)
    main()
