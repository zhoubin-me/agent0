# from tensorboardX import SummaryWriter
import time
from concurrent import futures

import hydra
import launchpad as lp
import numpy as np
from absl import logging
from dacite import from_dict
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from agent0.common.atari_wrappers import make_atari
import agent0.deepq.new_agent as agents
from agent0.deepq.new_config import ExpConfig
from agent0.deepq.new_trainer import Trainer


class TrainerNode(Trainer):
    def __init__(self, cfg: ExpConfig, actors):
        super().__init__(cfg, use_lp=True)
        self.actors = actors

    def run(self):
        trainer_steps = self.cfg.trainer.total_steps // self.num_transitions + 1
        sample_eps = self.epsilon_fn(self.frame_count)
        tasks = [
            actor.futures.sample(sample_eps, self.learner.model.state_dict())
            for actor in self.actors
        ]

        for step in range(trainer_steps):
            dones, not_dones = futures.wait(tasks, return_when=futures.FIRST_COMPLETED)
            tasks = list(dones) + list(not_dones)
            rank, (transitions, returns, qmax) = tasks.pop(0).result()
            result = self.step(transitions, returns, qmax)
            sample_eps = self.epsilon_fn(self.frame_count)
            tasks.append(
                self.actors[rank].futures.sample(sample_eps, self.learner.model.state_dict())
            )

            msg = ""
            for k, v in result.items():
                if v is None:
                    continue
                self.writer.add_scalar(k, v, self.frame_count)
                if k in ["frames", "loss", "qmax"] or "return" in k:
                    msg += f"{k}: {v:.2f} | "
            if step % self.cfg.trainer.log_freq == 0:
                logging.info(msg)

        futures.wait([actor.close() for actor in self.actors], return_when=futures.ALL_COMPLETED)
        
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
