import launchpad as lp
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent0.deepq.dqn import Learner, Trainer, Config, ActorNode, TrainerNode
from agent0.common.utils import set_random_seed
from agent0.common.atari_wrappers import make_atari

class DoubleConfig(Config):
    test_epsilon = 0.001

class DoubleLearner(Learner):
    def train_step(self, batch):
        obs, actions, rewards, terminals, obs_next = batch
        with torch.no_grad():
            next_q = self.model_target(obs_next)
            _, next_a = self.model(obs).max(dim=-1)
            next_q = next_q.gather(1, next_a.unsqueeze(-1)).squeeze(-1)
            target_q = rewards + self.cfg.discount * (1 - terminals) * next_q
        curr_q = self.model(obs)
        curr_q = curr_q.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        loss = self.loss_fn(curr_q, target_q)
        return loss

class DoubleTrainer(Trainer):
    def __init__(self, cfg: DoubleConfig):
        super(DoubleTrainer, self).__init__(cfg)
        self.learner = DoubleLearner(cfg, self.learner.model)


class DoubleTrainerNode(TrainerNode):
    def __init__(self, cfg: Config, actors):
        super(DoubleTrainerNode, self).__init__(cfg, actors)
        self.learner = DoubleLearner(cfg, self.learner.model)

def make_program(cfg: DoubleConfig):
    program = lp.Program("dqn")
    with program.group("actors"):
        actors = [
            program.add_node(lp.CourierNode(ActorNode, cfg, rank))
            for rank in range(cfg.num_actors)
        ]

    node = lp.CourierNode(DoubleTrainerNode, cfg=cfg, actors=actors)
    program.add_node(node, label="trainer")
    return program


def main(cfg: DoubleConfig):
    set_random_seed(cfg.seed)
    if cfg.use_lp:
        program = make_program(cfg)
        lp.launch(program, launch_type="local_mp", terminal="tmux_session")
    else:
        trainer = DoubleTrainer(cfg)
        trainer.run()


if __name__ == '__main__':
    from wonderwords import RandomWord
    import tyro
    import git

    cfg = tyro.cli(DoubleConfig)
    env = make_atari(cfg.game, 1)
    cfg.obs_shape = env.observation_space.shape[1:]
    cfg.act_dim = env.action_space[0].n
    env.close()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    wordstr = "-".join(RandomWord().random_words(2))
    sha = git.Repo(search_parent_directories=True).head.object.hexsha[:7]
    cfg.expname = f"double-dqn-{cfg.game}-{wordstr}"
    cfg.logdir = f"{cfg.logdir}/double-dqn-{cfg.game}-{timestr}-{sha}-{wordstr}"
    os.makedirs(cfg.logdir, exist_ok=False)

    main(cfg)
