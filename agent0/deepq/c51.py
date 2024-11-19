import launchpad as lp
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from agent0.deepq.dqn import Learner, Trainer, Config, ActorNode, TrainerNode, NatureCNN
from agent0.common.utils import set_random_seed, init
from agent0.common.atari_wrappers import make_atari

class C51Config(Config):
    atoms = 51
    vmax = -10
    vmin = 10

class C51NN(NatureCNN):
    def __init__(self, cfg: C51Config):
        super(C51NN, self).__init__(cfg)
        self.cfg = cfg
        self.fc2 = nn.Linear(512, cfg.act_dim * cfg.atoms)
        self.fc2.apply(lambda m: init(m, 0.01))
        self.register_buffer(
            'atoms',
            torch.linspace(cfg.vmin, cfg.vmax, cfg.atoms)
        )
        self.atoms = torch.linspace(
            cfg.vmin, cfg.vmax, cfg.atoms).view(1, 1, -1)
        self.delta = (cfg.vmax - cfg.vmin) / (cfg.atoms - 1)

    def forward(self, x):
        x = self.convs(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x

class C51Learner(Learner):
    def train_step(self, batch):
        obs, actions, rewards, terminals, obs_next = batch
        with torch.no_grad():
            prob_next = self.model_target(obs_next)
            a_next = prob_next.mul(self.model.atoms).sum(-1).argmax(-1)

            prob_next = prob_next.gather(1, a_next.unsqueeze(-1))


class C51Trainer(Trainer):
    def __init__(self, cfg: C51Config):
        super(C51Trainer, self).__init__(cfg)
        self.learner = C51Learner(cfg, self.learner.model)


class C51TrainerNode(TrainerNode):
    def __init__(self, cfg: C51Config, actors):
        super(C51TrainerNode, self).__init__(cfg, actors)
        self.learner = C51Learner(cfg, self.learner.model)

def make_program(cfg: C51Config):
    program = lp.Program("dqn")
    with program.group("actors"):
        actors = [
            program.add_node(lp.CourierNode(ActorNode, cfg, rank))
            for rank in range(cfg.num_actors)
        ]

    node = lp.CourierNode(C51TrainerNode, cfg=cfg, actors=actors)
    program.add_node(node, label="trainer")
    return program


def main(cfg: C51Config):
    set_random_seed(cfg.seed)
    # if cfg.use_lp:
    #     program = make_program(cfg)
    #     lp.launch(program, launch_type="local_mp", terminal="tmux_session")
    # else:
    #     trainer = C51Trainer(cfg)
    #     trainer.run()


if __name__ == '__main__':
    from wonderwords import RandomWord
    import tyro
    import git

    cfg = tyro.cli(C51Config)
    env = make_atari(cfg.game, 1)
    cfg.obs_shape = env.observation_space.shape[1:]
    cfg.act_dim = env.action_space[0].n
    env.close()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    wordstr = "-".join(RandomWord().random_words(2))
    sha = git.Repo(search_parent_directories=True).head.object.hexsha[:7]
    cfg.expname = f"c51-{cfg.game}-{wordstr}"
    cfg.logdir = f"{cfg.logdir}/c51-{cfg.game}-{timestr}-{sha}-{wordstr}"
    os.makedirs(cfg.logdir, exist_ok=False)

    main(cfg)
