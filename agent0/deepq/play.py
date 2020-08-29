import pickle

import torch

from agent0.common.utils import parse_arguments
from agent0.deepq.actor import Actor
from agent0.deepq.config import Config


# import ray


def main():
    cfg = Config()
    args = parse_arguments(cfg)
    cfg = Config(**vars(args))
    cfg.update()

    # ray.init()
    if 'best.pth' in cfg.restore_checkpoint:
        state_dict = torch.load(cfg.restore_checkpoint, map_location=torch.device('cpu'))['model']
    else:
        with open(cfg.restore_checkpoint, 'rb') as f:
            data = pickle.load(f)
            state_dict = data['model']
    actor = Actor(rank=0, **vars(cfg))
    sample_op = actor.sample(cfg.actor_steps, 0.01, state_dict, testing=True, test_episodes=100, render=True)
    _, rs, qs, _, speed = sample_op


if __name__ == '__main__':
    main()
