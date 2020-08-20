import argparse

import git
import ray
from ray import tune
from ray.tune import CLIReporter

from src.deepq.config import Config
from src.deepq.trainer import Trainer

def str2bool(s: str):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError(f'{s} is not a boolean')

def parse_arguments(config):
    parser = argparse.ArgumentParser()
    for k, v in vars(config).items():
        if type(v) == bool:
            parser.add_argument(f"--{k}", type=str2bool, default=str(v))
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    return Config(**vars(args))


def trial_str_creator(trial, sha):
    return "{}_{}_{}_{}".format(trial.trainable_name, trial.config['game'], sha, trial.trial_id)


def main():
    cfg = Config()
    cfg = parse_arguments(cfg)

    games = ['Breakout', 'Enduro', 'Seaquest', 'BeamRider', 'Pong', 'Asterix', 'Qbert', 'SpaceInvaders']
    cfg.update(
        num_atoms=None,
        game=tune.grid_search(games)
    )

    ray.init(memory=20 * 2 ** 30, object_store_memory=80 * 2 ** 30)
    reporter = CLIReporter(
        metric_columns=["frames", "loss", "ep_reward_test", "ep_reward_train", "ep_reward_test_max",
                        "ep_reward_train_max", "time_past", "time_remain", "speed", "epsilon", "qmax"])

    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head.object.hexsha, short=True)

    analysis = tune.run(
        Trainer,
        name=cfg.exp_name,
        verbose=1,
        checkpoint_at_end=True,
        fail_fast=True,
        restore=cfg.restore_ckpt,
        stop=lambda trial_id, result: result['frames'] > cfg.total_steps,
        checkpoint_freq=1000,
        trial_name_creator=tune.function(lambda trial: trial_str_creator(trial, sha)),
        config=vars(cfg),
        progress_reporter=reporter,
        resources_per_trial={"gpu": 1, "extra_gpu": 1},
    )

if __name__ == '__main__':
    main()
