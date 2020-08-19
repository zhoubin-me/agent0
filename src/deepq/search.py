import argparse

import git
import ray
from ray import tune
from ray.tune import CLIReporter

from src.deepq.config import Config
from src.deepq.trainer import Trainer


def parse_arguments(config):
    parser = argparse.ArgumentParser()
    for k, v in vars(config).items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    return Config(**vars(args))


def trial_str_creator(trial, sha):
    return "{}_{}_{}".format(trial.trainable_name, sha, trial.trial_id)


if __name__ == '__main__':
    cfg = Config()
    cfg = parse_arguments(cfg)
    ray.init(memory=20 * 2 ** 30, object_store_memory=80 * 2 ** 30)

    reporter = CLIReporter(
        metric_columns=["game", "frames", "loss", "ep_reward_test", "ep_reward_train", "ep_reward_test_max",
                        "ep_reward_train_max", "time_past", "time_remain", "speed", "epsilon", "qmax"])

    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head.object.hexsha, True)

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
        config=dict(
            game=tune.grid_search(
                ['Breakout', 'Enduro', 'Seaquest', 'BeamRider', 'Pong', 'Asterix', 'Qbert', 'SpaceInvaders']),
            # game=tune.grid_search([kwargs['game']]),
            epochs=cfg.epochs,
            total_steps=cfg.total_steps,
            distributional=cfg.distributional,
            noisy=cfg.noisy,
            num_atoms=cfg.default_num_atoms(),
        ),
        progress_reporter=reporter,
        resources_per_trial={"gpu": 1, "extra_gpu": 1},
    )
