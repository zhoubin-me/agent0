import argparse

import ray
from ray import tune
from ray.tune import CLIReporter

from src.deepq.agent import default_hyperparams, Trainer


def parse_arguments(params):
    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    return vars(args)




if __name__ == '__main__':
    params = default_hyperparams()
    kwargs = parse_arguments(params)
    ray.init(memory=20 * 2 ** 30, object_store_memory=80 * 2 ** 30)
    reporter = CLIReporter(
        metric_columns=["game", "frames", "loss", "ep_reward_test", "ep_reward_train", "ep_reward_test_max",
                        "ep_reward_train_max", "time_past", "time_remain", "speed", "epsilon", "qmax"])

    analysis = tune.run(
        Trainer,
        name=kwargs['exp_name'],
        verbose=1,
        checkpoint_at_end=True,
        fail_fast=True,
        stop=lambda trial_id, result: result['frames'] > kwargs['total_steps'],
        checkpoint_freq=1000,
        config=dict(
            game=tune.grid_search(
                ['Breakout', 'Enduro', 'Seaquest', 'BeamRider', 'Pong', 'Asterix', 'Qbert', 'SpaceInvaders']),
            # game=tune.grid_search([kwargs['game']]),
            epoches=kwargs['total_steps'] // int(1e4),
            total_steps=kwargs['total_steps'],
            distributional=kwargs['distributional'],
            num_atoms=kwargs['num_atoms'],
            adamw=kwargs['adamw'],
            noisy=kwargs['noisy'],
        ),
        progress_reporter=reporter,
        resources_per_trial={"gpu": 1, "extra_gpu": 1},
    )
