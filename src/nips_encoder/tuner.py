import argparse
import json
import random

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining

from src.nips_encoder.trainer import Trainer, default_hyperparams


def parse_arguments(params):
    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return vars(args)


if __name__ == '__main__':
    params = default_hyperparams()
    kwargs = parse_arguments(params)

    ray.init(memory=20 * 2 ** 30, object_store_memory=80 * 2 ** 30)
    reporter = CLIReporter(
        metric_columns=["game", "speed", "loss", "adam_lr", "time_remain", "time_past"]
    )

    pbt_scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        metric='loss',
        mode='min',
        perturbation_interval=50,
        hyperparam_mutations={
            "adam_lr": lambda: random.uniform(1e-5, 1e-4),
        })

    analysis = tune.run(
        Trainer,
        name='nips_encoder_tune',
        verbose=1,
        stop=lambda trial_id, result: result['epoch'] > kwargs['epoches'],
        checkpoint_at_end=True,
        progress_reporter=reporter,
        checkpoint_freq=800,
        reuse_actors=True,
        scheduler=pbt_scheduler,
        resources_per_trial={"gpu": 1},
        fail_fast=True,
        config={
            "adam_lr": tune.grid_search([5e-5, 1e-4, 2e-4, 5e-4]),
        }
    )
