import argparse
import json
import random

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

from src.agents.deepq_agent import Trainer, default_hyperparams


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

    ray.init(memory=100 * 2 ** 30, object_store_memory=200 * 2 ** 30)

    pbt_scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        metric='ep_reward_train',
        mode='max',
        perturbation_interval=600.0,
        hyperparam_mutations={
            "adam_lr": lambda: random.uniform(1e-5, 1e-3),
            "target_update_freq": [500, 250],
            "agent_train_freq": [20, 16],
        })

    analysis = tune.run(
        Trainer,
        name='atari_dqn',
        checkpoint_at_end=True,
        checkpoint_freq=800,
        reuse_actors=True,
        verbose=1,
        scheduler=pbt_scheduler,
        resources_per_trial={"gpu": 4},
        fail_fast=True,
        config={
            "adam_lr": 1e-3,
            "game": "Breakout"
        }
    )

    print("Best config: ", analysis.get_best_config(metric="final_test_rewards"))
    df = analysis.dataframe()
    df.to_csv('ptb_atari_breakout.csv')
