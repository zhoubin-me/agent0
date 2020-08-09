import argparse
import json

import ray
from ray import tune
from ray.tune import CLIReporter

from src.deepq.agent import default_hyperparams, Trainer


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
        metric_columns=["game", "frames", "loss", "ep_reward_test", "ep_reward_train", "ep_reward_test_max",
                        "time_past", "time_remain", "speed", "epsilon"])

    analysis = tune.run(
        Trainer,
        name=kwargs['exp_name'],
        verbose=1,
        checkpoint_at_end=True,
        fail_fast=True,
        stop = lambda trial_id, result: result['frames'] > kwargs['total_steps'],
        checkpoint_freq=800,
        config={
            "game": tune.grid_search(['Breakout', 'BeamRider', 'Qbert', 'SpaceInvaders'])
        },
        progress_reporter=reporter,
        resources_per_trial={"gpu": 3},
    )

    print("Best config: ", analysis.get_best_config(metric="ep_reward_test"))
    df = analysis.dataframe()
    df.to_csv('out.csv')
