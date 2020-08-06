import argparse
import json

import ray
from ray import tune
from ray.tune import Stopper

from src.deepq.agent import default_hyperparams, Trainer


def parse_arguments(params):
    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return vars(args)


class CustomStopper(Stopper):
    def __init__(self, max_frames):
        self.should_stop = False
        self.max_frames = max_frames

    def __call__(self, trial_id, result):
        if not self.should_stop and result['frames'] > self.max_frames:
            self.should_stop = True
        return self.should_stop

    def stop_all(self):
        return self.should_stop


if __name__ == '__main__':
    params = default_hyperparams()
    kwargs = parse_arguments(params)
    ray.init(memory=20 * 2 ** 30, object_store_memory=80 * 2 ** 30)
    stopper = CustomStopper(kwargs['total_steps'])
    analysis = tune.run(
        Trainer,
        name=kwargs['exp_name'],
        verbose=0,
        checkpoint_at_end=True,
        fail_fast=True,
        stop=stopper,
        checkpoint_freq=800,
        config={
            "exploration_ratio": tune.grid_search([0.1, 0.15]),
            "adam_lr": tune.grid_search([5e-4, 1e-4, 2e-4]),
            "agent_train_freq": tune.grid_search([15, 10]),
            "game": tune.grid_search(["Breakout"])
        },
        resources_per_trial={"gpu": 3},
    )

    print("Best config: ", analysis.get_best_config(metric="final_test_rewards"))
    df = analysis.dataframe()
    df.to_csv('out.csv')
