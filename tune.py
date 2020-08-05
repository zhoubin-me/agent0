import argparse
import json

import ray
from ray import tune

from src.agents.deepq_agent import run, default_hyperparams


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
    analysis = tune.run(
        run,
        config={
            "adam_lr": tune.grid_search([1e-3, 1e-4, 2e-4]),
            "target_update_freq": tune.grid_search([500, 200]),
            "agent_train_freq": tune.grid_search([20, 16]),
            "game": tune.grid_search(["Breakout"])
        },
        resources_per_trial={"gpu": 4},
    )
    print("Best config: ", analysis.get_best_config(metric="final_test_rewards"))
    df = analysis.dataframe()
    df.to_csv('out.csv')
