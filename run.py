import os

import ray
import neptune
import argparse
from src.agents.deepq_agent import Agent, default_hyperparams

def parse_arguments(params):
    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    params = default_hyperparams()
    kwargs = parse_arguments(params)
    ray.init(num_cpus=50, num_gpus=4)
    agent = Agent(**kwargs)
    # agent.run()




