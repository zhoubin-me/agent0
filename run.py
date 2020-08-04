import os

import neptune
import argparse
from src.agents.deepq_agent import Agent, default_hyperparams, run

def parse_arguments(params):
    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    params = default_hyperparams()
    kwargs = parse_arguments(params)
    agent = Agent(**kwargs)
<<<<<<< Updated upstream
    run(agent)
=======
    agent.run()


>>>>>>> Stashed changes


