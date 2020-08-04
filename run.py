import argparse
import json

from src.agents.deepq_agent import default_hyperparams, run


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
    run(**kwargs)
