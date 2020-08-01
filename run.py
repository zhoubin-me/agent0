import os

import ray
import neptune
from src.agents.deepq_agent import Agent

if __name__ == '__main__':

    ray.init(num_cpus=50, num_gpus=4)
    agent = Agent()
    agent.run()




