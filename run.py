import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8,9'

import ray
import neptune
from src.agents.dqn_agent import Agent

if __name__ == '__main__':

    ray.init(num_cpus=32, num_gpus=2)
    agent = Agent()
    agent.benchmark()




