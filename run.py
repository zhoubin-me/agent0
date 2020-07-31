import ray
import neptune
from src.agents.dqn_agent import Agent

if __name__ == '__main__':

    ray.init(num_cpus=10, num_gpus=2)
    neptune.init('zhoubinxyz/agentzero')
    agent = Agent()
    agent.run()




