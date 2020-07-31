import ray
import time

from src.agents.dqn_agent import Trainer

if __name__ == '__main__':

    ray.init(num_cpus=10, num_gpus=2)

    trainer = Trainer()    

    trainer.run()




