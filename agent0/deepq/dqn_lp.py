import launchpad as lp
from agent0.deepq.dqn import Actor, Learner, ReplayBuffer
import time

# Define a courier that will handle signals between nodes
class SignalCourier:
    def __init__(self):
        self.actor_ready = False
        self.actor_should_sample = False

        self.data_ready = False
        self.data_should_sample = False
        
        self.learner_ready = False
        self.learner_shoud_step = False
    
    def stop(self):
        self._stop = True
        
    def pause(self):
        self._pause = True
        
    def resume(self):
        self._pause = False
        
    def should_stop(self):
        return self._stop
        
    def should_pause(self):
        return self._pause
    
    def should_sample(self):
        return self._sample

class ActorNode:
    def __init__(self, courier: SignalCourier, actor: Actor):
        self.courier = courier
        self.actor = actor

    def run(self):
        while True:
            if self.courier.actor_should_sample():
                self.actor.sample()

    def sync(self):
        

