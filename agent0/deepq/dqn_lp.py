import launchpad as lp
from agent0.deepq.dqn import Actor, Learner, ReplayBuffer, Config
import time

class Shared:
    def __init__(self):
        self.actor_ready = False
        self.actor_should_sample = False

        self.replay_ready = False
        self.replay_should_sample = False
        
        self.leaner_ready = False
        self.leaner_should_learn = False

class ActorNode:
    def __init__(self, cfg: Config, actor: Actor, signal: Signal, Trainer: None):
        self.cfg = cfg
        self.actor = actor
        self.signal = signal
        self.transitions = None

    def run(self):
        while True:
            if self.signal.actor_should_sample:
                self.transitions, rs, qs = self.actor.sample()
                self.signal.actor_ready = True
                self.signal.actor_should_sample = False
            else:
                time.sleep(0.01)


class ReplayNode:
    def __init__(self, cfg: Config, replay: ReplayBuffer, signal: Signal, Trainer: None):
        self.cfg = cfg
        self.replay = replay
        self.signal = signal
        self.transitions = None

    def run(self):
        while True:
            if self.signal.replay_should_sample:
                self.transitions = self.replay.sample()
                self.signal.replay_ready = True
                self.signal.replay_should_sample = False
            else:
                time.sleep(0.01)


class LearnerNode:
    def __init__(self, cfg: Config, learner: Learner, signal: Signal, Trainer: None):
        self.cfg = cfg
        self.learner = learner
        self.signal = signal
        self.transitions = None

    def run(self):
        while True:
            if self.signal.leaner_should_learn:
                loss = self.learner.step(self.transitions)
                self.signal.replay_ready = True
                self.signal.replay_should_sample = False
            else:
                time.sleep(0.01)


class TrainerNode:
    def __init__(self, cfg: Config, signal: Signal, actor, learner, replay):
        self.cfg = cfg
        self.signal = signal
        self.actor = actor
        self.learner = learner
        self.replay = replay


def make_program(cfg: Config):
    program = lp.Program("dqn")
    signal = Signal()

    actor = lp.CourierNode(ActorNode, signal)
    program.add_node(actor, label="actor")

    learner = lp.CourierNode(LearnerNode, signal)
    program.add_node(learner, label="leaner")

    replay = lp.CourierNode(ReplayNode, signal)
    program.add_node(replay, label="replay")

    trainer = lp.CourierNode(TrainerNode, cfg=cfg, actors=actors)
    program.add_node(trainer, label="trainer")
    return program
        


