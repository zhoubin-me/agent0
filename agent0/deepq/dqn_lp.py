import launchpad as lp
from agent0.deepq.dqn import Config, Actor, Learner, ReplayBuffer
from typing import List
import logging
from concurrent import futures
import time
import torch
from tqdm import tqdm

class ActorNode:
    def __init__(self, cfg: Config, rank: int):
        self.actor = Actor(cfg)
        self.rank = rank

    def sample(self, epsilon):
        return self.rank, self.actor.sample(epsilon)


class ReplayNode:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.replay = ReplayBuffer(cfg.replay_size)
        self.data = None
        self.stream = torch.cuda.Stream()

    def preload(self):
        transitions = self.replay.sample()
        self.data = list(torch.from_numpy(x).pin_memory() for x in transitions)
        with torch.cuda.stream(self.stream):
            self.data = list(
                x.cuda(non_blocking=True) for x in self.data
            )
        
    def sample(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        return data

    def extend(self, data):
        self.replay.extend(data)

class TrainerNode:
    def __init__(self, cfg: Config, actors: List[ActorNode], replay: ReplayNode):
        self.cfg = cfg
        self.actors = actors
        self.learner = Learner(cfg)
        self.replay = ReplayNode(cfg)
        self.epsilon_fn = (
            lambda step: cfg.min_eps
            if step > cfg.exploration_steps
            else (1.0 - step / cfg.exploration_steps) + cfg.min_eps
        )
        self.steps = 1

    def run(self):
        sample_tasks = [x.futures.sample(1.0) for x in self.actors]
        pbar = tqdm(total=self.cfg.training_start_steps, desc="Filling replay buffer")
        while pbar.n < pbar.total:
            dones, not_dones = futures.wait(sample_tasks, return_when=futures.FIRST_COMPLETED)
            sample_tasks = list(dones) + list(not_dones)
            rank, (transitions, qs, rs) = sample_tasks.pop(0).result()
            sample_tasks.append(self.actors[rank].futures.sample(1.0))
            self.replay.extend(transitions)
            pbar.update(len(transitions))

        step_frames = self.cfg.num_envs * self.cfg.sample_steps
        self.replay.preload()
        for step in range(self.cfg.total_steps // step_frames + 1):
            epsilon = self.epsilon_fn(self.steps)
            dones, not_dones = futures.wait(sample_tasks, return_when=futures.FIRST_COMPLETED)
            sample_tasks = list(dones) + list(not_dones)
            rank, (transitions, qs, rs) = sample_tasks.pop(0).result()
            sample_tasks.append(self.actors[rank].futures.sample(epsilon))
            self.replay.extend(transitions)
            


def make_program():
    cfg = Config()
    cfg.obs_shape = (4, 84, 84)
    cfg.act_dim = 4
    cfg.num_actors = 2

    program = lp.Program("dqn")
    with program.group("actors"):
        actors = [
            program.add_node(lp.CourierNode(ActorNode, cfg, rank))
            for rank in range(cfg.num_actors)
        ]
    # with program.group("replay"):
    #     replay = program.add_node(lp.CourierNode(ReplayNode, cfg))
    node = lp.CourierNode(TrainerNode, cfg=cfg, actors=actors, replay=None)
    program.add_node(node, label="trainer")
    return program



if __name__ == '__main__':
    program = make_program()
    lp.launch(program, launch_type="local_mp", terminal="tmux_session")

