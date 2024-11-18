import launchpad as lp
from agent0.deepq.dqn import Config, Actor, Trainer
from agent0.common.atari_wrappers import make_atari
from typing import List
from concurrent import futures
import time
from tqdm import tqdm
import os

class ActorNode:
    def __init__(self, cfg: Config, rank: int):
        self.actor = Actor(cfg)
        self.rank = rank

    def sample(self, epsilon, state_dict=None):
        if state_dict is not None:
            self.actor.model.load_state_dict(state_dict)
        return self.rank, self.actor.sample(epsilon)

class TrainerNode(Trainer):
    def __init__(self, cfg: Config, actors: List[ActorNode]):
        super(TrainerNode, self).__init__(cfg, actors)

    def run(self):
        sample_tasks = [x.futures.sample(1.0) for x in self.actor]
        pbar = tqdm(total=self.cfg.training_start_steps, desc="Filling replay buffer")
        while pbar.n < pbar.total:
            dones, not_dones = futures.wait(sample_tasks, return_when=futures.FIRST_COMPLETED)
            sample_tasks = list(dones) + list(not_dones)
            rank, (transitions, qs, rs) = sample_tasks.pop(0).result()
            sample_tasks.append(self.actor[rank].futures.sample(1.0))
            self.replay.extend(transitions)
            pbar.update(len(transitions))

        pbar.close()
        data_iter = self.get_data_fetcher()
        step_frames = self.cfg.num_envs * self.cfg.sample_steps
        for _ in range(self.cfg.total_steps // step_frames + 1):
            if self.steps % (step_frames * self.cfg.test_freq) == 1:
                self.test()
            epsilon = self.epsilon_fn(self.steps)
            dones, not_dones = futures.wait(sample_tasks, return_when=futures.FIRST_COMPLETED)
            sample_tasks = list(dones) + list(not_dones)
            rank, (transitions, rs, qs) = sample_tasks.pop(0).result()
            sample_tasks.append(self.actor[rank].futures.sample(epsilon, self.learner.model.state_dict()))
            self.replay.extend(transitions)
            self.steps += step_frames
            
            # Train
            losses = []
            for _ in range(self.cfg.learner_steps):
                data = data_iter.next()
                loss = self.learner.step(data)
                losses.append(loss)
            
            logdata = dict(
                loss=losses,
                qvals=qs,
                returns=rs,
            )
            self.log(logdata, test=False)

def make_program(cfg):
    program = lp.Program("dqn")
    with program.group("actors"):
        actors = [
            program.add_node(lp.CourierNode(ActorNode, cfg, rank))
            for rank in range(cfg.num_actors)
        ]

    node = lp.CourierNode(TrainerNode, cfg=cfg, actors=actors)
    program.add_node(node, label="trainer")
    return program



if __name__ == '__main__':
    from wonderwords import RandomWord
    import tyro
    import git

    cfg = tyro.cli(Config)
    env = make_atari(cfg.game, 1)
    cfg.obs_shape = env.observation_space.shape[1:]
    cfg.act_dim = env.action_space[0].n
    env.close()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    wordstr = "-".join(RandomWord().random_words(2))
    sha = git.Repo(search_parent_directories=True).head.object.hexsha[:7]
    cfg.exp_name = f"{cfg.game}-{wordstr}"
    cfg.logdir = f"{cfg.logdir}/{cfg.game}-{timestr}-{sha}-{wordstr}"
    os.makedirs(cfg.logdir, exist_ok=False)

    program = make_program(cfg)
    lp.launch(program, launch_type="local_mp", terminal="tmux_session")