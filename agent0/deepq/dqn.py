# Standard library imports
import logging
from collections import deque
from concurrent import futures
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import List

# Third party imports
import lz4.block
import mediapy
import numpy as np
import launchpad as lp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

# Local imports
from agent0.common.atari_wrappers import make_atari
from agent0.common.utils import DataPrefetcher, set_random_seed, init


@dataclass
class Config:
    game: str = 'Breakout'
    logdir: str = 'logdir'
    seed: int = 42

    use_wandb: bool = False
    use_tb: bool = False
    use_lp: bool = False
    record_video: bool = False
    exp_name = None

    num_envs: int = 16
    num_actors: int = 2
    sample_steps: int = 80
    min_epsilon: float = 0.01
    test_epsilon: float = 0.001
    test_max_steps: int = 450
    test_rs_len: int = 32
    test_freq: int = 320

    discount: float = 0.99
    batch_size: int = 512
    learning_rate: float = 5e-4
    target_update_freq: int = 500
    learner_steps: int = 20

    total_steps: int = int(1e7)
    training_start_steps: int = int(1e5)
    exploration_steps: int = int(1e6)
    replay_size: int = int(1e6)

    act_dim = None
    obs_shape = None

class NatureCNN(nn.Module):
    def __init__(self, cfg: Config):
        super(NatureCNN, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(cfg.obs_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.convs.apply(lambda m: init(m, nn.init.calculate_gain("relu")))

        def conv2d_size_out(size, kernel_size, stride, padding):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        height, width = cfg.obs_shape[1], cfg.obs_shape[2]
        for kernel_size, stride in [(8, 4), (4, 2), (3, 1)]:
            height = conv2d_size_out(height, kernel_size, stride, 0)
            width = conv2d_size_out(width, kernel_size, stride, 0)
        feature_dim = 64 * height * width

        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, cfg.act_dim)

        self.fc1.apply(lambda m: init(m, nn.init.calculate_gain("relu")))
        self.fc2.apply(lambda m: init(m, 0.01))

    def forward(self, x):
        x = self.convs(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Actor:
    def __init__(self, cfg: Config, model=None):
        self.cfg = cfg
        self.envs = make_atari(cfg.game, cfg.num_envs)
        self.obs, _ = self.envs.reset()
        self.model = NatureCNN(cfg).cuda() if model is None else model

    @torch.no_grad()
    def act(self, epsilon):
        action_random = np.random.randint(0, self.cfg.act_dim, self.cfg.num_envs)
        if epsilon >= 1.0:
            return action_random, 0
        
        obs = torch.from_numpy(self.obs).cuda().float().div(255.0)
        qvals = self.model(obs)
        qvals, action_greedy = qvals.max(dim=-1)
        action_greedy = action_greedy.cpu().numpy()
        action = np.where(
            np.random.rand(self.cfg.num_envs) > epsilon,
            action_greedy,
            action_random,
        )
        return action, qvals.mean().item()

    def sample(self, epsilon):
        rs, qs, transitions = [], [], []
        for _ in range(self.cfg.sample_steps):
            action, qt_max = self.act(epsilon)
            obs_next, reward, terminal, truncated, info = self.envs.step(action)
            done = np.logical_or(terminal, truncated)
            done = np.logical_or(done, info["lifeloss"])

            for st, at, rt, dt, st_next in zip(self.obs, action, reward, done, obs_next):
                frames = lz4.block.compress(np.concat([st, st_next], axis=0))
                transitions.append((frames, at, rt, dt))

            self.obs = obs_next
            qs.append(qt_max)
            if "episode" in info:
                rs += info["episode"]['r'][info["_episode"]].tolist()
        return transitions, rs, qs


class Learner:
    def __init__(self, cfg: Config, model=None):
        self.cfg = cfg
        self.model = NatureCNN(cfg).cuda() if model is None else model
        self.model_target = deepcopy(self.model)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            cfg.learning_rate,
            eps=1e-2 / cfg.batch_size
        )
        self.update_steps = 0
    
    def train_step(self, batch):
        obs, actions, rewards, terminals, obs_next = batch
        with torch.no_grad():
            next_q = self.model_target(obs_next)
            next_q, _ = next_q.max(dim=1)
            target_q = rewards + self.cfg.discount * (1 - terminals) * next_q
        curr_q = self.model(obs)
        curr_q = curr_q.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(curr_q, target_q, reduction='sum')
        return loss
    
    def step(self, data):
        frames, actions, rewards, terminals = map(
            lambda x: x.cuda().float(), data
        )
        frames = frames.view(-1, self.cfg.obs_shape[0] * 2, *self.cfg.obs_shape[1:])
        frames = frames.div(255.0)
        obs, obs_next = torch.split(frames, self.cfg.obs_shape[0], dim=1)

        batch = obs, actions, rewards, terminals, obs_next
        loss = self.train_step(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_steps += 1
        if self.update_steps % self.cfg.target_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
            
        return loss.item()


class ReplayBuffer(Dataset):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.replay = deque(maxlen=cfg.replay_size)
        self.batch = None
        self.stream = torch.cuda.Stream()

    def __len__(self):
        return len(self.replay)

    def __getitem__(self, idx):
        frames, at, rt, dt = self.replay[idx]
        frames = np.frombuffer(lz4.block.decompress(frames), dtype=np.uint8)
        return np.array(frames), at, rt, dt

    def extend(self, data):
        self.replay.extend(data)

class Trainer:
    def __init__(self, cfg: Config, actors=None):
        self.cfg = cfg
        model = NatureCNN(cfg).cuda()
        learner = Learner(cfg, model)
        replay = ReplayBuffer(cfg)
        self.actor = actors if actors is not None else Actor(cfg, model)
        self.learner = learner
        self.replay = replay
        self.dataloder = None
        self.steps = 0
        self.epsilon_fn = (
            lambda step: cfg.min_epsilon
            if step > cfg.exploration_steps
            else (1.0 - step / cfg.exploration_steps) + cfg.min_epsilon
        )

        # Set up logging
        self.logger = logging.getLogger("dqn")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        fh = logging.FileHandler(f"{cfg.logdir}/train.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Not using launchpad need to add console handler
        if not cfg.use_lp:
            ch = logging.StreamHandler() 
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        

    def get_data_fetcher(self):
        sampler = RandomSampler(
            self.replay,
            replacement=True
        )
        
        data_loader = DataLoader(
            self.replay,
            sampler=sampler,
            batch_size=self.cfg.batch_size,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
        )

        data_fetcher = DataPrefetcher(data_loader)
        return data_fetcher

    def fill_replay(self, epsilon=1.0):
        # No need to sync model weigths as it's shared by actor and learner
        transitions, rs, qs = self.actor.sample(epsilon)
        self.replay.extend(transitions)
        return rs, qs
    
    def run(self):
        # Initial exploration
        pbar = tqdm(total=self.cfg.training_start_steps, desc="Filling replay buffer")
        step_frames = self.cfg.num_envs * self.cfg.sample_steps
        while pbar.n < pbar.total:
            self.fill_replay(1.0)
            pbar.update(step_frames)
        pbar.close()
        data_iter = self.get_data_fetcher()

        # Main training loop
        for _ in range(self.cfg.total_steps // step_frames + 1):
            if self.steps % (step_frames * self.cfg.test_freq) == 0:
                self.test()

            epsilon = self.epsilon_fn(self.steps)
            rs, qs = self.fill_replay(epsilon)
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
        
        self.final()
    
    def final(self):
        self.test()
        self.actor.envs.close()
        wandb.finish()
    
    def test(self):
        rss = []
        qss = []
        video = []
        pbar = tqdm(total=self.cfg.test_max_steps, desc="Testing")
        while pbar.n < pbar.total:
            transitions, rs, qs = self.actor.sample(epsilon=self.cfg.test_epsilon)
            rss.extend(rs)
            qss.extend(qs)
            if self.cfg.record_video:
                video.extend([x[0] for x in transitions])
            pbar.update(1)
            if len(rss) > self.cfg.test_rs_len:
                break
        pbar.close()
        logdata = dict(
            qvals=qss,
            loss=[],
            returns=rss
        )
        self.log(logdata, video=video, test=True)

    def log(self, logdata, video=[], test=False):
        if self.cfg.use_wandb and wandb.run is None:
            wandb.init(
                project="dqn-atari", 
                config=asdict(self.cfg), 
                dir=self.cfg.logdir,
                name=self.cfg.exp_name
            )
        if self.cfg.use_tb and not hasattr(self, 'writer'):
            self.writer = SummaryWriter(log_dir=self.cfg.logdir)

        data_stat = dict()
        prefix = 'train' if not test else 'test '
        logstr = f"{prefix} - Frames: {self.steps:8d}"
        for k, v in logdata.items():
            if len(v) > 0:
                data_stat[f"{k}/{prefix}_mean"] = np.mean(v)
                data_stat[f"{k}/{prefix}_max"] = np.max(v)
                data_stat[f"{k}/{prefix}_min"] = np.min(v)
                if k == "returns":
                    data_stat[f"{k}/{prefix}_count"] = len(v)
                    logstr += f" | {k} - Mean {np.mean(v):5.0f}, Max {np.max(v):5.0f}, Count: {len(v):3d}"
                else:
                    logstr += f" | {k} - Mean {np.mean(v):5.2f}, Max {np.max(v):5.2f}"

        self.logger.info(logstr)
        if test:
            self.logger.info("=" * 100)
        
        data_stat.update(frames=self.steps)

        if self.cfg.use_tb:
            for k, v in data_stat.items():
                self.writer.add_scalar(k, v, self.steps)
            for k, v in logdata.items():
                if len(v) > 0:
                    self.writer.add_histogram(k, np.array(v), self.steps)
        


        if self.cfg.record_video and len(video) > 0:
            frames = [np.frombuffer(lz4.block.decompress(x), dtype=np.uint8) for x in video]
            frames = [x.reshape(-1, *self.cfg.obs_shape[1:])[0] for x in frames]
            video_path = f"{self.cfg.logdir}/{self.steps:09d}.mp4"
            mediapy.write_video(video_path, frames, fps=15)
            data_stat.update(video=wandb.Video(video_path))
        
        if self.cfg.use_wandb:
            wandb.log(data_stat)

class ActorNode:
    def __init__(self, cfg: Config, rank: int):
        self.actor = Actor(cfg)
        self.rank = rank

    def sample(self, epsilon, state_dict=None):
        if state_dict is not None:
            self.actor.model.load_state_dict(state_dict)
        return self.rank, self.actor.sample(epsilon)

    def close(self):
        self.actor.envs.close()

class TrainerNode(Trainer):
    def __init__(self, cfg: Config, actors: List[ActorNode]):
        super(TrainerNode, self).__init__(cfg, actors)
        self.tasks = [x.futures.sample(1.0) for x in actors]

    def fill_replay(self, epsilon=1.0):
        dones, not_dones = futures.wait(self.tasks, return_when=futures.FIRST_COMPLETED)
        self.tasks = list(dones) + list(not_dones)
        rank, (transitions, rs, qs) = self.tasks.pop(0).result()
        if epsilon < 1.0:
            state_dict = self.learner.model.state_dict()
        else:
            state_dict = None
        self.tasks.append(self.actor[rank].futures.sample(epsilon, state_dict))
        self.replay.extend(transitions)
        return rs, qs

    def test(self):
        rss = []
        qss = []
        video = []

        futures.wait(self.tasks, return_when=futures.ALL_COMPLETED)
        pbar = tqdm(total=self.cfg.test_max_steps, desc="Testing")
        while pbar.n < pbar.total:
            dones, not_dones = futures.wait(self.tasks, return_when=futures.FIRST_COMPLETED)
            self.tasks = list(dones) + list(not_dones)
            rank, (transitions, rs, qs) = self.tasks.pop(0).result()
            rss.extend(rs)
            qss.extend(qs)
            if self.cfg.record_video and rank == 0:
                video.extend([x[0] for x in transitions])
            self.tasks.append(self.actor[rank].futures.sample(self.cfg.test_epsilon, None))
            pbar.update(1)
            if len(rss) > self.cfg.test_rs_len:
                break
        pbar.close()

        futures.wait(self.tasks, return_when=futures.ALL_COMPLETED)
        epsilon = self.epsilon_fn(self.steps)
        self.tasks = [x.futures.sample(epsilon) for x in self.actor]

        logdata = dict(
            qvals=qss,
            loss=[],
            returns=rss
        )        
        self.log(logdata, video=video, test=True)


    def final(self):
        self.test()
        futures.wait(self.tasks, return_when=futures.ALL_COMPLETED)
        futures.wait([x.futures.close() for x in self.actor], return_when=futures.ALL_COMPLETED)
        wandb.finish()
        lp.stop()

def make_program(cfg: Config):
    program = lp.Program("dqn")
    with program.group("actors"):
        actors = [
            program.add_node(lp.CourierNode(ActorNode, cfg, rank))
            for rank in range(cfg.num_actors)
        ]

    node = lp.CourierNode(TrainerNode, cfg=cfg, actors=actors)
    program.add_node(node, label="trainer")
    return program


def main(cfg: Config):
    set_random_seed(cfg.seed)
    if cfg.use_lp:
        program = make_program(cfg)
        lp.launch(program, launch_type="local_mp", terminal="tmux_session")
    else:
        trainer = Trainer(cfg)
        trainer.run()


if __name__ == '__main__':
    from wonderwords import RandomWord
    import tyro
    import git
    import time
    import os

    cfg = tyro.cli(Config)
    env = make_atari(cfg.game, 1)
    cfg.obs_shape = env.observation_space.shape[1:]
    cfg.act_dim = env.action_space[0].n
    env.close()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    wordstr = "-".join(RandomWord().random_words(2))
    sha = git.Repo(search_parent_directories=True).head.object.hexsha[:7]
    cfg.exp_name = f"dqn-{cfg.game}-{wordstr}"
    cfg.logdir = f"{cfg.logdir}/dqn-{cfg.game}-{timestr}-{sha}-{wordstr}"
    os.makedirs(cfg.logdir, exist_ok=False)

    main(cfg)