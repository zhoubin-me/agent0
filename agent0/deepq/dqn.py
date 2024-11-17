import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from copy import deepcopy
from collections.abc import Sequence

from agent0.common.atari_wrappers import make_atari
import lz4.block
import random
from dataclasses import dataclass, asdict
import logging
import wandb
import os
import time
import tyro
from tqdm import tqdm

@dataclass
class Config:
    game: str = 'breakout'
    num_envs: int = 16

    use_wandb: bool = False
    logdir: str = 'logdir'

    num_envs: int = 16
    sample_steps: int = 80
    min_eps: float = 0.01
    test_eps: float = 0.001
    test_steps: int = 1000
    test_freq: int = 750

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

def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)


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
    def __init__(self, cfg: Config, model):
        self.cfg = cfg
        self.envs = make_atari(cfg.game, cfg.num_envs)
        self.obs, _ = self.envs.reset()
        self.model = model

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

    def sample(self, epsilon, test=False):
        rs, qs, transitions = [], [], []
        for _ in range(self.cfg.sample_steps):
            action, qt_max = self.act(epsilon)
            obs_next, reward, terminal, truncated, info = self.envs.step(action)
            if not test:
                done = np.logical_or(terminal, truncated)
                done = np.logical_or(done, info["lifeloss"])

                for st, at, rt, dt, st_next in zip(self.obs, action, reward, done, obs_next):
                    frames = np.concat([st, st_next], axis=0)
                    frames = lz4.block.compress(frames)
                    transitions.append((frames, at, rt, dt))

            self.obs = obs_next
            qs.append(qt_max)
            if "episode" in info:
                rs += info["episode"]['r'][info["_episode"]].tolist()
        return transitions, rs, qs

    
    def sync(self, state_dict):
        self.model.load_state_dict(state_dict)




class Learner:
    def __init__(self, cfg: Config, model):
        self.cfg = cfg
        self.model = model
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
        frames = frames.view(self.cfg.batch_size, -1, *self.cfg.obs_shape[1:])
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


class ReplayBuffer(Sequence):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data = deque(maxlen=cfg.replay_size)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames, at, rt, dt = self.data[idx]
        frames = np.frombuffer(lz4.block.decompress(frames), dtype=np.uint8)
        return np.array(frames), at, rt, dt

    def extend(self, transitions):
        self.data.extend(transitions)
    
    def sample(self):
        transitions = random.sample(self, self.cfg.batch_size)
        return map(lambda x: torch.from_numpy(np.array(x)),
                   zip(*transitions))
    
class Trainer:
    def __init__(self, cfg: Config, model):
        self.cfg = cfg
        self.model = model
        self.learner = Learner(cfg, model)
        self.actor = Actor(cfg, model)
        self.buffer = ReplayBuffer(cfg)
        self.steps = 0
        self.epsilon_fn = (
            lambda step: cfg.min_eps
            if step > cfg.exploration_steps
            else (1.0 - step / cfg.exploration_steps) + cfg.min_eps
        )

        os.makedirs(cfg.logdir, exist_ok=True)
        # Initialize wandb and logging
        if cfg.use_wandb:
            wandb.init(project="dqn-atari", config=asdict(cfg), dir=cfg.logdir)
            
        # Set up logging
        self.logger = logging.getLogger("dqn")
        self.logger.setLevel(logging.INFO)
        
        # Add console handler
        ch = logging.StreamHandler() 
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        timestr = time.strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(cfg.logdir, f'training_{timestr}.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        
    def train(self):
        # Initial exploration
        pbar = tqdm(total=self.cfg.training_start_steps, desc="Filling replay buffer")
        while len(self.buffer) < self.cfg.training_start_steps:
            transitions, _, _ = self.actor.sample(epsilon=1.0)
            self.buffer.extend(transitions)
            pbar.update(len(transitions))
        pbar.close()

        # Main training loop
        returns = []
        losses = []
        qvals = []
        
        while self.steps < self.cfg.total_steps:
            # Sample transitions
            if self.steps % (self.cfg.sample_steps * self.cfg.num_envs * self.cfg.test_freq) == 0:
                self.evaluate()

            epsilon = self.epsilon_fn(self.steps)
            transitions, rs, qs = self.actor.sample(epsilon)
            self.buffer.extend(transitions)
            returns.extend(rs)
            qvals.extend(qs)
            self.steps += self.cfg.sample_steps * self.cfg.num_envs
            
            # Train
            for _ in range(self.cfg.learner_steps):
                batch = self.buffer.sample()
                loss = self.learner.step(batch)
                losses.append(loss)
        
            self.log(qvals=qvals[-20:],
                     losses = losses[-20:],
                     returns = returns[-20:],
                     test=False)
        
        self.evaluate()
        self.actor.envs.close()
    
    def evaluate(self):
        returns = []
        qvals = []
        pbar = tqdm(total=self.cfg.test_steps, desc="Testing")
        for _ in range(self.cfg.test_steps):
            _, rs, qs = self.actor.sample(epsilon=self.cfg.test_eps, test=True)
            returns.extend(rs)
            qvals.extend(qs)
            pbar.update(1)
        pbar.close()
        self.log(qvals=qvals, losses=[0, 0], returns=returns, test=True)

    def log(self, qvals, losses, returns, test=False):
        # Check if enough data exists for statistics
        if len(qvals) <= 1 or len(losses) <= 1 or len(returns) <= 1:
            return
        # Calculate statistics
        qvals_mean = np.mean(qvals)
        qvals_max = np.max(qvals)
        losses_mean = np.mean(losses)
        losses_max = np.max(losses)
        returns_mean = np.mean(returns)
        returns_max = np.max(returns)

        # Log to wandb if enabled
        prefix = 'train' if not test else 'test'
        if self.cfg.use_wandb:
            wandb.log({
                'steps': self.steps,
                f'qvals/{prefix}_mean': qvals_mean,
                f'qvals/{prefix}_max': qvals_max,
                f'losses/{prefix}_mean': losses_mean,
                f'losses/{prefix}_max': losses_max,
                f'returns/{prefix}_mean': returns_mean,
                f'returns/{prefix}_max': returns_max,
                f'returns/{prefix}_count': len(returns),
            })

        # Log to logger
        prefix = 'Train' if not test else "Test Result====>\n"
        self.logger.info(
            f"{prefix} - Steps: {self.steps:7d} | "
            f"Q-Values - Mean: {qvals_mean:.3f}, Max: {qvals_max:.3f} | "
            f"Losses - Mean: {losses_mean:.3f}, Max: {losses_max:.3f} | "
            f"Returns - Mean: {returns_mean:.3f}, Max: {returns_max:.3f} Count: {len(returns)}"
        )



        

def main():
    cfg = tyro.cli(Config)
    env = make_atari(cfg.game, 1)
    cfg.obs_shape = env.observation_space.shape[1:]
    cfg.act_dim = env.action_space[0].n
    env.close()
    
    # Create and train agent
    model = NatureCNN(cfg).cuda()
    trainer = Trainer(cfg, model)
    trainer.train()


if __name__ == "__main__":
    main()
