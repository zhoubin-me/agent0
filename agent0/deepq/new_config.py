from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

AlgoEnum = Enum('Algo', {k: i for i, k in enumerate(['dqn', 'c51'])})
ActorEnum = Enum('Actor', {k: i for i, k in enumerate(['greedy', 'random', 'eps-greedy'])})
ReplayEnum = Enum('Replay', {k: i for i, k in enumerate(['uniform', 'prior'])})
ModeEnum = Enum('Mode', {k: i for i, k in enumerate(['train', 'finetune', 'play'])})
GameEnum = Enum('Game', {k: i for i, k in enumerate(['atari', 'mujoco'])})
DeviceEnum = Enum('Device', {'cuda': 'cuda', 'cpu': 'cpu'})

@dataclass
class AlgoConfig:
    name: AlgoEnum = AlgoEnum.dqn
    discount: float = 0.99

@dataclass
class TrainerConfig:
    batch_size: int = 512
    learning_rate: float = 5e-4
    total_steps: int = int(1e7)
    
    training_start_steps: int = int(1e5)
    exploration_steps: int = int(1e6)
    target_update_freq: int = 500
    log_freq: int = 100
    device: DeviceEnum = DeviceEnum.cuda
    
    learner_steps: int = 20

@dataclass
class ActorConfig:
    policy: ActorEnum = ActorEnum.random
    num_envs: int = 16
    actor_steps: int = 80
    min_eps: float = 0.01
    test_eps: float = 0.001
    device: DeviceEnum = DeviceEnum.cuda
    
@dataclass
class ReplayConfig:
    size: int = int(1e6)
    policy: ReplayEnum = ReplayEnum.uniform

@dataclass
class ExpConfig:
    game: str = "Breakout"
    env: GameEnum = GameEnum.atari
    num_actors: int = 2
    seed: int = 42
    name: str = ""
    mode: ModeEnum = ModeEnum.train
    logdir: str = "output"

    algo: AlgoConfig = field(default_factory=AlgoConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)