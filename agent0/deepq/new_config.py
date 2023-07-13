from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import gymnasium as gym


class AlgoEnum(Enum):
    dqn = 0
    c51 = 1
    qr = 2
    iqn = 3
    fqf = 4


class ActorEnum(Enum):
    greedy = 0
    random = 1
    epsilon = 2


class ReplayEnum(Enum):
    uniform = 0
    prior = 1


class ModeEnum(Enum):
    train = 0
    finetune = 1
    play = 2


class EnvEnum(Enum):
    atari = 0
    mujoco = 1


class DeviceEnum(Enum):
    cuda = "cuda"
    cpu = "cpu"


@dataclass
class C51Config:
    num_atoms: int = 51
    vmax: float = 10
    vmin: float = -10


@dataclass
class QRConfig:
    num_atoms: int = 51
    vmax: Any = None
    vmin: Any = None


@dataclass
class IQNConfig:
    K: int = 32
    N: int = 64
    N_dash: int = 64
    num_cosines: int = 64
    F: int = 32


@dataclass
class LearnerConfig:
    algo: AlgoEnum = AlgoEnum.dqn

    discount: float = 0.99
    batch_size: int = 512
    learning_rate: float = 5e-4
    fraction_lr: float = 2.5e-8
    max_grad_norm: float = -1.0

    target_update_freq: int = 500
    learner_steps: int = 20

    double_q: bool = False
    dueling_head: bool = False
    noisy_net: bool = False
    prioritize_replay: bool = False
    n_step_q: int = 3

    c51: C51Config = field(default_factory=C51Config)
    qr: QRConfig = field(default=QRConfig)
    iqn: IQNConfig = field(default=IQNConfig)


@dataclass
class TrainerConfig:
    total_steps: int = int(1e7)
    training_start_steps: int = int(1e5)
    exploration_steps: int = int(1e6)
    log_freq: int = 100


@dataclass
class ActorConfig:
    policy: ActorEnum = ActorEnum.random
    num_envs: int = 16
    actor_steps: int = 80
    min_eps: float = 0.01
    test_eps: float = 0.001


@dataclass
class ReplayConfig:
    size: int = int(1e6)
    policy: ReplayEnum = ReplayEnum.uniform


@dataclass
class ExpConfig:
    env_id: str = "Breakout"
    env_type: EnvEnum = EnvEnum.atari
    obs_shape: Any = (0,)
    action_dim: int = 0
    num_actors: int = 2
    seed: int = 42
    device: DeviceEnum = DeviceEnum.cuda
    name: str = ""
    mode: ModeEnum = ModeEnum.train
    logdir: str = "tblog"

    learner: LearnerConfig = field(default_factory=LearnerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
