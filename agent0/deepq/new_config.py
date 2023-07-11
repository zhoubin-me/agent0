from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Any
import gymnasium as gym

AlgoEnum = Enum("Algo", {k: i for i, k in enumerate(["dqn", "c51"])})
ActorEnum = Enum(
    "Actor", {k: i for i, k in enumerate(["greedy", "random", "eps-greedy"])}
)
ReplayEnum = Enum("Replay", {k: i for i, k in enumerate(["uniform", "prior"])})
ModeEnum = Enum("Mode", {k: i for i, k in enumerate(["train", "finetune", "play"])})
EnvEnum = Enum("Env", {k: i for i, k in enumerate(["atari", "mujoco"])})
DeviceEnum = Enum("Device", {"cuda": "cuda", "cpu": "cpu"})


@dataclass
class C51Config:
    atoms: int = 51
    vmax: float = 10
    vmin: float = -10

@dataclass
class LearnerConfig:
    algo: AlgoEnum = AlgoEnum.dqn
    discount: float = 0.99
    batch_size: int = 512
    learning_rate: float = 5e-4
    target_update_freq: int = 500
    learner_steps: int = 20
    double_q: bool = False
    dueling_head: bool = False

    c51: C51Config = field(default_factory=C51Config)


    
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
