from dataclasses import dataclass

from src.common.bench import _atari, _atari8, _atari10, _atariexp7


@dataclass
class Config:
    game: str = ""
    double_q: bool = False
    dueling: bool = False
    noisy: bool = False
    n_step: int = 1
    distributional: bool = False
    qr: bool = False

    adam_lr: float = 5e-4
    v_max: float = 10
    v_min: float = -10
    num_atoms: int = -1

    num_actors: int = 8
    num_envs: int = 16
    num_data_workers: int = 4
    reset_noise_freq: int = 4

    batch_size: int = 512
    replay_size: int = int(1e6)
    discount: float = 0.99
    exploration_ratio: float = 0.1
    min_eps: float = 0.01

    total_steps: int = int(2.5e7)
    steps_per_epoch: int = 10000
    epochs: int = 2500
    start_training_step: int = int(1e5)
    target_update_freq: int = 500
    agent_train_steps: int = 10
    actor_steps: int = 40

    pin_memory = True
    restore_checkpoint: str = None
    random_seed: int = 42
    exp_name: str = 'atari_deepq'
    frame_stack: int = 4

    def update(self):
        if self.num_atoms < 0:
            if self.distributional:
                self.num_atoms = 51
            elif self.qr:
                self.num_atoms = 200
            else:
                self.num_atoms = 1

        if self.game == "":
            self.game = "Breakout"

        if self.game not in _atari:
            if self.game == 'first':
                self.game = _atari8[:4]
            elif self.game == 'second':
                self.game = _atari8[-4:]
            elif self.game == 'atari8':
                self.game = _atari8
            elif self.game == 'atari10':
                self.game = _atari10
            elif self.game == 'atariexp7':
                self.game = _atariexp7
            elif self.game == 'atari_full':
                self.game = _atari
            else:
                raise ValueError("No such atari games")

        self.epochs = self.total_steps // self.steps_per_epoch
        self.actor_steps = self.steps_per_epoch // (self.num_envs * self.num_actors)

        assert not (self.distributional and self.qr)
        assert self.n_step >= 1
