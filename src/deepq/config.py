from dataclasses import dataclass


@dataclass
class Config:
    game: str = 'Breakout'
    double_q: bool = True
    dueling: bool = True
    noisy: bool = False
    distributional: bool = False
    qr: bool = False
    nstep: int = 3

    adam_lr: float = 5e-4
    v_max: float = 10
    v_min: float = -10
    num_atoms: int = 1


    num_actors: int = 8
    num_envs: int = 16
    num_data_workers: int = 4
    reset_noise_freq: int = 5

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
    restore_ckpt: str = None
    random_seed: int = 42
    exp_name: str = 'atari_deepq'
    frame_stack: int = 4

    def update(self, num_atoms=None, game=None):
        if num_atoms is None:
            if self.distributional:
                self.num_atoms = 51
            elif self.qr:
                self.num_atoms = 200
            else:
                self.num_atoms = 1
        else:
            self.num_atoms = num_atoms

        if game is not None:
            self.game = game

        self.epochs = self.total_steps // self.steps_per_epoch
        self.actor_steps = self.steps_per_epoch // (self.num_envs * self.num_actors)
