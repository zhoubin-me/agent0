from dataclasses import dataclass


@dataclass
class Config:
    game: str = 'Breakout'
    double_q: bool = True
    dueling: bool = True
    noisy: bool = False
    distributional: bool = False
    qr: bool = False
    pin_memory = True

    adam_lr: float = 5e-4
    v_max: float = 10
    v_min: float = -10
    num_atoms: int = 1

    reset_noise_freq: int = 5
    exp_name: str = 'atari_deepq'

    num_actors: int = 8
    num_envs: int = 16
    num_data_workers: int = 4

    batch_size: int = 512
    replay_size: int = int(1e6)
    discount: float = 0.99
    exploration_ratio: float = 0.1
    min_eps: float = 0.01

    total_steps: int = int(2.5e7)
    epochs: int = 2500
    start_training_step: int = int(1e5)
    target_update_freq: int = 500
    agent_train_freq: int = 10

    restore_ckpt: str = None
    random_seed: int = 1234

    def default_num_atoms(self):
        if self.distributional:
            self.num_atoms = 51
        elif self.qr:
            self.num_atoms = 200
        else:
            self.num_atoms = 1

        return self.num_atoms
