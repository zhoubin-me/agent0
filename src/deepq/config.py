from dataclasses import dataclass

from src.common.bench import atari8, atari10, atari47, atari_exp7, atari63

GPU_SIZE = 1.0
assert GPU_SIZE > 0


@dataclass
class Config:
    game: str = ""
    double_q: bool = False
    dueling: bool = False
    noisy: bool = False
    n_step: int = 1
    distributional: bool = False
    qr: bool = False
    prioritize: bool = False

    adam_lr: float = 5e-4
    v_max: float = 10
    v_min: float = -10
    num_atoms: int = -1
    priority_alpha: float = 0.5
    priority_beta0: float = 0.4

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

    num_actors: int = 8
    num_envs: int = 16
    num_data_workers: int = 4
    reset_noise_freq: int = 4

    pin_memory = True
    fast_replay = True
    restore_checkpoint: str = None
    random_seed: int = 42
    exp_name: str = 'atari_deepq'
    frame_stack: int = 4
    sha: str = ""

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

        if self.game not in atari63:
            game_dict = {
                'atari8': atari8,
                'first': atari8[:4],
                'second': atari8[4:],
                'atari10': atari10,
                'atari47': atari47,
                'atari_exp7': atari_exp7,
                'atari63': atari63
            }

            try:
                self.game = game_dict[self.game.lower()]
            except Exception as e:
                print(e)
                raise ValueError(f"No such atari games as {self.game}\n"
                                 f"available games[list] are {game_dict} and:\n"
                                 f"{atari63}")

        self.epochs = self.total_steps // self.steps_per_epoch
        self.actor_steps = self.steps_per_epoch // (self.num_envs * self.num_actors) + 1

        assert not (self.distributional and self.qr)
        assert self.n_step >= 1
