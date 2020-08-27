from dataclasses import dataclass

from src.common.bench import atari6, atari10, atari47, atari_exp7, atari63
from src.common.gpuinfo import get_gpus

GPU_SIZE = max(1, min(get_gpus()[-1].total_memory // 10000, 2.0))
NUM_TASKS = 1.0


@dataclass
class Config:
    game: str = ""
    double_q: bool = False
    dueling: bool = False
    noisy: bool = False
    prioritize: bool = False
    n_step: int = 1
    algo: str = 'dqn'

    adam_lr: float = 5e-4
    v_max: float = 10
    v_min: float = -10
    num_atoms: int = -1
    priority_alpha: float = 0.5
    priority_beta0: float = 0.4
    mdqn_tau: float = 0.03
    mdqn_alpha: float = 0.9
    mdqn_lo: float = -1
    clip_grad_norm: float = 5.0

    batch_size: int = 512
    replay_size: int = int(1e6)
    discount: float = 0.99
    exploration_ratio: float = 0.1
    min_eps: float = 0.01

    total_steps: int = int(1e7)
    steps_per_epoch: int = 10000
    epochs: int = 2500
    start_training_step: int = int(1e5)
    target_update_freq: int = 500

    agent_train_steps: int = 10
    actor_steps: int = 40

    num_actors: int = 5
    num_envs: int = 16
    num_data_workers: int = 4
    reset_noise_freq: int = 4

    max_record_ep_len = 10000
    pin_memory = True
    fast_replay = True
    restore_checkpoint: str = ""
    random_seed: int = 42
    exp_name: str = 'atari_deepq'
    frame_stack: int = 4
    sha: str = ""

    def update_game(self):
        if self.game == "":
            self.game = "Breakout"
        if self.game.capitalize() not in atari63:
            game_dict = {
                'atari6': atari6,
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
                                 f"available games[list] are {game_dict.keys()} and:\n"
                                 f"{atari63}")
        else:
            self.game = self.game.capitalize()

    def update_atoms(self):
        algo_num_atoms = {
            'dqn': 1,
            'c51': 51,
            'qr': 200,
            'mdqn': 1,
            'kl': 1,
        }

        if self.num_atoms < 1:
            try:
                self.num_atoms = algo_num_atoms[self.algo]
            except Exception as e:
                print(e)
                raise ValueError(f"Algo {self.algo} not implemented\n"
                                 f"available algorithms are:\n"
                                 f"{algo_num_atoms.keys()}")
