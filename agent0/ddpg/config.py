from dataclasses import dataclass

from agent0.common.bench import bullet, mujoco7


@dataclass
class Config:
    game: str = "HalfCheetah"
    algo: str = "DDPG"
    seed: int = 0
    sha: str = ""
    exp_name: str = "mujoco_ddpg"
    restore_checkpoint: str = ""
    ckpt_freq: int = 50000

    total_steps: int = int(1e6)
    exploration_steps: int = 25000
    eval_episodes: int = 20
    save_interval: int = 50000
    action_noise_level: float = 0.1

    # Replay related
    buffer_size: int = int(1e6)
    batch_size: int = 256

    # Optimizer related
    optimizer: str = 'adam'
    gamma: float = 0.99
    p_lr: float = 3e-4
    v_lr: float = 3e-4
    tau: float = 0.005

    # Others
    hidden_size: int = 256
    reversed: bool = False

    def update_game(self):
        if self.game == "":
            self.game = "Reacher"
        if self.game not in bullet:
            game_dict = {
                'mujoco7': mujoco7,
                'bullet15': bullet
            }

            try:
                self.game = game_dict[self.game]
            except Exception as e:
                print(e)
                raise ValueError(f"No such atari games as {self.game}\n"
                                 f"available games[list] are {game_dict.keys()} and:\n"
                                 f"{bullet}")
