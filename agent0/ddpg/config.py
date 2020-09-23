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
    test_episodes: int = 20
    save_interval: int = 50000
    action_noise_level: float = 0.1
    policy_update_freq: int = 2

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

    def update(self):
        if self.game == "":
            self.game = "HalfCheetah"

        if self.game not in bullet:
            if self.game == 'mujoco7':
                self.game = mujoco7
            elif self.game == 'bullet15':
                self.game = bullet
            else:
                raise ValueError(f"No such atari games as {self.game}\n"
                                 f"available games[list] are [mujoco7, bullet15] and:\n"
                                 f"{bullet}")

        if self.algo == "":
            self.algo = "ddpg"

        if self.algo not in ['ddpg', 'sac', 'td3']:
            if self.algo == 'all':
                self.algo = ['ddpg', 'sac', 'td3']
            else:
                raise ValueError(f"No such algo as {self.algo}\n"
                                 f"available algos are [ddpg, sac, td3']")
