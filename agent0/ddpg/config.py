from dataclasses import dataclass

from agent0.common.bench import bullet, mujoco7


@dataclass
class Config:
    game: str = ""
    algo: str = "ddpg"
    sha: str = 'master'
    exp_name: str = 'mujoco_ddpg'

    # Training related
    num_envs = 1
    actor_steps = 1
    test_episodes = 20
    ckpt_freq = int(1e5)

    total_steps = int(1e6)
    exploration_steps = 25000
    action_noise_level = 0.1

    # Replay related
    replay_size = int(1e6)
    batch_size = 256

    # Optimizer related
    optimizer = 'adam'
    gamma = 0.99
    p_lr = 3e-4
    v_lr = 3e-4
    tau = 0.005

    # Others
    restore_checkpoint: str = ""
    play = False
    random_seed = 42
    num_data_workers = 4
    pin_memory = True
    reversed: bool = False

    def update(self):
        self.update_game()

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
