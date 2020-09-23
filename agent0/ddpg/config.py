from dataclasses import dataclass


@dataclass
class Config:
    game: str = "HalfCheetah"
    algo: str = "DDPG"
    seed: int = 0
    sha: str = ""

    max_steps = int(1e6)
    exploration_steps = 25000
    log_interval = 5000
    eval_episodes = 10
    save_interval = int(1e5)
    action_noise_level = 0.1

    # Replay related
    buffer_size = int(1e6)
    batch_size = 256

    # Optimizer related
    optimizer = 'adam'
    gamma = 0.99
    p_lr = 3e-4
    v_lr = 3e-4
    tau = 0.005

    # Others
    hidden_size = 256
    ckpt = ""
    log_dir = ""
    play = False

    def update(self):
        pass
