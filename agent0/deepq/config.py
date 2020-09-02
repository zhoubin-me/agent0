from dataclasses import dataclass

from agent0.common.bench import atari6, atari10, atari47, atari_exp7, atari63


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
    fraction_lr: float = 2.5e-8
    v_max: float = 10
    v_min: float = -10
    num_atoms: int = -1
    priority_alpha: float = 0.5
    priority_beta0: float = 0.4
    mdqn_tau: float = 0.03
    mdqn_alpha: float = 0.9
    mdqn_lo: float = -1
    K_iqr: int = 32
    N_iqr: int = 64
    N_iqr_dash: int = 64
    N_fqf: int = 32
    num_cosines: int = 64
    clip_grad_norm: float = 5.0

    batch_size: int = 512
    replay_size: int = int(1e6)
    discount: float = 0.99
    exploration_steps: int = int(1e6)
    min_eps: float = 0.01
    test_eps: float = 0.001

    total_steps: int = int(1e7)
    steps_per_epoch: int = 10000
    epochs: int = 1000
    start_training_step: int = int(1e5)
    target_update_freq: int = 500

    agent_train_steps: int = 10
    actor_steps: int = 80

    num_actors: int = 5
    num_envs: int = 16
    num_data_workers: int = 4
    reset_noise_freq: int = 4

    max_record_ep_len = 100000
    test_episode_per_actor = 20
    pin_memory = True
    fast_replay = True
    restore_checkpoint: str = ""
    random_seed: int = 42
    exp_name: str = 'atari_deepq'
    frame_stack: int = 4
    sha: str = ""
    mem_mult: float = 2.0
    gpu_mult: float = 0.5
    step_mult: int = 1

    def update(self):
        self.actor_steps *= self.step_mult
        self.update_game()
        self.update_atoms()

    def update_game(self):
        if self.game == "":
            self.game = "Breakout"
        if self.game not in atari63:
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

    def update_atoms(self):
        algo_num_atoms = {
            'dqn': 1,
            'c51': 51,
            'qr': 200,
            'mdqn': 1,
            'iqr': 1,
        }

        if self.num_atoms < 1:
            try:
                if self.algo != 'all':
                    self.num_atoms = algo_num_atoms[self.algo]
            except Exception as e:
                print(e)
                raise ValueError(f"Algo {self.algo} not implemented\n"
                                 f"available algorithms are:\n"
                                 f"{algo_num_atoms.keys()}")
