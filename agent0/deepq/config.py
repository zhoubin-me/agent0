from dataclasses import dataclass

from agent0.common.bench import atari7, atari11, atari47, atari_exp7, atari63


@dataclass
class Config:
    # algo setting
    game: str = ""
    double_q: bool = False
    dueling: bool = False
    noisy: bool = False
    prioritize: bool = False
    n_step: int = 1
    # support: dqn, mdqn, c51, qr, iqr, fqf, gmm
    algo: str = 'dqn'
    # support: epsilon_greedy, soft_explore
    policy: str = 'epsilon_greedy'

    # replace noisy layer to gmm layer. not working
    gmm_layer: bool = False

    # opitimizer related
    adam_lr: float = 5e-4
    fraction_lr: float = 2.5e-8

    # c51 related
    v_max: float = 10
    v_min: float = -10
    num_atoms: int = -1

    # prioritize related
    priority_alpha: float = 0.5
    priority_beta0: float = 0.4

    # mdqn related
    mdqn_tau: float = 0.03
    mdqn_alpha: float = 0.9
    mdqn_lo: float = -1

    # IQR related
    K_iqr: int = 32
    N_iqr: int = 64
    N_iqr_dash: int = 64
    N_fqf: int = 32
    num_cosines: int = 64
    clip_grad_norm: float = -1

    # gmm_related
    gmm_num_samples: int = 25000
    gmm_max_std: float = 3.0

    # others
    cor_loss: bool = False
    cor_loss2: bool = False
    cor_loss3: bool = False
    cor_N: int = 256
    cor_reg: float = 0.05
    best_ep: bool = False
    ep_best_len: int = 100000
    best_ep_reg: float = 0.01

    # basic training setting
    batch_size: int = 400
    replay_size: int = int(1e6)
    discount: float = 0.99
    exploration_steps: int = int(1e6)
    min_eps: float = 0.01
    test_eps: float = 0.001

    total_steps: int = int(1e7)
    start_training_step: int = int(1e5)

    target_update_freq: int = 500
    agent_train_steps: int = 20
    actor_steps: int = 80
    num_actors: int = 2
    num_envs: int = 16
    num_data_workers: int = 4
    reset_noise_freq: int = 4
    checkpoint_freq: int = 250
    max_record_ep_len: int = 100000
    test_episode_per_actor: int = 5

    pin_memory: bool = True
    fast_replay: bool = True
    restore_checkpoint: str = ""
    random_seed: int = 42
    trials: int = 1
    exp_name: str = 'atari_deepq'
    frame_stack: int = 4
    sha: str = ""
    gpu_mult: float = 1.0 / 4.0
    step_mult: int = 1
    num_samples: int = 1
    reversed: bool = False

    def update(self):
        self.actor_steps *= self.step_mult
        self.update_game()
        self.update_atoms()
        self.update_seeds()

    def update_seeds(self):
        if self.trials > 1:
            self.random_seed = list([self.random_seed+x for x in range(self.trials)])

    def update_game(self):
        if self.game == "":
            self.game = "Breakout"
        if self.game not in atari63:
            game_dict = {
                'atari7': atari7,
                'atari11': atari11,
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
            'fqf': 1,
            'gmm': 33,
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
