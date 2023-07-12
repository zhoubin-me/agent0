import hydra
from dacite import from_dict
from hydra.core.config_store import ConfigStore

from agent0.common.atari_wrappers import make_atari
from agent0.common.utils import set_random_seed
from agent0.deepq.new_config import ExpConfig
from agent0.deepq.new_trainer import Trainer


@hydra.main(version_base=None, config_name="config")
def main(cfg: ExpConfig):
    print(cfg)
    cfg = from_dict(ExpConfig, cfg)
    dummy_env = make_atari(cfg.env_id, num_envs=1)
    cfg.obs_shape = dummy_env.observation_space.shape[1:]
    cfg.action_dim = dummy_env.action_space[0].n
    dummy_env.close()
    set_random_seed(cfg.seed)
    Trainer(cfg).run()


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=ExpConfig)
    main()
