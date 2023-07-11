import hydra
from hydra.core.config_store import ConfigStore

from agent0.deepq.new_config import ExpConfig
from agent0.deepq.new_trainer import Trainer
from agent0.common.atari_wrappers import make_atari
from omegaconf import OmegaConf
from dacite import from_dict

@hydra.main(version_base=None, config_name="config")
def main(cfg: ExpConfig):
    cfg = from_dict(ExpConfig, cfg)
    dummy_env = make_atari(cfg.env_id, num_envs=1)
    cfg.obs_shape = dummy_env.observation_space.shape[1:]
    cfg.action_dim = dummy_env.action_space[0].n
    dummy_env.close()
    Trainer(cfg).run()


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=ExpConfig)
    main()
