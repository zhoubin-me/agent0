import hydra
import os
from dacite import from_dict
from hydra.core.config_store import ConfigStore
import git
import shortuuid

from agent0.common.atari_wrappers import make_atari
from agent0.common.utils import set_random_seed
from agent0.deepq.config import ExpConfig
from agent0.deepq.trainer import Trainer

@hydra.main(version_base=None, config_name="config")
def main(cfg: ExpConfig):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:8]
    uuid = shortuuid.uuid()[:8]
    subdir = f"{cfg.name}-{cfg.env_id}-{cfg.learner.algo}-{cfg.seed}-{sha}-{uuid}"
    dummy_env = make_atari(cfg.env_id, num_envs=1)
    dummy_env.close()

    cfg = from_dict(ExpConfig, cfg)

    cfg.logdir = os.path.join(cfg.logdir, subdir)
    cfg.obs_shape = dummy_env.observation_space.shape[1:]
    cfg.action_dim = int(dummy_env.action_space[0].n)

    set_random_seed(cfg.seed)
    Trainer(cfg).run()

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=ExpConfig)
    main()
