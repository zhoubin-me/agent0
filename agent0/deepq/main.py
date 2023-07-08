import hydra
from new_config import ExpConfig
from hydra.core.config_store import ConfigStore
from trainer import Trainer


@hydra.main(version_base=None, config_name="config")
def main(cfg: ExpConfig):
    Trainer(cfg).run()

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=ExpConfig)
    main()