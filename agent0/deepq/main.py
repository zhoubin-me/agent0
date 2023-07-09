from agent0.deepq.new_trainer import Trainer
from agent0.deepq.config import Config

if __name__ == "__main__":
    cfg = Config()
    cfg.update()
    print(cfg)
    trainer = Trainer()
    trainer.setup(vars(cfg))
    trainer.run()