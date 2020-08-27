import git
import ray
from ray import tune
from ray.tune import CLIReporter

from src.nips_encoder.trainer import Trainer, Config
from src.common.utils import parse_arguments


if __name__ == '__main__':
    repo = git.Repo(search_parent_directories=True)
    sha = repo.git.rev_parse(repo.head.object.hexsha, short=True)
    sha_long = repo.head.object.hexsha

    cfg = Config(sha=sha_long)
    args = parse_arguments(cfg)
    cfg = Config(**vars(args))

    ray.init(memory=20 * 2 ** 30, object_store_memory=80 * 2 ** 30)
    reporter = CLIReporter(
        metric_columns=["game", "speed", "loss", "adam_lr", "time_remain", "time_past"]
    )

    analysis = tune.run(
        Trainer,
        name='nips_encoder_tune',
        verbose=1,
        stop=lambda trial_id, result: result['epoch'] > cfg.epochs,
        checkpoint_at_end=True,
        progress_reporter=reporter,
        checkpoint_freq=1000,
        resources_per_trial={"gpu": 1},
        config=vars(cfg),
        fail_fast=True,
        reuse_actors=True,
        restore=cfg.restore_checkpoint,
    )
