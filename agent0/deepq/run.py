import git
import ray
from ray import tune
from ray.tune import CLIReporter

from agent0.common.utils import parse_arguments
from agent0.deepq.config import Config
from agent0.deepq.trainer import Trainer


def trial_str_creator(trial, sha_short):
    return "{}_{}_{}_{}".format(
        trial.trainable_name, trial.config["game"], sha_short, trial.trial_id
    )


if __name__ == "__main__":
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.git.rev_parse(repo.head.object.hexsha, short=True)
        sha_long = repo.head.object.hexsha
    except Exception as e:
        sha = "master"
        sha_long = "master"

    cfg = Config(sha=sha_long)
    args = parse_arguments(cfg)
    cfg = Config(**vars(args))
    cfg.update()

    if cfg.algo == "all":
        cfg.algo = tune.grid_search(["dqn", "c51", "qr", "iqr"])

    if isinstance(cfg.game, list):
        if cfg.reversed:
            cfg.game = tune.grid_search(cfg.game[::-1])
        else:
            cfg.game = tune.grid_search(cfg.game)

    if isinstance(cfg.random_seed, list):
        cfg.random_seed = tune.grid_search(cfg.random_seed)

    ray.init()
    metric_columns = [
        "frames",
        "loss",
        "ep_reward_test",
        "ep_reward_train",
        "ep_reward_train_max",
        "time_past",
        "time_remain",
        "speed",
        "velocity",
        "epsilon",
        "qmax",
    ]
    if cfg.algo in ["fqf"]:
        metric_columns.append("fraction_loss")
    if cfg.best_ep:
        metric_columns.append("ce_loss")
    reporter = CLIReporter(metric_columns=metric_columns)

    tune.run(
        Trainer,
        name=cfg.exp_name,
        verbose=1,
        checkpoint_at_end=True,
        num_samples=cfg.num_samples,
        fail_fast=True,
        local_dir="/home/bzhou/ssd/ray_results",
        restore=cfg.restore_checkpoint,
        stop=lambda trial_id, result: result["frames"] > cfg.total_steps,
        checkpoint_freq=cfg.checkpoint_freq,
        trial_name_creator=lambda trial: trial_str_creator(trial, sha),
        progress_reporter=reporter,
        resources_per_trial={"gpu": 1, "cpu": 16},
        config=vars(cfg),
    )
