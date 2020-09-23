import git
import ray
from agent0.common.utils import parse_arguments
from agent0.ddpg.config import Config
from agent0.ddpg.trainer import Trainer
from ray import tune
from ray.tune import CLIReporter


def trial_str_creator(trial, sha_short):
    return "{}_{}_{}_{}".format(trial.trainable_name, trial.config['game'], sha_short, trial.trial_id)


if __name__ == '__main__':
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.git.rev_parse(repo.head.object.hexsha, short=True)
        sha_long = repo.head.object.hexsha
    except Exception as e:
        sha = 'master'
        sha_long = 'master'

    cfg = Config(sha=sha_long)
    args = parse_arguments(cfg)
    cfg = Config(**vars(args))
    cfg.update_game()

    if isinstance(cfg.game, list):
        if cfg.reversed:
            cfg.game = tune.grid_search(cfg.game[::-1])
        else:
            cfg.game = tune.grid_search(cfg.game)

    ray.init(memory=10 * 2 ** 30, object_store_memory=20 * 2 ** 30, num_cpus=20)
    metric_columns = ["frames", "v_loss", "p_loss", "ep_reward_test", "ep_reward_train",
                      "ep_reward_train_max", "time_past", "time_remain", "speed", "velocity"]

    reporter = CLIReporter(metric_columns=metric_columns)
    tune.run(
        Trainer,
        name=cfg.exp_name,
        verbose=1,
        checkpoint_at_end=True,
        fail_fast=True,
        reuse_actors=True,
        restore=cfg.restore_checkpoint,
        stop=lambda trial_id, result: result['frames'] > cfg.total_steps,
        checkpoint_freq=cfg.ckpt_freq,
        trial_name_creator=tune.function(lambda trial: trial_str_creator(trial, sha)),
        progress_reporter=reporter,
        resources_per_trial={"gpu": 0.1},
        config=vars(cfg),
    )
