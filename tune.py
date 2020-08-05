import ray
from ray import tune

from src.agents.deepq_agent import run

if __name__ == '__main__':
    ray.init(memory=200 * 1024 * 1024 * 1024, object_store_memory= 100 * 1024 * 1024 * 1024)
    analysis = tune.run(
        run,
        config={
            "adam_lr": tune.grid_search([1e-3, 1e-4, 2e-4]),
            "target_update_freq": tune.grid_search([500, 200]),
            "agent_train_freq": tune.grid_search([20, 16]),
            "game": tune.grid_search(["Breakout"])
        },
        resources_per_trial={"gpu": 4},
    )
    print("Best config: ", analysis.get_best_config(metric="final_test_rewards"))
    df = analysis.dataframe()
    df.to_csv('out.csv')
