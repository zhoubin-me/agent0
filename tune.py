from ray import tune

from src.agents.deepq_agent import run

if __name__ == '__main__':
    analysis = tune.run(
        run,
        config={
            "adam_lr": tune.grid_search([1e-3, 1e-4, 2e-4]),
            "target_update_freq": tune.grid_search([500, 200]),
            "agent_train_freq": tune.grid_search([20, 16]),
            "game": tune.grid_search(["Breakout"])
        })

    print("Best config: ", analysis.get_best_config(metric="final_test_rewards"))
    df = analysis.dataframe()
    df.to_csv('out.csv')
