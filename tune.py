from ray import tune
from src.agents.deepq_agent import run
from ray import tune

from src.agents.deepq_agent import run

if __name__ == '__main__':
    analysis = tune.run(
        run,
        config={
            "alpha": tune.grid_search([0.001, 0.01, 0.1]),
            "beta": tune.choice([1, 2, 3])
        })

    print("Best config: ", analysis.get_best_config(metric="final_test_rewards"))
    df = analysis.dataframe()
    df.to_csv('out.csv')
