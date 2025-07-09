from src.cifar10_fl import Net
import numpy as np
from typing import List

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.recipes.fedavg import FedAvgRecipe
from nvflare.environments.sim_environment import SimEnv

def my_aggregate_fn(results: List[FLModel]) -> FLModel:
    print(f"My aggregate function: averaging {len(results)} results")

    params = {}
    for result in results:
        for key, value in result.params.items():
            if key not in params:
                params[key] = []
            params[key].append(value)

    # average the params
    aggr_params = {}
    for key, value in params.items():
        aggr_params[key] = np.mean(value)

    print(f"Aggregated params: {len(aggr_params)}")

    return FLModel(params=aggr_params)


if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_fl.py"

    # Now, let's create an FL recipe, defining the training logic, number rounds, min_clients, for next round, etc.
    # We can also define our own aggregation function here
    recipe = FedAvgRecipe(
        num_clients=n_clients,
        num_rounds=num_rounds,
        train_script=train_script,
        train_args="--epochs 1 --batch_size 32",
        initial_model=Net(),
        aggregate_fn=my_aggregate_fn,
    )

    # Use a the SimEnv to run the experiment locally.
    recipe.run(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10", name="cifar10_fedavg"))
    # recipe.run(env=FlareEnv())
