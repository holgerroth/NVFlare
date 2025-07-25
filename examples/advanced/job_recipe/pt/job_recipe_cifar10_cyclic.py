from examples.advanced.job_recipe.pt.src.cifar10_client import Net

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.recipes.cyclic import CyclicRecipe
from nvflare.environments.sim_environment import SimEnv

if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 3
    train_script = "src/cifar10_client.py"

    # Next, create an FL recipe, devining the training logic, number rounds, min_clients, for next round, etc.
    # We can also define our own aggregation function here

    aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)

    recipe = CyclicRecipe(
        num_clients=n_clients,
        num_rounds=num_rounds,
        train_script=train_script,
        train_args="--local_epochs 1 --batch_size 32",
        initial_model=Net()
    )

    # Use a the SimEnv to run the experiment locally.
    recipe.execute(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10_cyclic", n_clients=n_clients))
    # recipe.execute(env=FlareEnv())
