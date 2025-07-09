from src.cifar10_fl import Net

from nvflare.app_common.recipes.fedavg import FedAvgRecipe
from nvflare.environments.sim_environment import SimEnv

if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 10
    train_script = "src/cifar10_fl.py"

    # Now, let's create an FL recipe, defining the training logic, number rounds, min_clients, for next round, etc.
    # We can also define our own aggregation function here
    recipe = FedAvgRecipe(
        num_clients=n_clients,
        num_rounds=num_rounds,
        train_script=train_script,
        train_args="--epochs 1",
        initial_model=Net(),
        aggregate_fn=None,  # optional, if not provided, the default FedAvg aggr_func will be used
        sample_clients_fn=None,  # optional, if not provided, the default FedAvg sample_client_func will be used
        load_model_fn=None,  # optional, if not provided, the default FedAvg load_model_func will be used
        save_model_fn=None,  # optional, if not provided, the default FedAvg save_model_func will be used
        early_stop_fn=None,  # optional, if not provided, the default FedAvg early_stop_func will be used
    )

    # Use a the SimEnv to run the experiment locally.
    recipe.run(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10", name="cifar10_fedavg"))
    # recipe.run(env=FlareEnv())
