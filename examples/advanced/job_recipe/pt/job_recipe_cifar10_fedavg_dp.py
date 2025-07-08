from src.cifar10_fl import Net

from nvflare.app_common.filters.percentile_privacy import PercentilePrivacy
from nvflare.app_common.recipes.fedavg import FedAvgRecipe
from nvflare.environments.sim_environment import SimEnv

if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_fl.py"

    # Create FL recipe
    recipe = FedAvgRecipe(
        num_rounds=num_rounds,
        num_clients=n_clients,
        initial_model=Net(),
        aggregate_fn=None,  # optional
        privacy_filters=[PercentilePrivacy(percentile=10, gamma=0.01)],  # optional
    )

    # Define experiment
    # Run experiment
    recipe.run(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10"))
    # exp.run(env=FlareEnv())
