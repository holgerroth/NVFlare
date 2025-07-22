from examples.advanced.job_recipe.pt.src.cifar10_client import Net

from nvflare.recipes.fedavg import FedAvgRecipe, PrivacyConfig
from nvflare.environments.sim_environment import SimEnv

if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_client.py"

    # Create FL recipe
    recipe = FedAvgRecipe(
        num_rounds=num_rounds,
        num_clients=n_clients,
        train_script=train_script,
        train_args="--local_epochs 1 --batch_size 32",
        initial_model=Net(),
        privacy_config=PrivacyConfig(
            epsilon=0.1, 
        )
    )

    # Define experiment
    # Run experiment
    recipe.execute(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10_fedavg_dp"))
    # exp.execute(env=FlareEnv())
