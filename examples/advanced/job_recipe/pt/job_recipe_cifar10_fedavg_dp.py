from src.cifar10_client import Net

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
            fraction=0.1, 
            epsilon=0.1, 
            noise_var=0.1, 
            gamma=1e-5, 
            tau=1e-6, 
            replace=True,
            percentile=10,
            percentile_gamma=1e-5
        )
    )

    # Define experiment
    # Run experiment
    recipe.execute(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10_fedavg_dp"))
    # exp.execute(env=FlareEnv())
