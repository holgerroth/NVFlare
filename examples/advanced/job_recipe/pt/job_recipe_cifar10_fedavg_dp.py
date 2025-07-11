from src.cifar10_fl import Net

from nvflare.app_common.recipes.fedavg_dp import FedAvgRecipeDP
from nvflare.environments.sim_environment import SimEnv

if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_fl.py"

    # Create FL recipe
    recipe = FedAvgRecipeDP(
        num_rounds=num_rounds,
        num_clients=n_clients,
        train_script=train_script,
        train_args="--local_epochs 1 --batch_size 32",
        initial_model=Net(),
        privacy_fraction=0.1, 
        privacy_epsilon=0.1, 
        privacy_noise_var=0.1, 
    )

    # Define experiment
    # Run experiment
    recipe.run(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10"))
    # exp.run(env=FlareEnv())
