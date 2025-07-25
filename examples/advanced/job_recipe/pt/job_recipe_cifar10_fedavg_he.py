from examples.advanced.job_recipe.pt.src.cifar10_client import Net

from nvflare.recipes.fedavg import FedAvgRecipe, HEPrivacyPolicy, PercentilePrivacyPolicy
from nvflare.environments.sim_environment import SimEnv 

if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_client.py"



    # Create FL recipe with HE privacy policy (using basic configuration)
    recipe = FedAvgRecipe(
        num_rounds=num_rounds,
        min_clients=n_clients,
        train_script=train_script,
        train_args="--local_epochs 1 --batch_size 32",
        initial_model=Net(),
    )

    recipe.add_homomorphic_encryption_policy(poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40], scale_bits=40, scheme="CKKS")

    # Define experiment
    # Run experiment
    recipe.execute(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10_fedavg_he", n_clients=n_clients))
    # exp.execute(env=FlareEnv())
