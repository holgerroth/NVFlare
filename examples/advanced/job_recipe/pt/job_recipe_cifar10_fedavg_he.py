from examples.advanced.job_recipe.pt.src.cifar10_client import Net

from nvflare.recipes.fedavg import FedAvgRecipe, HEPrivacyPolicy, PercentilePrivacyPolicy
from nvflare.environments.sim_environment import SimEnv 

if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_client.py"

    # Example 1: Basic HE privacy policy (encrypt all layers)
    he_policy_basic = HEPrivacyPolicy(
        tenseal_context_file="client_context.tenseal",
        encrypt_layers=None,  # Encrypt all layers
        weigh_by_local_iter=True
    )

    # Example 2: HE privacy policy with layer-specific encryption
    he_policy_layers = HEPrivacyPolicy(
        tenseal_context_file="client_context.tenseal",
        encrypt_layers=["conv", "fc"],  # Only encrypt layers with "conv" or "fc" in the name
        weigh_by_local_iter=True
    )

    # Example 3: HE privacy policy with regex pattern for layer selection
    he_policy_regex = HEPrivacyPolicy(
        tenseal_context_file="client_context.tenseal",
        encrypt_layers="conv.*",  # Encrypt all layers matching regex pattern "conv.*"
        weigh_by_local_iter=True
    )

    # Example 4: HE privacy policy with aggregation weights
    he_policy_weights = HEPrivacyPolicy(
        tenseal_context_file="client_context.tenseal",
        encrypt_layers=None,
        aggregation_weights={"site-1": 1.0, "site-2": 2.0},  # Different weights for different clients
        weigh_by_local_iter=True
    )

    # Example 5: Combining HE with other privacy policies
    he_policy = HEPrivacyPolicy(
        tenseal_context_file="client_context.tenseal",
        encrypt_layers=None,
        weigh_by_local_iter=True
    )
    
    percentile_policy = PercentilePrivacyPolicy(
        percentile=10,
        gamma=0.01
    )

    # Create FL recipe with HE privacy policy (using basic configuration)
    recipe = FedAvgRecipe(
        num_rounds=num_rounds,
        num_clients=n_clients,
        train_script=train_script,
        train_args="--local_epochs 1 --batch_size 32",
        initial_model=Net(),
        privacy_policies=[he_policy_basic]  # Use the HE privacy policy
    )

    # Alternative: Use multiple privacy policies
    # recipe = FedAvgRecipe(
    #     num_rounds=num_rounds,
    #     num_clients=n_clients,
    #     train_script=train_script,
    #     train_args="--local_epochs 1 --batch_size 32",
    #     initial_model=Net(),
    #     privacy_policies=[he_policy, percentile_policy]  # Combine HE with percentile privacy
    # )

    # Define experiment
    # Run experiment
    recipe.execute(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10_fedavg_he"))
    # exp.execute(env=FlareEnv())
