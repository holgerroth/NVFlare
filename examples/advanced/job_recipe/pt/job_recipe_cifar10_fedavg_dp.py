from examples.advanced.job_recipe.pt.src.cifar10_client import Net

from nvflare.recipes.fedavg import FedAvgRecipe
from nvflare.app_common.filters.svt_privacy import SVTPrivacy, PercentilePrivacy
from nvflare.environments.sim_environment import SimEnv
from nvflare.app_common.filters.percentile_privacy import PercentilePrivacy

class CustomPrivacy(ModelFilter):  # TODO: implement ModelFilter base class
    def __init__(self, percentile: int = 15, gamma: float = 0.02):
        self.percentile = percentile
        self.gamma = gamma
    
    def filter_model(self, model: FLModel) -> FLModel:
        # TODO: implement custom privacy policy
        print(f"##### CustomPrivacy: Filtering model with percentile {self.percentile} and gamma {self.gamma} #####")

        return model
    

# Example of creating a custom privacy policy
class CustomPrivacyPolicy(PrivacyPolicy):
    """Example of a custom privacy policy that combines multiple filters."""
    
    def __init__(self, percentile: int = 15, gamma: float = 0.02):
        self.percentile = percentile
        self.gamma = gamma
    
    def create_filter(self):
        # You can create any DXOFilter here
        return PercentilePrivacy(
            percentile=self.percentile,
            gamma=self.gamma
        )

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
        initial_model=Net()
    )

    # Add privacy filters on the client's output
    recipe.add_client_output_filter(SVTPrivacy(fraction=0.1, epsilon=0.1, noise_var=0.1, gamma=1e-5, tau=1e-6, replace=True))
    recipe.add_client_output_filter(PercentilePrivacy(percentile=10, gamma=0.01))
    # recipe.add_client_output_filter(CustomPrivacy(percentile=15, gamma=0.02))

    # Define experiment
    # Run experiment
    recipe.execute(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10_fedavg_dp", n_clients=n_clients))
    # exp.execute(env=FlareEnv())
