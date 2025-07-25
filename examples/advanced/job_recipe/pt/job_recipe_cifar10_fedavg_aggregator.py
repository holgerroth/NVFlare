from src.cifar10_client import Net
import torch
from typing import List

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.recipes.fedavg import FedAvgRecipe
from nvflare.environments.sim_environment import SimEnv
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator

# Option 1: use a custom aggregator function
def my_aggregate_func(models: List[FLModel]) -> FLModel:

    print(f"##### my_aggregator_func: Aggregating {len(models)} models #####")
    
    # Collect model params
    aggregated_params = {}
    for model in models:
        for key in model.params:
            if key not in aggregated_params:
                aggregated_params[key] = []
            aggregated_params[key].append(model.params[key])
    
    # compute the average
    for key in aggregated_params:
        aggregated_params[key] = torch.mean(aggregated_params[key])
    
    return FLModel(params=aggregated_params)

# Option 2: use custom ModelAggregator
class MyAggregator(ModelAggregator):
    def __init__(self):
        super().__init__()
        self.sum = {}    
        self.count = 0
    
    def accept_model(self, model: FLModel):
        # accept submitted model and add to the sum
        print(f"##### MyAggregator: Accepting model with {len(model.params)} variables #####")
        for key, value in model.params.items():
            if key not in self.sum:
                self.sum[key] = 0
            self.sum[key] += value
        self.count += 1

    def aggregate_model(self) -> FLModel:
        print(f"##### MyAggregator: Aggregating {self.count} models #####")

        # compute the average
        for key in self.sum:
            self.sum[key] = self.sum[key] / self.count
        
        return FLModel(params=self.sum)

    def reset_stats(self):
        print(f"##### MyAggregator: Resetting #####")
        # reset the sum and count
        self.sum = {}    
        self.count = 0



if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 3
    train_script = "src/cifar10_client.py"


    # Now, let's create an FL recipe, defining the training logic, number rounds, min_clients, for next round, etc.
    # We can also define our own aggregation function here
    # Option 1: Pass the function directly (will be automatically wrapped)
    recipe = FedAvgRecipe(
        min_clients=n_clients,
        num_rounds=num_rounds,
        train_script=train_script,
        train_args="--local_epochs 1 --batch_size 32",
        initial_model=Net(),
        aggregator=my_aggregate_func, #MyAggregator()  # Both Callable and Aggregator are now supported
    )
 
    # Use a the SimEnv to run the experiment locally.
    recipe.execute(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10_fedavg_myaggregator", clients=n_clients))
    # recipe.execute(env=FlareEnv())
