from src.cifar10_fl import Net

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.recipes.fedavg import FedAvgRecipe
from nvflare.environments.sim_environment import SimEnv
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator

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
    train_script = "src/cifar10_fl.py"


    # Now, let's create an FL recipe, defining the training logic, number rounds, min_clients, for next round, etc.
    # We can also define our own aggregation function here
    recipe = FedAvgRecipe(
        num_clients=n_clients,
        num_rounds=num_rounds,
        train_script=train_script,
        train_args="--local_epochs 1 --batch_size 32",
        initial_model=Net(),
        aggregator=MyAggregator(),
    )

    # Use a the SimEnv to run the experiment locally.
    recipe.execute(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10_fedavg_myaggregator"))
    # recipe.execute(env=FlareEnv())
