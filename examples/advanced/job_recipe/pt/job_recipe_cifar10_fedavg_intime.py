from src.cifar10_fl import Net

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.recipes.fedavg_intime import InTimeFedAvgRecipe
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
        for key, value in self.sum.items():
            self.sum[key] = value / self.count

        # reset the sum and count
        self.sum = {}    
        self.count = 0
        
        return FLModel(params=self.sum)



if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 10
    train_script = "src/cifar10_fl.py"


    # Now, let's create an FL recipe, defining the training logic, number rounds, min_clients, for next round, etc.
    # We can also define our own aggregation function here
    recipe = InTimeFedAvgRecipe(
        num_clients=n_clients,
        num_rounds=num_rounds,
        train_script=train_script,
        train_args="--epochs 1 --batch_size 32",
        initial_model=Net(),
        aggregator=MyAggregator(),
    )

    # Use a the SimEnv to run the experiment locally.
    recipe.run(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10", name="cifar10_fedavg_intime"))
    # recipe.run(env=FlareEnv())
