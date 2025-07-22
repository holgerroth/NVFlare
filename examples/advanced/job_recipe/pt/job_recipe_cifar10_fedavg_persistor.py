from examples.advanced.job_recipe.pt.src.cifar10_client import Net

from nvflare.recipes.fedavg import FedAvgRecipe
from nvflare.environments.sim_environment import SimEnv
from nvflare.app_opt.pt import PTFileModelPersistor

class MyPersistor(PTFileModelPersistor):
    pass # implement your own persistor here


if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 3
    train_script = "src/cifar10_client.py"


    # Now, let's create an FL recipe, defining the training logic, number rounds, min_clients, for next round, etc.
    # We can also define our own aggregation function here
    recipe = FedAvgRecipe(
        num_clients=n_clients,
        num_rounds=num_rounds,
        train_script=train_script,
        train_args="--local_epochs 1 --batch_size 32",
        persistor=MyPersistor(model=Net()),
    )

    # Use a the SimEnv to run the experiment locally.
    recipe.export(path="/tmp/nvflare/cifar10_fedavg_persistor_job")
    # recipe.execute(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10_fedavg_persistor"))
    # recipe.execute(env=FlareEnv())
