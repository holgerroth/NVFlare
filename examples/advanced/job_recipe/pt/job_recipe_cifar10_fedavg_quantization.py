from src.cifar10_client import Net
import torch

from nvflare.recipes.fedavg import FedAvgRecipe
from nvflare.environments.sim_environment import SimEnv
from nvflare.app_common.abstract.fl_model import FLModel


def my_server_load_model_func() -> FLModel:
    net = Net()
    net.load_state_dict(torch.load("FL_global_model.pt"))

    return FLModel(params=net.state_dict())

def my_server_save_model_func(model: FLModel):
    torch.save(model.params, "FL_global_model.pt")


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
    )

    # Add quantization client side
    recipe.add_client_input_filter(ModelDequantizer())
    recipe.add_client_output_filter(ModelQuantizer(quantization_type=quantization_type))

    # Add quantization server side
    recipe.add_server_input_filter(ModelDequantizer())
    recipe.add_server_output_filter(ModelQuantizer(quantization_type=quantization_type))    

    # Use a the SimEnv to run the experiment locally.
    recipe.export(path="/tmp/nvflare/cifar10_fedavg_model_loading_job")
    recipe.execute(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10_fedavg_model_loading"))
    # recipe.execute(env=FlareEnv())
