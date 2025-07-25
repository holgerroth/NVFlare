from src.llm_model import CausalLMModel

from nvflare.recipes.fedavg import FedAvgRecipe
from nvflare.environments.sim_environment import SimEnv
from nvflare.app_opt.pt.quantization.dequantizer import ModelDequantizer
from nvflare.app_opt.pt.quantization.quantizer import ModelQuantizer

if __name__ == "__main__":
    # Example usage
    n_clients = 3
    num_rounds = 3
    train_script = "src/llm_client.py"
    model_name = "facebook/opt-125m"
    quantization_type = "float4"

    # Now, let's create an FL recipe, defining the training logic, number rounds, min_clients, for next round, etc.
    recipe = FedAvgRecipe(
        min_clients=n_clients,
        num_rounds=num_rounds,
        train_script=train_script,
        train_args=f"--local_epoch 1 --model_name_or_path {model_name}",
        initial_model=CausalLMModel(model_name_or_path=model_name),
        framework="pytorch", # "raw", "tensorflow"
    )

    # Add quantization client side
    recipe.add_client_output_filter(ModelQuantizer(quantization_type=quantization_type))
    recipe.add_client_input_filter(ModelDequantizer())
    
    # Add quantization server side
    recipe.add_server_output_filter(ModelQuantizer(quantization_type=quantization_type))    
    recipe.add_server_input_filter(ModelDequantizer())

    # Use a the SimEnv to run the experiment locally.
    recipe.export(path="/tmp/nvflare/llm_fedavg_job")
    recipe.execute(env=SimEnv(clients=["dolly", "alpaca", "oasst1"], gpu="[0,1],[2,3],[0,1]", workdir="/tmp/nvflare/llm_fedavg"))  # TODO: detect multi-gpu training of clients and start external process
    # recipe.execute(env=FlareEnv())
