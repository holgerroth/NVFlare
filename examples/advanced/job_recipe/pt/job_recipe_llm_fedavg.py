from src.llm_model import CausalLMModel

from nvflare.recipes.fedavg import FedAvgRecipe
from nvflare.environments.sim_environment import SimEnv

if __name__ == "__main__":
    # Example usage
    n_clients = 3
    num_rounds = 3
    train_script = "src/llm_client.py"
    model_name = "facebook/opt-125m"

    # Now, let's create an FL recipe, defining the training logic, number rounds, min_clients, for next round, etc.
    recipe = FedAvgRecipe(
        min_clients=n_clients,
        num_rounds=num_rounds,
        train_script=train_script,
        train_args=f"--local_epoch 1 --model_name_or_path {model_name}",
        initial_model=CausalLMModel(model_name_or_path=model_name),
        framework="pytorch", # "raw", "tensorflow"
        server_load_model_func=None,
        server_save_model_func=None,
        stop_condition=None,
    )

    # Use a the SimEnv to run the experiment locally.
    recipe.export(path="/tmp/nvflare/llm_fedavg_job")
    recipe.execute(env=SimEnv(clients=["dolly", "alpaca", "oasst1"], gpu="[0,1],[2,3],[0,1]", workdir="/tmp/nvflare/llm_fedavg"))
    # recipe.execute(env=FlareEnv())
