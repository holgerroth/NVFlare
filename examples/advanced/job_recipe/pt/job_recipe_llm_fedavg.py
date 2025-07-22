from src.llm_model import CausalLMModel

from nvflare.recipes.fedavg import FedAvgRecipe
from nvflare.environments.sim_environment import SimEnv

if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 3
    train_script = "src/llm_client.py"

    # Now, let's create an FL recipe, defining the training logic, number rounds, min_clients, for next round, etc.
    recipe = FedAvgRecipe(
        num_clients=1, #["dolly", "alpaca", "oasst1"],
        num_rounds=num_rounds,
        train_script=train_script,
        train_args=f"--local_epoch 1",
        initial_model=CausalLMModel(model_name_or_path="facebook/opt-125m"),
        external_client_process=True,
        server_expected_format="pytorch",
        client_command_prefix=f"accelerate launch --num_processes 2",  # would mean each client runs on 2 gpus.
        #quantization_type="float4",
    )

    # Use a the SimEnv to run the experiment locally.
    recipe.export(path="/tmp/nvflare/llm_fedavg_job")
    #"[0,1],[2,3],[0,1]"
    recipe.execute(env=SimEnv(gpu=0, workdir="/tmp/nvflare/llm_fedavg"))
    # recipe.execute(env=FlareEnv())
