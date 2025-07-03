from src.cifar10_fl import Net

from nvflare.app_common.recipes.swarm import SwarmRecipe
from nvflare.app_common.trainers.script_trainer import ScriptTrainer
from nvflare.experiment import SimEnv

if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_fl.py"

    # Create client trainer. We use the ScriptTrainer that runs a script using the client API.
    trainer = ScriptTrainer(train_script, train_args)

    # Next, create an FL recipe, devining the training logic, number rounds, min_clients, for next round, etc.
    # We can also define our own aggregation function here
    recipe = SwarmRecipe(
        min_clients=n_clients,
        num_rounds=num_rounds,
        trainer=trainer,
        initial_model=Net(),
    )

    # Use a the SimEnv to run the experiment locally.
    recipe.run(n_clients=n_clients, env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10"))
    # recipe.run(env=FlareEnv())
