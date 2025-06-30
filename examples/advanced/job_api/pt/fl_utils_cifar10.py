from abc import ABC, abstractmethod
from typing import List, Optional, Dict

from src.cifar10_fl import Net
from fl_utils import FedAvgStrategy, FLExperiment, SimEnv
from nvflare.app_common.trainers.pt_trainer import PyTorchTrainer



if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_fl.py"

    # Create client trainer
    trainer = PyTorchTrainer(train_script)


    # Create FL strategy
    strategy = FedAvgStrategy(trainer=trainer, num_rounds=num_rounds, num_clients=n_clients, initial_model=Net())

    
    # Define experiment
    exp = FLExperiment(strategy=strategy, 
                       num_clients=n_clients, config=None)

    # Run experiment
    exp.run(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10"))
    #exp.run(env=FlareEnv())
