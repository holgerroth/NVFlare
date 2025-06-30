from abc import ABC, abstractmethod
from typing import List, Optional, Dict

from src.cifar10_fl import Net

from nvflare.app_common.workflows.fedavg import FedAvg
from fl_utils import Server, Client, FLRunner, FedAvgStrategy, FLExperiment, SimEnv
from nvflare.app_common.trainers.pt_trainer import PyTorchTrainer



if __name__ == "__main__":
    # Example usage
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_fl.py"

    # Create trainer
    trainer = PyTorchTrainer(train_script)


    # Create strategy and setup FL
    strategy = FedAvgStrategy(trainer=trainer, num_rounds=num_rounds, min_clients=n_clients)

    
    # Define experiment
    exp = FLExperiment(strategy=strategy, 
                       n_clients=1000, config=None)

    # Run experiment
    exp.run(env=SimEnv(gpu="0", workdir="/tmp/nvflare/cifar10"))
    #exp.run(env=FlareEnv())
