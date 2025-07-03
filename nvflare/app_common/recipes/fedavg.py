from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from nvflare import FilterType
from nvflare.app_common.trainers.pt_trainer import Trainer
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob

from .strategy import Client, Server, Strategy


class FedAvgRecipe(Recipe):
    """Federated Averaging Strategy.

    This strategy implements FedAvg with configurable
    number of clients and training rounds.
    """

    def __init__(
        self, trainer: Trainer, num_clients: int, num_rounds: int, initial_model=None, aggregate_fn=None, filters=None
    ):
        """Setup FedAvg configuration.

        Args:
            num_rounds: Number of training rounds
            num_clients: Number of clients to participate in FedAvg algorithm
            initial_model: Initial model to start training with
            aggregate_fn: Function to aggregate the models from clients
            filters: List of filters to apply to the strategy

        Returns:
        """
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.aggregate_fn = aggregate_fn
        self.filters = filters
        super().__init__(trainer)

    def setup(self): -> FedJob
        # Create the FedAvg controller
        controller = FedAvg(num_clients=self.num_clients, num_rounds=self.num_rounds)
        if self.aggregate_fn is not None:
            controller.aggregate_fn = self.aggregate_fn  # TODO: this won't work with job api

        # Create server with controller
        self.server = Server(controller, self.initial_model)

        # Create a single client template
        self.client = Client(self.trainer.get_executor())

        for filter in self.filters:
            print(f"Adding client filter: {type(filter)}")
            self.client.add_filter(filter, FilterType.TASK_RESULT, ["train"])

        return job

    def run(self, env: "Env"):
        self.setup()
        job = create_job(self.server, self.client)
        env.run(job)

class FLExperiment:
    def __init__(self, strategy: Strategy, num_clients: int, config: Optional[Dict] = None):
        self.strategy = strategy
        self.num_clients = num_clients
        self.config = config

    def run(self, env: "Env"):

        self.strategy.setup()
        job = create_job(self.strategy.get_server(), self.strategy.get_clients(n_clients=self.num_clients))

        # Create and return FLRunner
        env.run(job)


def create_job(server, clients, job_name="fed_sim_job"):
    """
    Simplified API to run a federated learning simulation.
    Args:
        server: Server object.
        clients: List of Client objects.
    """
    job = BaseFedJob(
        name=job_name,
        initial_model=server.initial_model,
    )
    for controller in server.controllers:
        job.to_server(controller)

    # Add clients to the job
    for i, client in enumerate(clients):
        for executor in client.executors:
            job.to(executor, f"site-{i}")

    return job


class Env(ABC):
    """Abstract base class for execution environments."""

    @abstractmethod
    def run(self):
        """Run the federated learning experiment."""
        raise NotImplementedError("Not implemented")


class SimEnv(Env):
    """Simulation environment for federated learning."""

    def __init__(self, gpu: Optional[str] = None, workdir: str = "/tmp/nvflare"):
        self.gpu = gpu
        self.workdir = workdir

    def run(self, job):
        """Run the simulation."""
        # Implementation would go here
        print(f"Running simulation with GPU: {self.gpu}, workdir: {self.workdir}")
        job.simulator_run(self.workdir, gpu=self.gpu)


class FlareEnv(Env):
    """NVFlare environment for federated learning."""

    def run(self):
        """Run the NVFlare experiment."""
        # Implementation would go here
        print("Running NVFlare experiment")
