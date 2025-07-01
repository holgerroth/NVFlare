from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from nvflare.app_common.strategies.strategy import Strategy
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob


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
