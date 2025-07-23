import os
from typing import Optional

from nvflare.environments.environment import Env
from nvflare.job_config.api import FedJob


class SimEnv(Env):
    """Simulation environment for federated learning."""

    def __init__(self, gpu: Optional[str] = None, workdir: str = "/tmp/nvflare/workdir", n_clients: int = None, clients: list[str] = None, log_config: str = "full"):
        self.gpu = gpu
        self.workdir = workdir
        self.n_clients = n_clients
        self.clients = clients
        self.log_config = log_config

    def execute(self, job: FedJob):
        """Run the simulation."""

        if self.clients is not None:
            job.clients = self.clients
            job.simulator_run(self.workdir, gpu=self.gpu, log_config=self.log_config)
        else:
            job.simulator_run(self.workdir, gpu=self.gpu, n_clients=self.n_clients, log_config=self.log_config)
        
        print(f"Running simulation with GPU: {self.gpu}, output dir: {self.workdir}")
        
