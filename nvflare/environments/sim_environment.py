import os
from typing import Optional, Union

from nvflare.environments.environment import Env
from nvflare.job_config.api import FedJob


class SimEnv(Env):
    """Simulation environment for federated learning."""

    def __init__(self, gpu: Optional[str] = None, workdir: str = "/tmp/nvflare/workdir", clients: Union[int, list[str]] = None, log_config: str = None):
        self.gpu = gpu
        self.workdir = workdir
        self.clients = clients
        self.log_config = log_config

    def execute(self, job: FedJob):
        """Run the simulation."""

        if isinstance(self.clients, list):
            job.clients = self.clients
            job.simulator_run(self.workdir, gpu=self.gpu, log_config=self.log_config)
        else:
            job.simulator_run(self.workdir, gpu=self.gpu, n_clients=self.clients, log_config=self.log_config)
        
        print(f"Running simulation with GPU: {self.gpu}, output dir: {self.workdir}")
        
