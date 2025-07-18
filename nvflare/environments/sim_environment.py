import os
from typing import Optional

from nvflare.environments.environment import Env
from nvflare.job_config.api import FedJob


class SimEnv(Env):
    """Simulation environment for federated learning."""

    def __init__(self, gpu: Optional[str] = None, workdir: str = "/tmp/nvflare/workdir"):
        self.gpu = gpu
        self.workdir = workdir

    def run(self, job: FedJob, n_clients: int = None):
        """Run the simulation."""
        
        print(f"Running simulation with GPU: {self.gpu}, output dir: {self.workdir}")
        job.simulator_run(self.workdir, gpu=self.gpu, n_clients=n_clients) #, log_config="full")
