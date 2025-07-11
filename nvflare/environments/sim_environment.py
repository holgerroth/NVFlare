import os
from typing import Optional

from nvflare.environments.environment import Env
from nvflare.job_config.api import FedJob


class SimEnv(Env):
    """Simulation environment for federated learning."""

    def __init__(self, gpu: Optional[str] = None, workdir: str = "/tmp/nvflare", name: str = "nvflare"):
        self.gpu = gpu
        self.workdir = workdir
        self.name = name

    def run(self, job: FedJob, n_clients: int = None):
        """Run the simulation."""
        job.name = self.name
        workdir = os.path.join(self.workdir, self.name)
        # Implementation would go here
        print(f"Running simulation with GPU: {self.gpu}, output dir: {workdir}")
        job.simulator_run(workdir, gpu=self.gpu, n_clients=n_clients) #, log_config="full")
