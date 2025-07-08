from typing import Optional

from nvflare.environments.environment import Env


class SimEnv(Env):
    """Simulation environment for federated learning."""

    def __init__(self, gpu: Optional[str] = None, workdir: str = "/tmp/nvflare", name: str = "nvflare"):
        self.gpu = gpu
        self.workdir = workdir
        self.name = name

    def run(self, job):
        """Run the simulation."""
        job.name = self.name
        # Implementation would go here
        print(f"Running simulation with GPU: {self.gpu}, workdir: {self.workdir}")
        job.simulator_run(self.workdir, gpu=self.gpu)
