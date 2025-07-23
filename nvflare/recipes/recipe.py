from abc import ABC, abstractmethod

from nvflare.environments.environment import Env
from nvflare.job_config.api import FedJob


class Recipe(ABC):
    def __init__(self):
        self.job = None
        self.num_clients = None

    @abstractmethod
    def setup(self) -> FedJob:
        raise NotImplementedError("Subclasses must implement this method")

    def execute(self, env: Env):
        if self.job is None:
            self.job = self.setup()
        
        env.execute(job=self.job)

    def export(self, path: str):
        if self.job is None:
            self.job = self.setup()
        
        self.job.export_job(path)
        print(f"Job exported to {path}")
