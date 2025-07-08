from abc import ABC, abstractmethod

from nvflare.job_config.api import FedJob
from nvflare.environments.environment import Env

class Recipe(ABC):
    def __init__(self):
        self.job = None

    @abstractmethod
    def setup(self) -> FedJob:
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, env: Env):
        self.job = self.setup()
        env.run(job=self.job)
