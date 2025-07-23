from abc import ABC, abstractmethod

from nvflare.job_config.api import FedJob


class Env(ABC):
    """Abstract base class for execution environments."""

    @abstractmethod
    def execute(self, job: FedJob):
        """Execute the federated learning experiment."""
        raise NotImplementedError("Not implemented")
