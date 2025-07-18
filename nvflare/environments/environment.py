from abc import ABC, abstractmethod


class Env(ABC):
    """Abstract base class for execution environments."""

    @abstractmethod
    def execute(self):
        """Execute the federated learning experiment."""
        raise NotImplementedError("Not implemented")
