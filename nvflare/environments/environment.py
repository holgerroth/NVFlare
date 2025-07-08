from abc import ABC, abstractmethod

class Env(ABC):
    """Abstract base class for execution environments."""

    @abstractmethod
    def run(self):
        """Run the federated learning experiment."""
        raise NotImplementedError("Not implemented")


