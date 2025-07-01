from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def get_executor(self):
        raise NotImplementedError("Not implemented")
