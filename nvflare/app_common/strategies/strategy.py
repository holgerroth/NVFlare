from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict

from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.widgets.widget import Widget
from nvflare.apis.executor import Executor
from nvflare.apis.impl.controller import Controller
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_common.trainers.pt_trainer import Trainer


class Peer:
    def __init__(self):
        self.widgets = []

    def add_dependency(self, dependency: Any) -> None:
        # submit any job dependency (file, directory, etc.) to the peer (server or client)
        raise NotImplementedError("Not implemented")
    
    def add_widget(self, widget: Widget) -> None:
        self.widgets.append(widget)


class Server(Peer):
    def __init__(self, controller: Controller, initial_model: Optional[object] = None) -> None:
        self.controllers = [controller]
        self.initial_model = initial_model
        super().__init__()

    def add_controller(self, controller: Controller) -> None:
        self.controllers.append(controller)



class Client(Peer):
    def __init__(self, executor: Executor) -> None:
        self.executors = [executor]
        super().__init__()

    def add_executor(self, executor: Executor) -> None:
        self.executors.append(executor)



class Strategy(ABC):
    """Abstract base class for federated learning strategies.
    
    This class defines the interface for different federated learning strategies.
    Each strategy should implement the setup method to configure the FL components.
    """
    
    def __init__(self, trainer: Trainer):
        """Initialize the strategy with a PyTorch trainer.
        
        Args:
            trainer: PyTorchTrainer instance to be used by clients
        """
        self.trainer = trainer
        self.clients = []
        self.client = None
        self.server = None
    
    @abstractmethod
    def setup(self, n_clients: int, num_rounds: int, initial_model=None):
        """Setup the federated learning configuration.
        
        Args:
            n_clients: Number of clients to participate in training
            num_rounds: Number of training rounds
            initial_model: Initial model to start training with
            
        Returns: None
        """
        raise NotImplementedError("Not implemented")

    def get_clients(self, n_clients: int) -> List[Client]:
        """Get a list of clients.
        
        Args:
            n_clients: Number of clients to create
        """

        return [self.client for _ in range(n_clients)]

    def get_server(self) -> Server:
        """Get a server."""
        return self.server

