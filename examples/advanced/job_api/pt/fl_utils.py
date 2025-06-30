from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict

from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.widgets.widget import Widget
from nvflare.apis.executor import Executor
from nvflare.apis.impl.controller import Controller
from nvflare.app_common.trainers.pt import PyTorchTrainer
from src.net import Net

# Import FedAvg from the correct location
try:
    from nvflare.app_common.workflows.fedavg import FedAvg
except ImportError:
    # Fallback if the import fails
    class FedAvg:
        def __init__(self, num_clients: int, num_rounds: int, initial_model=None):
            self.num_clients = num_clients
            self.num_rounds = num_rounds
            self.initial_model = initial_model


class Peer:
    def add_dependency(self, dependency: Any) -> None:
        # submit any job dependency (file, directory, etc.) to the peer (server or client)
        raise NotImplementedError("Not implemented")


class Server(Peer):
    def __init__(self, controller: Controller, initial_model: Optional[object] = None) -> None:
        self.controller = controller
        self.initial_model = initial_model

    def add_controller(self, controller: Controller) -> None:
        raise NotImplementedError("Not implemented")

    def add_widget(self, widget: Widget) -> None:
        raise NotImplementedError("Not implemented")


class Client(Peer):
    def __init__(self, train_script: str) -> None:
        self.train_script = train_script
        self.runner = ScriptRunner(script=train_script)

    def add_executor(self, executor: Executor) -> None:
        raise NotImplementedError("Not implemented")


class ClientWithArgs(Client):
    def __init__(self, train_script: str, script_args: str) -> None:
        super().__init__(train_script)
        self.script_args = script_args
        self.runner = ScriptRunner(script=train_script, args=script_args)


class Strategy(ABC):
    """Abstract base class for federated learning strategies.
    
    This class defines the interface for different federated learning strategies.
    Each strategy should implement the setup method to configure the FL components.
    """
    
    def __init__(self, trainer: PyTorchTrainer):
        """Initialize the strategy with a PyTorch trainer.
        
        Args:
            trainer: PyTorchTrainer instance to be used by clients
        """
        self.trainer = trainer
    
    @abstractmethod
    def setup(self, n_clients: int, num_rounds: int, initial_model=None) -> 'FLRunner':
        """Setup the federated learning configuration.
        
        Args:
            n_clients: Number of clients to participate in training
            num_rounds: Number of training rounds
            initial_model: Initial model to start training with
            
        Returns:
            FLRunner: Configured FLRunner instance ready for simulation
        """
        pass


class FedAvgStrategy(Strategy):
    """CIFAR10 Federated Averaging Strategy.
    
    This strategy implements FedAvg for CIFAR10 dataset with configurable
    number of clients and training rounds.
    """
    
    def __init__(self, trainer: Trainer, n_clients: int, num_rounds: int, initial_model=None):
        """Setup FedAvg configuration.
        
        Args:
            num_rounds: Number of training rounds
            initial_model: Initial model to start training with (defaults to Net())
            
        Returns:
            FLRunner: Configured FLRunner instance ready for simulation
        """
        if initial_model is None:
            initial_model = Net()
            
        train_script = "src/cifar10_fl.py"
        
        # Create the FedAvg controller
        controller = FedAvg(
            num_clients=n_clients,
            num_rounds=num_rounds,
            initial_model=initial_model
        )
        
        # Create server with controller
        server = Server(controller, initial_model)
        
        # Create a single client template
        client = Client(trainer)

        return server, client        


class FLExperiment:
    def __init__(self, strategy: Strategy, n_clients: int, config: Optional[Dict] = None):
        self.strategy = strategy
        self.n_clients = n_clients
        self.config = config

    def run(self, env: 'Env'):
        server, client = self.strategy.setup(self.n_clients, self.config)

        job = create_job(server, client)
        
        # Create and return FLRunner
        env.run(job)


def create_job(server, client, job_name="fed_sim_job"):
    """
    Simplified API to run a federated learning simulation.
    Args:
        server: Server object containing the controller.
        clients: List of Client objects with training scripts.
        workdir: Directory for simulation output.
        gpu: GPU id as string, or None for CPU.
        job_name: Name for the job (optional).
    """
    job = BaseFedJob(
        name=job_name,
        initial_model=server.initial_model,
    )
    job.to_server(server.controller)

    # Add clients to the job
    for i, client in enumerate(clients):
        job.to(client.runner, f"site-{i}")

    return job


class Env(ABC):
    """Abstract base class for execution environments."""
    
    @abstractmethod
    def run(self):
        """Run the federated learning experiment."""
        pass


class SimEnv(Env):
    """Simulation environment for federated learning."""
    
    def __init__(self, gpu: Optional[str] = None, workdir: str = "/tmp/nvflare"):
        self.gpu = gpu
        self.workdir = workdir
    
    def run(self):
        """Run the simulation."""
        # Implementation would go here
        print(f"Running simulation with GPU: {self.gpu}, workdir: {self.workdir}")


class FlareEnv(Env):
    """NVFlare environment for federated learning."""
    
    def run(self):
        """Run the NVFlare experiment."""
        # Implementation would go here
        print("Running NVFlare experiment")


class FLRunner:
    def __init__(
        self,
        server: Server,
        client: Optional[Client] = None,
        clients: Optional[List[Client]] = None,
        n_clients: Optional[int] = None
    ) -> None:
        """Initialize the FLRunner with server and clients.
        
        Args:
            server: Server object containing the controller
            client: Single Client object with training script (optional)
            clients: List of Client objects with training scripts (optional)
            n_clients: Number of clients to create from the single client (optional)
        """
        self.server = server
        
        if client is not None and n_clients is not None:
            # Create n_clients copies of the same client
            self.clients = [client for _ in range(n_clients)]
        elif clients is not None:
            self.clients = clients
        else:
            raise ValueError("Either provide a single client with n_clients, or a list of clients")
        
        # Create the job
        self._job = self._create_job()


    def simulate(self, workdir: str, gpu: Optional[str] = None) -> None:
        """Run a federated learning simulation.
        
        Args:
            workdir: Directory for simulation output
            gpu: GPU id as string, or None for CPU
        """
        
        self._job.simulator_run(workdir, gpu=gpu)

    def export(self, job_dir: str) -> None:
        """Export the job configuration to a directory.
        
        Args:
            job_dir: Directory to export job configuration to
        """

        self._job.export(job_dir)

    def deploy(self, admindir: str) -> None:
        """Deploy the job to a running NVFlare system.
        
        Args:
            admindir: Directory containing admin startup kit
        """
        
        self._job.deploy(admindir) 
