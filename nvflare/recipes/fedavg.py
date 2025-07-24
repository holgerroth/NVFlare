from dataclasses import dataclass, field
from typing import Optional, List, Union
from abc import ABC, abstractmethod

from nvflare import FilterType
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.api import FedJob
from nvflare.recipes.recipe import Recipe
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.app_common.filters.percentile_privacy import PercentilePrivacy
from nvflare.app_common.filters.svt_privacy import SVTPrivacy
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.app_opt.he.model_encryptor import HEModelEncryptor
from nvflare.app_opt.he.model_decryptor import HEModelDecryptor
from nvflare.apis.fl_component import FLComponent


class PrivacyPolicy(ABC):
    """Base class for privacy policies.
    
    All privacy policies should inherit from this class and implement
    the appropriate filter creation methods based on where the filter
    should be applied in the federated learning pipeline.
    """
    
    @abstractmethod
    def create_client_result_filters(self) -> List[DXOFilter]:
        """Create and return the list of DXOFilter instances for client task results (outgoing data).
        
        Returns:
            List[DXOFilter]: The list of filter instances to be applied to client task results
        """
        pass
    
    @abstractmethod
    def create_client_data_filters(self) -> List[DXOFilter]:
        """Create and return the list of DXOFilter instances for client task data (incoming data).
        
        Returns:
            List[DXOFilter]: The list of filter instances to be applied to client task data
        """
        pass
    
    @abstractmethod
    def create_server_result_filters(self) -> List[DXOFilter]:
        """Create and return the list of DXOFilter instances for server task results (outgoing data).
        
        Returns:
            List[DXOFilter]: The list of filter instances to be applied to server task results
        """
        pass
    
    @abstractmethod
    def create_server_data_filters(self) -> List[DXOFilter]:
        """Create and return the list of DXOFilter instances for server task data (incoming data).
        
        Returns:
            List[DXOFilter]: The list of filter instances to be applied to server task data
        """
        pass

    def create_server_components(self, component: FLComponent) -> list[FLComponent]:
        """Create and return the list of FLComponents for the server.
        
        Returns:
            list[FLComponent]: The list of FLComponents for the server
        """
        pass

    def create_client_components(self, component: FLComponent) -> list[FLComponent]:
        """Create and return the list of FLComponents for the server.
        
        Returns:
            list[FLComponent]: The list of FLComponents for the server
        """
        pass


@dataclass
class PercentilePrivacyPolicy(PrivacyPolicy):
    """Privacy policy for percentile-based filtering.
    
    Args:
        percentile: Only abs diff greater than this percentile is updated (0..100)
        gamma: The upper limit to truncate abs values of weight diff
    """
    percentile: int = 10
    gamma: float = 0.01
    
    def create_client_result_filters(self) -> List[DXOFilter]:
        return [PercentilePrivacy(
            percentile=self.percentile,
            gamma=self.gamma
        )]


@dataclass
class SVTPrivacyPolicy(PrivacyPolicy):
    """Privacy policy for Sparse Vector Technique differential privacy.
    
    Args:
        fraction: Fraction of the model to upload
        epsilon: Privacy parameter for differential privacy
        noise_var: Additive noise variance
        gamma: Clipping threshold
        tau: Threshold parameter
        replace: Whether to sample with replacement
    """
    fraction: float = 0.1
    epsilon: float = 0.1
    noise_var: float = 0.1
    gamma: float = 1e-5
    tau: float = 1e-6
    replace: bool = True
    
    def create_client_result_filter(self) -> List[DXOFilter]:
        return [SVTPrivacy(
            fraction=self.fraction,
            epsilon=self.epsilon,
            noise_var=self.noise_var,
            gamma=self.gamma,
            tau=self.tau,
            replace=self.replace
        )]
    
    def create_client_data_filter(self) -> List[DXOFilter]:
        return [SVTPrivacy(
            fraction=self.fraction,
            epsilon=self.epsilon,
            noise_var=self.noise_var,
            gamma=self.gamma,
            tau=self.tau,
            replace=self.replace
        )]
    
    def create_server_result_filter(self) -> List[DXOFilter]:
        return [SVTPrivacy(
            fraction=self.fraction,
            epsilon=self.epsilon,
            noise_var=self.noise_var,
            gamma=self.gamma,
            tau=self.tau,
            replace=self.replace
        )]
    
    def create_server_data_filter(self) -> List[DXOFilter]:
        return [SVTPrivacy(
            fraction=self.fraction,
            epsilon=self.epsilon,
            noise_var=self.noise_var,
            gamma=self.gamma,
            tau=self.tau,
            replace=self.replace
        )]


@dataclass
class HEPrivacyPolicy(PrivacyPolicy):
    """Privacy policy for Homomorphic Encryption.
    
    This policy creates both encryption and decryption filters for HE.
    The encryption filter is applied to task results (outgoing data),
    and the decryption filter is applied to task data (incoming data).
    
    Args:
        tenseal_context_file: TenSEAL context file containing encryption keys and parameters
        encrypt_layers: Layers to encrypt. If None, all layers are encrypted.
                       If list of strings, only specified layers are encrypted.
                       If string, treated as regex pattern to match layer names.
        aggregation_weights: Dictionary of client aggregation weights
        weigh_by_local_iter: Whether to multiply client weights by local iterations before encryption
    """
    tenseal_context_file: str = "client_context.tenseal"
    encrypt_layers: Optional[Union[List[str], str]] = None
    aggregation_weights: Optional[dict] = None
    weigh_by_local_iter: bool = True
    
    def create_client_result_filter(self) -> List[DXOFilter]:
        """Create the encryption filter for client outgoing data."""
        return [HEModelEncryptor(
            tenseal_context_file=self.tenseal_context_file,
            encrypt_layers=self.encrypt_layers,
            aggregation_weights=self.aggregation_weights,
            weigh_by_local_iter=self.weigh_by_local_iter
        )]
    
    def create_client_data_filter(self) -> List[DXOFilter]:
        """Create the decryption filter for client incoming data."""
        return [HEModelDecryptor(
            tenseal_context_file=self.tenseal_context_file
        )]
    
    def create_server_result_filter(self) -> List[DXOFilter]:
        """Create the encryption filter for server outgoing data."""
        return [HEModelEncryptor(
            tenseal_context_file=self.tenseal_context_file,
            encrypt_layers=self.encrypt_layers,
            aggregation_weights=self.aggregation_weights,
            weigh_by_local_iter=self.weigh_by_local_iter
        )]
    
    def create_server_data_filter(self) -> List[DXOFilter]:
        """Create the decryption filter for server incoming data."""
        return [HEModelDecryptor(
            tenseal_context_file=self.tenseal_context_file
        )]


class FedAvgRecipe(Recipe):
    """Federated Averaging Recipe.

    This recipe implements FedAvg with configurable
    number of clients and training rounds.
    Uses the ScatterAndGather controller to distribute the global model to the clients.
    Uses the ScriptRunner to run the training script.
    """

    def __init__(
        self,
        train_script,
        train_args="",
        min_clients=1,
        num_rounds=3,
        initial_model=None,
        aggregator: Optional[Aggregator] = None,
        framework: Optional[str] = "pytorch",
        server_load_model_func: Optional[Callable[[], FLModel]] = None,
        server_save_model_func: Optional[Callable[[FLModel], None]] = None,
        stop_condition: Optional[str] = None,
        patience: Optional[int] = None
    ):
        """Setup FedAvg configuration.

        Args:
            train_script: Script to train the model
            train_args: Arguments to pass to the train script
            min_clients: Minimum number of clients to proceed to next round of FedAvg algorithm. Default is 1.
            num_rounds: Number of training rounds
            initial_model: Initial model to start training with
            aggregator: Aggregator for combining client models. Can be `Aggregator` or `Callable`. If not provided, InTimeAccumulateWeightedAggregator will be used. 
            framework: Framework to use. Can be "pytorch", "raw", "tensorflow". Default is "pytorch".
            server_load_model_func: Function to load the model from the server.
            server_save_model_func: Function to save the model to the server.
            stop_cond (str, optional): early stopping condition based on metric. String
                literal in the format of '\\<key\\> \\<op\\> \\<value\\>' (e.g. "accuracy >= 80")
            patience (int, optional): The number of checks with no improvement after which
                the FL will be stopped. If set to `None`, this parameter is disabled.
                If stop_condition is None, patience does not apply
        """
        super().__init__()

        self.min_clients = min_clients
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.train_script = train_script
        self.train_args = train_args
        self.aggregator = aggregator
        self.framework = framework
        self.server_load_model_func = server_load_model_func
        self.server_save_model_func = server_save_model_func
        self.stop_condition = stop_condition
        self.patience = patience

        self.job = self.setup()

    def setup(self) -> FedJob:
        # Create BaseFedJob with initial model
        job = BaseFedJob(
            initial_model=self.initial_model,
            allow_server_numpy_conversion=True, # TODO: add logic based on framework
        )

        # Define the controller and send to server
        if self.aggregator is None:
            self.aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)

        if self.persistor is not None:
            if self.initial_model is not None:
                raise ValueError("Initial model is not supported when using a custom persistor")
            job.comp_ids["persistor_id"] = job.to_server(self.persistor, id="persistor")

        # Define the controller and send to server
        shareable_generator = FullModelShareableGenerator()   # TODO: Needs to be replaced with HE shareable generator if HE is used
        shareable_generator_id = job.to_server(shareable_generator, id="shareable_generator")
        aggregator_id = job.to_server(self.aggregator, id="aggregator")

        controller = ScatterAndGather(
            min_clients=self.min_clients,
            num_rounds=self.num_rounds,
            wait_time_after_min_received=10,
            aggregator_id=aggregator_id,
            persistor_id=job.comp_ids["persistor_id"],
            shareable_generator_id=shareable_generator_id,
        )
        # Send the controller to the server
        job.to_server(controller)

        # Add clients
        runner = ScriptRunner(
            script=self.train_script,
            script_args=self.train_args,
            launch_external_process=self.external_client_process,
            command=self.client_command_prefix,
            server_expected_format=self.server_expected_format,
        )
        job.to_clients(runner)

        # Add privacy filters from policies
        for i, policy in enumerate(self.privacy_policies):
            if not isinstance(policy, PrivacyPolicy):
                raise ValueError(f"Policy {i} must inherit from PrivacyPolicy, got {type(policy)}")
            
            # Add client filters
            client_result_filters = policy.create_client_result_filter()
            client_data_filters = policy.create_client_data_filter()
            
            for client_result_filter in client_result_filters:
                job.to_clients(client_result_filter, tasks=["train"], filter_type=FilterType.TASK_RESULT)
            for client_data_filter in client_data_filters:
                job.to_clients(client_data_filter, tasks=["train"], filter_type=FilterType.TASK_DATA)
            
            # Add server filters
            server_result_filters = policy.create_server_result_filter()
            server_data_filters = policy.create_server_data_filter()
            
            for server_result_filter in server_result_filters:
                job.to_server(server_result_filter, tasks=["train"], filter_type=FilterType.TASK_RESULT)
            for server_data_filter in server_data_filters:
                job.to_server(server_data_filter, tasks=["train"], filter_type=FilterType.TASK_DATA)

        return job
