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
from nvflare.app_opt.pt.quantization.dequantizer import ModelDequantizer
from nvflare.app_opt.pt.quantization.quantizer import ModelQuantizer
from nvflare.client.config import ExchangeFormat
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.app_opt.he.model_encryptor import HEModelEncryptor
from nvflare.app_opt.he.model_decryptor import HEModelDecryptor


class PrivacyPolicy(ABC):
    """Base class for privacy policies.
    
    All privacy policies should inherit from this class and implement
    the create_filter method to return the appropriate DXOFilter instance.
    """
    
    @abstractmethod
    def create_filter(self) -> DXOFilter:
        """Create and return the DXOFilter instance for this policy.
        
        Returns:
            DXOFilter: The filter instance to be applied
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
    
    def create_filter(self) -> DXOFilter:
        return PercentilePrivacy(
            percentile=self.percentile,
            gamma=self.gamma
        )


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
    
    def create_filter(self) -> DXOFilter:
        return SVTPrivacy(
            fraction=self.fraction,
            epsilon=self.epsilon,
            noise_var=self.noise_var,
            gamma=self.gamma,
            tau=self.tau,
            replace=self.replace
        )


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
    
    def create_encrypt_filter(self) -> DXOFilter:
        """Create the encryption filter for outgoing data."""
        return HEModelEncryptor(
            tenseal_context_file=self.tenseal_context_file,
            encrypt_layers=self.encrypt_layers,
            aggregation_weights=self.aggregation_weights,
            weigh_by_local_iter=self.weigh_by_local_iter
        )
    
    def create_decrypt_filter(self) -> DXOFilter:
        """Create the decryption filter for incoming data."""
        return HEModelDecryptor(
            tenseal_context_file=self.tenseal_context_file
        )
    
    def create_filter(self) -> DXOFilter:
        """Create and return the encryption filter (for backward compatibility).
        
        Note: For HE, you typically need both encryption and decryption filters.
        Use create_encrypt_filter() and create_decrypt_filter() separately
        to get the appropriate filters for different filter types.
        """
        return self.create_encrypt_filter()


@dataclass
class HEConfig:
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [60, 40, 40])
    scale_bits: int = 40
    scheme: str = "CKKS"


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
        num_clients=1,
        num_rounds=3,
        initial_model=None,
        aggregator: Optional[Aggregator] = None,
        privacy_policies: Optional[List[PrivacyPolicy]] = None,
        he_config: Optional[HEConfig] = None,
        quantization_type: Optional[str] = None,
        persistor: Optional[ModelPersistor] = None,
        external_client_process: bool = False,
        client_command_prefix: Optional[str] = "python3 -u",
        server_expected_format: Optional[str] = ExchangeFormat.NUMPY,
        min_clients: Optional[int] = None,
        allow_server_numpy_conversion: bool = True,
    ):
        """Setup FedAvg configuration.

        Args:
            train_script: Script to train the model
            train_args: Arguments to pass to the train script
            num_clients: Number of clients to participate in FedAvg algorithm
            num_rounds: Number of training rounds
            initial_model: Initial model to start training with
            aggregator: Aggregator for combining client models. If not provided, InTimeAccumulateWeightedAggregator will be used
            privacy_policies: List of privacy policies to apply. Each policy should inherit from PrivacyPolicy.
            he_config: Configuration for homomorphic encryption (deprecated, use HEPrivacyPolicy instead)
            quantization_type: Configuration type for quantization
            persistor: ModelPersistor for saving and loading models. If not provided, the default model persistor will be used.
            external_client_process: Whether to use an external process for the client. If True, the client script will be run as a separate process.
            client_command_prefix: If launch_external_process=True, command to run script (preprended to script). Defaults to "python3".
            min_clients: Minimum number of clients to proceed to next round of FedAvg algorithm. If not provided, the number of active clients will be used.
            allow_server_numpy_conversion: Whether to allow the server to convert the model to numpy. If True, the server will convert the model to numpy. Default is True.
        """
        super().__init__()

        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.train_script = train_script
        self.train_args = train_args
        self.privacy_policies = privacy_policies or []
        self.he_config = he_config
        self.aggregator = aggregator
        self.quantization_type = quantization_type
        self.persistor = persistor
        self.external_client_process = external_client_process
        self.client_command_prefix = client_command_prefix
        self.server_expected_format = server_expected_format
        self.min_clients = min_clients
        self.allow_server_numpy_conversion = allow_server_numpy_conversion

        # Handle deprecated he_config parameter
        if self.he_config is not None:
            import warnings
            warnings.warn(
                "he_config parameter is deprecated. Use HEPrivacyPolicy in privacy_policies instead.",
                DeprecationWarning,
                stacklevel=2
            )
            # Convert he_config to HEPrivacyPolicy for backward compatibility
            he_policy = HEPrivacyPolicy()
            self.privacy_policies.append(he_policy)

        if isinstance(self.num_clients, int):
            self.client_names = [f"site-{i+1}" for i in range(self.num_clients)]
        elif isinstance(self.num_clients, list):
            self.client_names = self.num_clients
        else:
            raise ValueError(f"Invalid type for num_clients: {type(self.num_clients)}. Expected int or list of strings but got {type(self.num_clients)}")
        
        if self.min_clients is None:
            self.min_clients = len(self.client_names)

        self.job = self.setup()

    def setup(self) -> FedJob:
        # Create BaseFedJob with initial model
        job = BaseFedJob(
            initial_model=self.initial_model,
            allow_server_numpy_conversion=self.allow_server_numpy_conversion,
        )

        # Define the controller and send to server
        if self.aggregator is None:
            self.aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)

        if self.persistor is not None:
            if self.initial_model is not None:
                raise ValueError("Initial model is not supported when using a custom persistor")
            job.comp_ids["persistor_id"] = job.to_server(self.persistor, id="persistor")

        # Define the controller and send to server
        shareable_generator = FullModelShareableGenerator()
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
        for client_name in self.client_names:
            runner = ScriptRunner(
                script=self.train_script,
                script_args=self.train_args,
                launch_external_process=self.external_client_process,
                command=self.client_command_prefix,
                server_expected_format=self.server_expected_format,
            )

            runner = ScriptRunner(
                script=self.train_script,
                script_args=self.train_args,
                launch_external_process=self.external_client_process,
                command=self.client_command_prefix,
                server_expected_format=self.server_expected_format,
            )
            job.to(runner, target=client_name)

            # Add privacy filters from policies
            for i, policy in enumerate(self.privacy_policies):
                if not isinstance(policy, PrivacyPolicy):
                    raise ValueError(f"Policy {i} must inherit from PrivacyPolicy, got {type(policy)}")
                
                if isinstance(policy, HEPrivacyPolicy):
                    # For HE, we need both encryption and decryption filters
                    encrypt_filter = policy.create_encrypt_filter()
                    decrypt_filter = policy.create_decrypt_filter()
                    
                    # Add encryption filter to task results (outgoing data)
                    job.to(encrypt_filter, target=client_name, tasks=["train"], filter_type=FilterType.TASK_RESULT)
                    
                    # Add decryption filter to task data (incoming data)
                    job.to(decrypt_filter, target=client_name, tasks=["train"], filter_type=FilterType.TASK_DATA)
                else:
                    # For other privacy policies, use the standard approach
                    filter_instance = policy.create_filter()
                    job.to(filter_instance, target=client_name, tasks=["train"], filter_type=FilterType.TASK_RESULT)

            if self.quantization_type is not None:
                # If using quantization, add quantize filters.
                quantizer = ModelQuantizer(quantization_type=self.quantization_type)
                dequantizer = ModelDequantizer()
                job.to_server(quantizer, tasks=["train"], filter_type=FilterType.TASK_DATA)
                job.to_server(dequantizer, tasks=["train"], filter_type=FilterType.TASK_RESULT)

                job.to(quantizer, target=client_name, tasks=["train"], filter_type=FilterType.TASK_RESULT)
                job.to(dequantizer, target=client_name, tasks=["train"], filter_type=FilterType.TASK_DATA)

        return job
