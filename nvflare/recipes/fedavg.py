from dataclasses import dataclass, field
from typing import Optional, List

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


@dataclass
class PrivacyConfig:
    """Configuration for privacy filters in FedAvg.

    Args:
        fraction: Fraction of the model to upload (default: 0.1)
        epsilon: Privacy parameter for differential privacy (default: 0.1)
        noise_var: Additive noise variance (default: 0.1)
        gamma: Clipping threshold (default: 1e-5)
        tau: Threshold parameter (default: 1e-6)
        replace: Whether to sample with replacement (default: True)
        percentile: Percentile for percentile privacy (default: None)
        percentile_gamma: Gamma for percentile privacy (default: 0.01)
    """

    fraction: float = 0.1
    epsilon: float = 0.1
    noise_var: float = 0.1
    gamma: float = 1e-5
    tau: float = 1e-6
    replace: bool = True
    percentile: Optional[int] = None
    percentile_gamma: float = 0.01  # will be ignored if percentile is None


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
        privacy_config: Optional[PrivacyConfig] = None,
        he_config: Optional[HEConfig] = None,
        quantization_type: Optional[str] = None,
        persistor: Optional[ModelPersistor] = None,
        external_client_process: bool = False,
        client_command_prefix: Optional[str] = "python3 -u",
        server_expected_format: Optional[str] = ExchangeFormat.NUMPY,
    ):
        """Setup FedAvg configuration.

        Args:
            train_script: Script to train the model
            train_args: Arguments to pass to the train script
            num_clients: Number of clients to participate in FedAvg algorithm
            num_rounds: Number of training rounds
            initial_model: Initial model to start training with
            aggregator: Aggregator for combining client models. If not provided, InTimeAccumulateWeightedAggregator will be used
            privacy_config: Configuration for privacy filters
            he_config: Configuration for homomorphic encryption
            quantization_type: Configuration type for quantization
            persistor: ModelPersistor for saving and loading models. If not provided, the default model persistor will be used.
            external_client_process: Whether to use an external process for the client. If True, the client script will be run as a separate process.
            client_command_prefix: If launch_external_process=True, command to run script (preprended to script). Defaults to "python3".
        """
        super().__init__()

        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.train_script = train_script
        self.train_args = train_args
        self.privacy_config = privacy_config
        self.he_config = he_config
        self.aggregator = aggregator
        self.quantization_type = quantization_type
        self.persistor = persistor
        self.external_client_process = external_client_process
        self.client_command_prefix = client_command_prefix
        self.server_expected_format = server_expected_format
        self.job = self.setup()

    def setup(self) -> FedJob:
        # Create BaseFedJob with initial model
        job = BaseFedJob(
            initial_model=self.initial_model,
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
            min_clients=self.num_clients,
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
            expected_format=self.server_expected_format,
        )
        job.to_clients(runner)

        # TODO: factor out to enable reuse of filters in different recipes
        # Add privacy filters
        if self.privacy_config is not None:
            if self.privacy_config.percentile is not None:
                filter = PercentilePrivacy(
                    percentile=self.privacy_config.percentile, gamma=self.privacy_config.percentile_gamma
                )
                job.to_clients(filter, tasks=["train"], filter_type=FilterType.TASK_RESULT)

            filter = SVTPrivacy(
                fraction=self.privacy_config.fraction,
                epsilon=self.privacy_config.epsilon,
                noise_var=self.privacy_config.noise_var,
                gamma=self.privacy_config.gamma,
                tau=self.privacy_config.tau,
                replace=self.privacy_config.replace,
            )
            job.to_clients(filter, tasks=["train"], filter_type=FilterType.TASK_RESULT)

        if self.he_config is not None:
            # TODO: add homomorphic encryption to the job
            raise NotImplementedError("Homomorphic encryption is not implemented yet")

        if self.quantization_type is not None:
            # If using quantization, add quantize filters.
            quantizer = ModelQuantizer(quantization_type=self.quantization_type)
            dequantizer = ModelDequantizer()
            job.to_server(quantizer, tasks=["train"], filter_type=FilterType.TASK_DATA)
            job.to_server(dequantizer, tasks=["train"], filter_type=FilterType.TASK_RESULT)

            job.to_clients(quantizer, tasks=["train"], filter_type=FilterType.TASK_RESULT)
            job.to_clients(dequantizer, tasks=["train"], filter_type=FilterType.TASK_DATA)

        return job
