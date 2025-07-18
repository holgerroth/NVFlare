import types
from dataclasses import dataclass
from typing import Optional, List

from nvflare import FilterType
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.api import FedJob
from nvflare.job_config.recipe import Recipe
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.app_common.filters.percentile_privacy import PercentilePrivacy
from nvflare.app_common.filters.svt_privacy import SVTPrivacy
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator


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
    coeff_mod_bit_sizes: List[int] = [60, 40, 40]
    scale_bits: int = 40
    scheme: str = "CKKS"


class FedAvgRecipe(Recipe):
    """Federated Averaging Recipe.

    This recipe implements FedAvg with configurable
    number of clients and training rounds.
    """

    def __init__(
        self,
        train_script,
        train_args="",
        num_clients=1,
        num_rounds=3,
        initial_model=None,
        aggregate_fn=None,  # only used with FedAvg controller
        sample_clients_fn=None,  # only used with FedAvg controller
        load_model_fn=None,  # only used with FedAvg controller
        save_model_fn=None,  # only used with FedAvg controller
        early_stop_fn=None,  # only used with FedAvg controller
        privacy_config: Optional[PrivacyConfig] = None,
        he_config: Optional[HEConfig] = None,
        intime_aggregation: bool = False,  # only used with ScatterAndGather controller
        aggregator: Optional[Aggregator] = None,  # only used with ScatterAndGather controller
    ):
        """Setup FedAvg configuration.

        Args:
            train_script: Script to train the model
            train_args: Arguments to pass to the train script
            num_clients: Number of clients to participate in FedAvg algorithm
            num_rounds: Number of training rounds
            initial_model: Initial model to start training with
            aggregate_fn: Function to aggregate the models from clients
            sample_clients_fn: Function to sample clients for training
            load_model_fn: Function to load model
            save_model_fn: Function to save model
            early_stop_fn: Function for early stopping
            privacy_config: Configuration for privacy filters
            he_config: Configuration for homomorphic encryption
            intime_aggregation: Whether to aggregate models as soon as they are received (saves memory but requires special Aggregator class)

        Returns:
        """
        super().__init__()
        
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.train_script = train_script
        self.train_args = train_args
        self.aggregate_fn = aggregate_fn
        self.sample_clients_fn = sample_clients_fn
        self.load_model_fn = load_model_fn
        self.save_model_fn = save_model_fn
        self.early_stop_fn = early_stop_fn
        self.privacy_config = privacy_config
        self.he_config = he_config
        self.intime_aggregation = intime_aggregation
        self.aggregator = aggregator

        self.job = self.setup()

    def setup(self) -> FedJob:
        # Create BaseFedJob with initial model
        job = BaseFedJob(
            initial_model=self.initial_model,
        )

        # Define the controller and send to server
        if self.intime_aggregation:
            if self.aggregator is None:
                self.aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)

            # Define the controller and send to server
            shareable_generator = FullModelShareableGenerator()
            shareable_generator_id = job.to_server(shareable_generator, id="shareable_generator")
            aggregator_id = job.to_server(
                self.aggregator, id="aggregator"
            )

            controller = ScatterAndGather(
                min_clients=self.num_clients,
                num_rounds=self.num_rounds,
                wait_time_after_min_received=10,
                aggregator_id=aggregator_id,
                persistor_id=job.comp_ids["persistor_id"],
                shareable_generator_id=shareable_generator_id,
            )
        else:
            controller = FedAvg(
                num_clients=self.num_clients,
                num_rounds=self.num_rounds,
            )
            # TODO: support overwriting these functions
            if self.aggregate_fn is not None:
                controller.aggregate_fn = types.MethodType(
                    self.aggregate_fn, controller
                )  # MethodType is used to bind the function to the controller object
            if self.sample_clients_fn is not None:
                controller.sample_clients = types.MethodType(self.sample_clients_fn, controller)
            if self.load_model_fn is not None:
                controller.load_model = types.MethodType(self.load_model_fn, controller)
            if self.save_model_fn is not None:
                controller.save_model = types.MethodType(self.save_model_fn, controller)
            # if self.early_stop_fn is not None:  # TODO: support early stop in FedAvg
            #    controller.early_stop_fn = types.MethodType(self.early_stop_fn, controller)

        # Send the controller to the server
        job.to_server(controller)

        # Add clients
        runner = ScriptRunner(script=self.train_script, script_args=self.train_args)
        job.to_clients(runner)

        # Add privacy filters
        if self.privacy_config is not None:
            if self.privacy_config.percentile is not None:
                filter = PercentilePrivacy(
                    percentile=self.privacy_config.percentile, 
                    gamma=self.privacy_config.percentile_gamma
                )
                job.to_clients(filter, tasks=["train"], filter_type=FilterType.TASK_RESULT)

            filter = SVTPrivacy(
                fraction=self.privacy_config.fraction, 
                epsilon=self.privacy_config.epsilon, 
                noise_var=self.privacy_config.noise_var, 
                gamma=self.privacy_config.gamma, 
                tau=self.privacy_config.tau, 
                replace=self.privacy_config.replace
            )
            job.to_clients(filter, tasks=["train"], filter_type=FilterType.TASK_RESULT)

        if self.he_config is not None:
            # TODO: add homomorphic encryption to the job
            raise NotImplementedError("Homomorphic encryption is not implemented yet")
        

        return job
