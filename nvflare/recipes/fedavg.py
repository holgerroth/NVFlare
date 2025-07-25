from dataclasses import dataclass, field
from typing import Optional, List, Union, Callable
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
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator, CallableAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.app_opt.he.model_encryptor import HEModelEncryptor
from nvflare.app_opt.he.model_decryptor import HEModelDecryptor
from nvflare.apis.fl_component import FLComponent
from nvflare.app_common.abstract.fl_model import FLModel


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
        train_args="",  # TODO: support different train args for different clients
        min_clients=1,
        num_rounds=3,
        initial_model=None,
        aggregator: Optional[Union[Aggregator, Callable[[List[FLModel]], FLModel]]] = None,
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
        elif callable(self.aggregator):
            # Wrap callable function into CallableAggregator
            self.aggregator = CallableAggregator(self.aggregator)
        elif isinstance(self.aggregator, Aggregator):
            pass
        else:
            raise ValueError(f"Invalid aggregator type: {type(self.aggregator)}")

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

        # TODO: add logic to support multi-gpu/external process
        # Add clients
        runner = ScriptRunner(
            script=self.train_script,
            script_args=self.train_args,
        )
        job.to_clients(runner)

        # Apply any filters that were added using the filter methods
        self._apply_filters_to_job(job)

        return job
