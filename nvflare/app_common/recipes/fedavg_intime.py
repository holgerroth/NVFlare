import types

from nvflare import FilterType
from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.api import FedJob
from nvflare.job_config.recipe import Recipe
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather

class InTimeFedAvgRecipe(Recipe):
    """Federated Averaging Recipe using InTimeAccumulateWeightedAggregator as the aggregator by default. 
    Adds each model update from a client to the aggregation as soon as they are received by the server.
    Use in cases where the server memory is limited and cannot store all the model updates.

    This recipe implements In-Time FedAvg with configurable
    number of clients and training rounds.
    """

    def __init__(
        self,
        train_script,
        train_args="",
        num_clients=1,
        num_rounds=3,
        initial_model=None,
        aggregator=None,
        filters=None,
    ):
        """Setup FedAvg configuration.

        Args:
            num_rounds: Number of training rounds
            num_clients: Number of clients to participate in FedAvg algorithm
            initial_model: Initial model to start training with
            aggregate_fn: Function to aggregate the models from clients
            filters: List of filters to apply to the strategy
            train_script: Script to train the model
            train_args: Arguments to pass to the train script
            filters: List of filters to apply to the strategy

        Returns:
        """

        super().__init__()
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.train_script = train_script
        self.train_args = train_args
        self.aggregator = aggregator
        self.filters = filters

        if self.aggregator is None:
            self.aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)

        self.job = self.setup()

    def setup(self) -> FedJob:
        # Create BaseFedJob with initial model
        job = BaseFedJob(
            initial_model=self.initial_model,
        )

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
        job.to_server(controller)

        # Add clients
        runner = ScriptRunner(script=self.train_script, script_args=self.train_args)
        job.to_clients(runner)

        if self.filters is not None:
            for filter in self.filters:
                job.add_filter(filter, FilterType.TASK_RESULT, ["train"])

        return job
