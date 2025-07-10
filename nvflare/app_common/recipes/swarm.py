from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.ccwf.ccwf_job import CCWFJob, CrossSiteEvalConfig, SwarmClientConfig, SwarmServerConfig
from nvflare.app_common.ccwf.comps.simple_model_shareable_generator import SimpleModelShareableGenerator
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.job_config.api import FedJob
from nvflare.job_config.recipe import Recipe
from nvflare.job_config.script_runner import ScriptRunner


class SwarmRecipe(Recipe):
    """Swarm Learning Recipe.

    This strategy implements Swarm Learning with configurable
    number of clients and training rounds.
    """

    def __init__(
        self,
        train_script,
        num_clients=1,
        num_rounds=3,
        train_args="",
        initial_model=None,
        aggregator=None,
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
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.train_script = train_script
        self.train_args = train_args
        self.aggregator = aggregator

        if self.aggregator is None:
            self.aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)

    def setup(self) -> FedJob:
        # Create client-controlled swarm learning job with initial model
        job = CCWFJob(name="swarm")

        job.add_swarm(
            server_config=SwarmServerConfig(num_rounds=self.num_rounds),
            client_config=SwarmClientConfig(
                executor=ScriptRunner(script=self.train_script, script_args=self.train_args),
                aggregator=self.aggregator,
                persistor=PTFileModelPersistor(model=self.initial_model),
                shareable_generator=SimpleModelShareableGenerator(),
            ),
            cse_config=CrossSiteEvalConfig(eval_task_timeout=300),
        )

        return job
