import types

from nvflare import FilterType
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.api import FedJob
from nvflare.job_config.recipe import Recipe
from nvflare.job_config.script_runner import ScriptRunner


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
        aggregate_fn=None,
        sample_clients_fn=None,
        load_model_fn=None,
        save_model_fn=None,
        early_stop_fn=None,
    ):
        """Setup FedAvg configuration.

        Args:
            num_rounds: Number of training rounds
            num_clients: Number of clients to participate in FedAvg algorithm
            initial_model: Initial model to start training with
            aggregate_fn: Function to aggregate the models from clients
            train_script: Script to train the model
            train_args: Arguments to pass to the train script

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

        self.job = self.setup()

    def setup(self) -> FedJob:
        # Create BaseFedJob with initial model
        job = BaseFedJob(
            initial_model=self.initial_model,
        )

        # Define the controller and send to server
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

        job.to_server(controller)

        # Add clients
        runner = ScriptRunner(script=self.train_script, script_args=self.train_args)
        job.to_clients(runner)

        return job
