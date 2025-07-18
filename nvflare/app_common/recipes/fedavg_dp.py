import types

from nvflare import FilterType
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.api import FedJob
from nvflare.app_common.recipes.fedavg import FedAvgRecipe
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.app_common.filters.percentile_privacy import PercentilePrivacy
from nvflare.app_common.filters.svt_privacy import SVTPrivacy


# TODO: remove this class
class _FedAvgRecipeDP(FedAvgRecipe):
    """Federated Averaging Recipe with DP filters.

    This recipe implements FedAvg with configurable
    number of clients and training rounds, and DP filters.
    """

    def __init__(
        self,
        train_script,
        train_args="",
        num_clients=1,
        num_rounds=3,
        initial_model=None,
        aggregate_fn=None,
        privacy_fraction=0.1,
        privacy_epsilon=0.1,
        privacy_noise_var=0.1,
        privacy_gamma=1e-5,
        privacy_tau=1e-6,
        privacy_replace=True,
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
            

        Returns:
        """
        super().__init__(train_script, train_args, num_clients, num_rounds, initial_model, aggregate_fn)
        self.privacy_fraction = privacy_fraction
        self.privacy_epsilon = privacy_epsilon
        self.privacy_noise_var = privacy_noise_var
        self.privacy_gamma = privacy_gamma
        self.privacy_tau = privacy_tau
        self.privacy_replace = privacy_replace

    def setup(self) -> FedJob:
        # Create BaseFedJob with initial model
        job = super().__init__()



        return job
