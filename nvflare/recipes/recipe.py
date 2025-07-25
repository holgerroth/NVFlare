from abc import ABC, abstractmethod
from typing import List

from nvflare.environments.environment import Env
from nvflare.job_config.api import FedJob
from nvflare import FilterType
from nvflare.apis.filter import Filter


class Recipe(ABC):
    def __init__(self):
        self.job = None
        self.num_clients = None
        self._client_input_filters = []
        self._client_output_filters = []        
        self._server_input_filters = []
        self._server_output_filters = []

    @abstractmethod
    def setup(self) -> FedJob:
        raise NotImplementedError("Subclasses must implement this method")

    def execute(self, env: Env):
        if self.job is None:
            self.job = self.setup()
        
        env.execute(job=self.job)

    def export(self, path: str):
        if self.job is None:
            self.job = self.setup()
        
        self.job.export_job(path)
        print(f"Job exported to {path}")

    def add_client_input_filter(self, filter: Filter):
        """Add a filter to be applied to client input (task data).
        
        Args:
            filter: The filter to be applied to client input
        """
        self._client_input_filters.append(filter)

    def add_client_output_filter(self, filter: Filter):
        """Add a filter to be applied to client output (task results).
        
        Args:
            filter: The filter to be applied to client output
        """
        self._client_output_filters.append(filter)

    def add_server_input_filter(self, filter: Filter):
        """Add a filter to be applied to server input (task data).
        
        Args:
            filter: The filter to be applied to server input
        """
        self._server_input_filters.append(filter)

    def add_server_output_filter(self, filter: Filter):
        """Add a filter to be applied to server output (task results).
        
        Args:
            filter: The filter to be applied to server output
        """
        self._server_output_filters.append(filter)

    def _apply_filters_to_job(self, job: FedJob):
        """Apply stored filters to the job.
        
        Args:
            job: The FedJob to apply filters to
        """
        # Apply client filters
        for filter in self._client_output_filters:
            job.to_clients(filter, tasks=["train"], filter_type=FilterType.TASK_RESULT)
        
        for filter in self._client_input_filters:
            job.to_clients(filter, tasks=["train"], filter_type=FilterType.TASK_DATA)
        
        # Apply server filters
        for filter in self._server_output_filters:
            job.to_server(filter, tasks=["train"], filter_type=FilterType.TASK_RESULT)
        
        for filter in self._server_input_filters:
            job.to_server(filter, tasks=["train"], filter_type=FilterType.TASK_DATA)
