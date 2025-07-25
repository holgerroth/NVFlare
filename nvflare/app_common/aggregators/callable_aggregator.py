# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Callable

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


class CallableAggregator(ModelAggregator):
    """
    A wrapper class that converts a Callable function into a ModelAggregator.
    
    This class allows users to pass a function with signature `Callable[[List[FLModel]], FLModel]`
    as an aggregator, which will be automatically wrapped into a ModelAggregator instance.
    
    Args:
        aggregate_func: A callable function that takes a list of FLModel and returns a single FLModel
    """
    
    def __init__(self, aggregate_func: Callable[[List[FLModel]], FLModel]):
        super().__init__()
        if not callable(aggregate_func):
            raise ValueError("aggregate_func must be a callable function")
        self.aggregate_func = aggregate_func
        self.models = []
    
    def accept_model(self, model: FLModel):
        """Accept a model and add it to the collection for later aggregation."""
        self.models.append(model)
    
    def aggregate_model(self) -> FLModel:
        """Aggregate all collected models using the provided function."""
        if not self.models:
            raise ValueError("No models to aggregate")
        
        # Call the user-provided aggregation function
        result = self.aggregate_func(self.models)
        
        # Clear the models for next round
        self.reset_stats()
        
        return result
    
    def reset_stats(self):
        """Reset the internal state by clearing the collected models."""
        self.models = [] 