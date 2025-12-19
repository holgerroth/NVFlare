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
"""
Custom aggregator implementations for FedAvg recipe.
This module provides two example aggregators:
1. WeightedAggregator: Aggregates based on client data size (num_steps)
2. MedianAggregator: Uses median aggregation for Byzantine robustness
"""

import numpy as np

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


class WeightedAggregator(ModelAggregator):
    """
    Weighted aggregation based on client data size.
    
    This aggregator weights each client's contribution by the number of training steps
    (or samples) they performed, which is more fair when clients have different dataset sizes.
    """

    def __init__(self):
        super().__init__()
        self.weighted_sum = {}
        self.total_weight = 0
        self.client_weights = []  # Track individual client weights for debugging

    def accept_model(self, model: FLModel):
        """Accept submitted model and add to the weighted sum."""
        # Get client's data size from metadata (NUM_STEPS_CURRENT_ROUND is sent by client)
        weight = model.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
        self.client_weights.append(weight)
        
        self.info(f"Accepting model with weight={weight}, {len(model.params)} parameters")
        
        for key, value in model.params.items():
            if key not in self.weighted_sum:
                self.weighted_sum[key] = value * weight
            else:
                self.weighted_sum[key] += value * weight
        self.total_weight += weight
        
        # Debug: check a sample parameter
        if len(model.params) > 0:
            sample_key = list(model.params.keys())[0]
            sample_value = model.params[sample_key]
            self.info(f"Sample param '{sample_key}': shape={sample_value.shape}, "
                     f"mean={np.mean(np.abs(sample_value)):.6f}, "
                     f"weighted_mean={np.mean(np.abs(sample_value * weight)):.6f}")

    def aggregate_model(self) -> FLModel:
        """Perform weighted aggregation and return result as FLModel."""
        self.info(f"Aggregating {len(self.client_weights)} clients with weights: {self.client_weights}")
        self.info(f"Total weight: {self.total_weight}, Mean weight: {np.mean(self.client_weights):.2f}, "
                 f"Std weight: {np.std(self.client_weights):.2f}")
        
        if self.total_weight == 0:
            self.error("Total weight is zero, cannot aggregate!")
            return FLModel(params={})
        
        aggregated_params = {
            key: val / self.total_weight 
            for key, val in self.weighted_sum.items()
        }
        
        # Debug: check a sample aggregated parameter
        if len(aggregated_params) > 0:
            sample_key = list(aggregated_params.keys())[0]
            sample_value = aggregated_params[sample_key]
            self.info(f"Aggregated sample param '{sample_key}': shape={sample_value.shape}, "
                     f"mean={np.mean(np.abs(sample_value)):.6f}")
        
        # Reset state after aggregation for next round
        self.weighted_sum = {}
        self.total_weight = 0
        self.client_weights = []
        
        return FLModel(params=aggregated_params, params_type=ParamsType.DIFF)

    def reset_stats(self):
        """Reset the aggregator state for next round."""
        self.info(f"Resetting WeightedAggregator (had {len(self.client_weights)} clients)")
        self.weighted_sum = {}
        self.total_weight = 0
        self.client_weights = []


class MedianAggregator(ModelAggregator):
    """
    Median aggregation for Byzantine robustness.
    
    Instead of averaging, this aggregator computes the median of each parameter
    across all clients. This provides robustness against Byzantine (malicious) clients
    who might send adversarial model updates.
    """

    def __init__(self):
        super().__init__()
        self.client_models = []

    def accept_model(self, model: FLModel):
        """Accept submitted model and add to collection."""
        self.info(f"Accepting model {len(self.client_models) + 1} with {len(model.params)} parameters")
        self.client_models.append(model.params)

    def aggregate_model(self) -> FLModel:
        """Perform median aggregation and return result as FLModel."""
        self.info(f"Aggregating {len(self.client_models)} models using median")
        
        if len(self.client_models) == 0:
            self.error("No client models to aggregate!")
            return FLModel(params={})
        
        # Stack all client parameters and compute median using numpy
        aggregated_params = {}
        param_keys = self.client_models[0].keys()
        
        for key in param_keys:
            # Stack arrays from all clients along axis 0
            stacked = np.stack([m[key] for m in self.client_models], axis=0)
            # Compute median along the client dimension (axis=0)
            aggregated_params[key] = np.median(stacked, axis=0)
        
        # Reset state after aggregation for next round
        self.client_models = []
        
        return FLModel(params=aggregated_params, params_type=ParamsType.DIFF)

    def reset_stats(self):
        """Reset the aggregator state for next round."""
        self.info("Resetting MedianAggregator")
        self.client_models = []
