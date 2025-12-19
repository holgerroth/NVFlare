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

import torch

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator


class WeightedAggregator(Aggregator):
    """
    Weighted aggregation based on client data size.
    
    This aggregator weights each client's contribution by the number of training steps
    (or samples) they performed, which is more fair when clients have different dataset sizes.
    """

    def __init__(self):
        super().__init__()
        self.weighted_sum = {}
        self.total_weight = 0

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Accept a shareable from a client."""
        dxo = from_shareable(shareable)
        if dxo.data_kind == DataKind.WEIGHTS or dxo.data_kind == DataKind.WEIGHT_DIFF:
            # Get client's data size from metadata (num_steps is sent by client)
            weight = dxo.get_meta_prop("num_steps", 1.0)
            
            self.info(f"Accepting model with weight={weight}, {len(dxo.data)} parameters")
            
            for key, value in dxo.data.items():
                if key not in self.weighted_sum:
                    self.weighted_sum[key] = 0
                self.weighted_sum[key] += value * weight
            self.total_weight += weight
            return True
        return False

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Perform weighted aggregation and return result as Shareable."""
        self.info(f"Aggregating with total weight: {self.total_weight}")
        
        if self.total_weight == 0:
            self.error("Total weight is zero, cannot aggregate!")
            return None
        
        aggregated_params = {
            key: val / self.total_weight 
            for key, val in self.weighted_sum.items()
        }
        
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=aggregated_params)
        return dxo.to_shareable()

    def reset(self, fl_ctx: FLContext):
        """Reset the aggregator state for next round."""
        self.info("Resetting WeightedAggregator")
        self.weighted_sum = {}
        self.total_weight = 0


class MedianAggregator(Aggregator):
    """
    Median aggregation for Byzantine robustness.
    
    Instead of averaging, this aggregator computes the median of each parameter
    across all clients. This provides robustness against Byzantine (malicious) clients
    who might send adversarial model updates.
    """

    def __init__(self):
        super().__init__()
        self.client_models = []

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Accept a shareable from a client."""
        dxo = from_shareable(shareable)
        if dxo.data_kind == DataKind.WEIGHTS or dxo.data_kind == DataKind.WEIGHT_DIFF:
            self.info(f"Accepting model {len(self.client_models) + 1} with {len(dxo.data)} parameters")
            self.client_models.append(dxo.data)
            return True
        return False

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Perform median aggregation and return result as Shareable."""
        self.info(f"Aggregating {len(self.client_models)} models using median")
        
        if len(self.client_models) == 0:
            self.error("No client models to aggregate!")
            return None
        
        # Stack all client parameters and compute median
        aggregated_params = {}
        param_keys = self.client_models[0].keys()
        
        for key in param_keys:
            # Stack tensors from all clients
            stacked = torch.stack([m[key] for m in self.client_models])
            # Compute median along the client dimension (dim=0)
            aggregated_params[key] = torch.median(stacked, dim=0)[0]
        
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=aggregated_params)
        return dxo.to_shareable()

    def reset(self, fl_ctx: FLContext):
        """Reset the aggregator state for next round."""
        self.info("Resetting MedianAggregator")
        self.client_models = []

