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
PyTorch FedAvg - Decorator-based implementation.

This example shows both the old Strategy-based approach and the new
decorator-based approach for comparison.
"""
import torch

from nvflare.fox.api import flare
from nvflare.fox.sys.recipe import FlareRecipe
from nvflare.fox.sim.simulator import SimEnv


# ============================================================================
# Server Implementation 
# ============================================================================

@flare.server
class FedAvgServer():
    def __init__(self, initial_model, num_rounds=3):
        self.initial_model = initial_model
        self.num_rounds = num_rounds

    @flare.main
    def run_fedavg(self):
        """
        Decorator-based federated averaging implementation.
        
        This replaces the PTFedAvg Strategy class with a simple function.
        """
        print(f"System info: {flare.sys_info()}")
        
        # Parse the initial model
        current_model = self.initial_model
        
        for i in range(self.num_rounds):
            print(f"\n=== Round {i} ===")
            
            # Call all clients to train
            results = flare.clients.train(i, current_model)
            
            # Aggregate results
            if not results:
                print(f"Round {i}: No results received")
                continue
            
            # Average the weights using the aggregate_results method
            current_model = self.aggregate_results(current_model, results)
            print(f"Round {i}: Aggregated from {len(results)} clients")
        
        print(f"\nFinal model: {current_model}")
        return current_model

    def aggregate_results(self, current_model, results):
        """
        Aggregate model weights from multiple clients using averaging.
        
        Args:
            current_model: Dictionary containing current model weights
            results: List of dictionaries containing client model updates
            
        Returns:
            Dictionary containing aggregated model weights
        """
        aggregated = {}
        for key in current_model.keys():
            total = None
            for result in results:
                if key in result:
                    if total is None:
                        total = result[key]
                    else:
                        total = total + result[key]
            
            if total is not None:
                aggregated[key] = torch.div(total, len(results))
        
        return aggregated


# ============================================================================
# Client Implementation 
# ============================================================================

@flare.client
class MyClient:

    def __init__(self, delta: float):
        self.delta = delta

    @flare.collab
    def train(self, current_round, weights):
        result = {}
        for k, v in weights.items():
            result[k] = v + self.delta

        print(f"Finished training round {current_round}")
        return result


# ============================================================================
# Main execution
# ============================================================================

def main():
    initial_model={
        "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    }

    recipe = FlareRecipe(
        name="pt_fedavg_intime",
        server=FedAvgServer(initial_model=initial_model, num_rounds=4),
        client=MyClient(delta=1.0),
    )

    env = SimEnv(num_clients=2, num_threads=2)
    run = recipe.execute(env)
    print(f"final result: {run.get_result()}")


if __name__ == "__main__":
    main()
