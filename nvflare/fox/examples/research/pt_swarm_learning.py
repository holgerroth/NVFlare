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
PyTorch Swarm Learning - Decentralized peer-to-peer learning implementation.

Swarm learning is a decentralized federated learning approach where:
- There is no central aggregator
- Each client can act as both trainer and aggregator
- Clients communicate in a peer-to-peer manner
- Each round, a random client is chosen to aggregate results from all clients
- The aggregator client then passes the aggregated model to the next randomly selected aggregator

This is different from traditional FedAvg where:
- Server always aggregates (centralized)
- Clients only train (never aggregate)
- Server controls the workflow

In swarm learning:
- Any client can be selected to aggregate in each round
- Aggregation responsibility rotates among clients
- No single point of failure or trust
"""
import random
import torch
import torch.nn as nn

from nvflare.fox.api import flare
from nvflare.fox.sys.recipe import FlareRecipe
from nvflare.fox.sim.simulator import SimEnv


# ============================================================================
# Server Implementation (minimal role - just coordination)
# ============================================================================

@flare.server
class SwarmServer():
    """
    Minimal server for swarm learning.
    
    In swarm learning, the server only:
    - Initiates the process by selecting the first aggregator
    - Waits for completion notification
    - Does NOT handle model weights or aggregation
    """
    
    def __init__(self, initial_model, num_rounds=3):
        self.initial_model = initial_model
        self.num_rounds = num_rounds
        self.completed = False

    @flare.main
    def run_swarm(self):
        """
        Initiate swarm learning by selecting a random client to start.
        """
        print(f"System info: {flare.sys_info()}")
        print(f"\n=== Starting Swarm Learning ===")
        print(f"Training for {self.num_rounds} rounds in decentralized manner")
        
        # Randomly select the first aggregator client
        # In a real swarm system, this could be based on various criteria
        first_aggregator_idx = random.randint(0, 2)  # Assuming 3 clients (0, 1, 2)
        print(f"Selected client {first_aggregator_idx} as initial aggregator\n")
        
        # Start the swarm learning process at the selected client
        result = flare.clients[first_aggregator_idx].swarm_round(
            model=self.initial_model,
            current_round=0,
            num_rounds=self.num_rounds
        )
        
        print("\n=== Swarm Learning Complete ===")
        if result and len(result) > 0:
            final_model = result[0]
            print(f"Final aggregated model received: {list(final_model.keys())}")
            return final_model
        else:
            print("No final result received")
            return None


# ============================================================================
# Client Implementation (handles both training and aggregation)
# ============================================================================

@flare.client
class SwarmClient:
    """
    Swarm learning client that can both train and aggregate.
    
    Each client:
    - Can train on local data when requested
    - Can aggregate results from all clients when selected as aggregator
    - Passes control to the next randomly selected aggregator
    """
    
    def __init__(self, client_id, learning_rate=0.01, local_epochs=1):
        self.client_id = client_id
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        
        # Simple model: just a linear layer for demonstration
        self.model = nn.Linear(10, 5)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        
        self.total_rounds_aggregated = 0
        self.total_rounds_trained = 0
    
    @flare.collab
    def train_local(self, model_weights, current_round):
        """
        Train the model on local data.
        
        This method is called by the aggregator client to request training
        from all clients (including itself).
        """
        # Load weights into model
        self._load_weights(model_weights)
        
        # Simulate local training
        for epoch in range(self.local_epochs):
            # Generate synthetic data
            x = torch.randn(32, 10)  # Batch size 32, input dim 10
            labels = torch.randint(0, 5, (32,))  # 5 classes
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        
        self.total_rounds_trained += 1
        
        # Return updated weights
        updated_weights = self._extract_weights()
        print(f"  Client {self.client_id}: Completed local training for round {current_round}, loss={loss.item():.4f}")
        
        return updated_weights
    
    @flare.collab
    def swarm_round(self, model, current_round, num_rounds):
        """
        Execute one round of swarm learning as the aggregator.
        
        Steps:
        1. Request all clients (including self) to train
        2. Aggregate the results
        3. If not final round, select next aggregator and pass control
        4. If final round, broadcast final model and return
        """
        print(f"\n--- Round {current_round}: Client {self.client_id} is the aggregator ---")
        
        # Step 1: Request all clients to train on current model
        print(f"  Aggregator {self.client_id}: Requesting training from all clients...")
        training_results = flare.clients.train_local(model, current_round)
        
        # Step 2: Aggregate results (simple averaging)
        print(f"  Aggregator {self.client_id}: Aggregating results from {len(training_results)} clients...")
        aggregated_model = self._aggregate(training_results)
        self.total_rounds_aggregated += 1
        
        # Step 3: Check if training is complete
        if current_round >= num_rounds - 1:
            print(f"\n  Aggregator {self.client_id}: Training complete! Broadcasting final model...")
            
            # Broadcast final model to all clients
            flare.clients.receive_final_model(aggregated_model)
            
            return aggregated_model
        
        # Step 4: Select next aggregator and continue
        next_round = current_round + 1
        # Randomly select next aggregator (could be any client including self)
        next_aggregator_idx = random.randint(0, 2)  # Assuming 3 clients
        
        print(f"  Aggregator {self.client_id}: Passing to client {next_aggregator_idx} for round {next_round}")
        
        # Pass control to next aggregator
        result = flare.clients[next_aggregator_idx].swarm_round(
            model=aggregated_model,
            current_round=next_round,
            num_rounds=num_rounds
        )
        
        # Return the final result up the chain
        if result and len(result) > 0:
            return result[0]
        return None
    
    @flare.collab
    def receive_final_model(self, final_model):
        """
        Receive the final trained model.
        
        In a real system, clients would save this model for inference.
        """
        self._load_weights(final_model)
        print(f"  Client {self.client_id}: Received final model")
        return {"status": "received"}
    
    @flare.collab
    def get_stats(self):
        """Return training statistics."""
        return {
            "client_id": self.client_id,
            "rounds_as_aggregator": self.total_rounds_aggregated,
            "rounds_as_trainer": self.total_rounds_trained,
        }
    
    # Helper methods
    
    def _extract_weights(self):
        """Extract model weights as a dictionary."""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def _load_weights(self, weights):
        """Load weights into the model."""
        for name, param in self.model.named_parameters():
            if name in weights:
                param.data.copy_(weights[name])
    
    def _aggregate(self, client_results):
        """
        Aggregate model weights from multiple clients using averaging.
        
        This is the same as FedAvg aggregation, but executed by a peer client
        rather than a central server.
        """
        if not client_results:
            return {}
        
        aggregated = {}
        
        # Get all parameter names from first result
        param_names = list(client_results[0].keys())
        
        # Average each parameter across all clients
        for param_name in param_names:
            param_sum = None
            count = 0
            
            for client_result in client_results:
                if param_name in client_result:
                    if param_sum is None:
                        param_sum = client_result[param_name].clone()
                    else:
                        param_sum += client_result[param_name]
                    count += 1
            
            if param_sum is not None and count > 0:
                aggregated[param_name] = param_sum / count
        
        return aggregated


# ============================================================================
# Main execution
# ============================================================================

def main():
    """
    Setup and execute swarm learning with multiple clients.
    
    Key characteristics:
    - 3 clients, each can be both trainer and aggregator
    - Aggregator role rotates randomly each round
    - Server only coordinates, never sees model weights
    - Fully decentralized after initialization
    """
    
    # Initialize model weights (in practice, one client might start with this)
    initial_model = {
        "weight": torch.randn(5, 10),
        "bias": torch.randn(5),
    }
    
    recipe = FlareRecipe(
        name="pt_swarm_learning",
        server=SwarmServer(initial_model=initial_model, num_rounds=5),
        clients=[
            SwarmClient(client_id=0, learning_rate=0.01, local_epochs=1)
        ],
    )
    
    # Use 3 clients and 3 threads for swarm learning
    env = SimEnv(num_clients=3, num_threads=3)
    run = recipe.execute(env)
    
    print(f"\n=== Execution Complete ===")
    print(f"Final result: {run.get_result()}")
    
    # Get statistics from all clients
    print("\n=== Client Statistics ===")
    # Note: In the current API, we'd need to add this capability
    # For now, the stats are implicitly tracked during execution


if __name__ == "__main__":
    main()

