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
PyTorch Split Learning - Two-client split learning implementation.

This example demonstrates split learning between two clients:
- Client 1 holds the front layers of the model (feature extraction)
- Client 2 holds the back layers of the model (classification/prediction)

The training proceeds as follows:
1. Client 1 performs forward pass and sends activations to Client 2
2. Client 2 performs forward pass, computes loss, and backpropagates
3. Client 2 sends gradients back to Client 1
4. Client 1 performs backward pass and updates its weights
"""
import torch
import torch.nn as nn

from nvflare.fox.api import flare
from nvflare.fox.sys.recipe import FlareRecipe
from nvflare.fox.sim.simulator import SimEnv


# ============================================================================
# Server Implementation 
# ============================================================================

@flare.server
class SplitLearningServer():
    def __init__(self, num_rounds=3, num_batches=5):
        self.num_rounds = num_rounds
        self.num_batches = num_batches

    @flare.main
    def run_split_learning(self):
        """
        Coordinate split learning between two clients.
        
        In split learning, the model is split vertically:
        - Client 0 (front client) has the bottom layers
        - Client 1 (back client) has the top layers
        
        Training flow:
        1. Front client computes forward pass -> activations
        2. Back client receives activations, completes forward pass, computes loss
        3. Back client computes gradients and sends back to front client
        4. Front client completes backward pass
        """
        print(f"System info: {flare.sys_info()}")
        print(f"\n=== Starting Split Learning ===")
        print(f"Training for {self.num_rounds} rounds with {self.num_batches} batches per round")
        
        assert len(flare.clients) == 2, "Split learning requires 2 clients"
        front_client = flare.clients[0]
        back_client = flare.clients[1]

        for round_idx in range(self.num_rounds):
            print(f"\n=== Round {round_idx} ===")
            
            for batch_idx in range(self.num_batches):
                # Step 1: Front client (client 0) performs forward pass
                front_results = front_client.forward_pass(round_idx, batch_idx)
                
                if not front_results or len(front_results) == 0:
                    print(f"Round {round_idx}, Batch {batch_idx}: No results from front client")
                    continue
                    
                activations = front_results[0]['activations']
                
                # Step 2: Back client (client 1) receives activations, computes forward pass and loss
                back_results = back_client.forward_and_backward(round_idx, batch_idx, activations)
                
                if not back_results or len(back_results) == 0:
                    print(f"Round {round_idx}, Batch {batch_idx}: No results from back client")
                    continue
                    
                gradients = back_results[0]['gradients']
                loss = back_results[0]['loss']
                
                # Step 3: Front client receives gradients and performs backward pass
                front_client.backward_pass(round_idx, batch_idx, gradients)
                
                if batch_idx % 2 == 0:
                    print(f"Round {round_idx}, Batch {batch_idx}: Loss = {loss:.4f}")
        
        # Get final statistics from both clients
        print("\n=== Training Complete ===")
        print("Getting final model statistics from clients...")
        
        stats = flare.clients.get_stats()
        for i, client_stats in enumerate(stats):
            print(f"\nClient {i} stats:")
            for key, value in client_stats.items():
                print(f"  {key}: {value}")
        
        return {"status": "completed", "rounds": self.num_rounds}


# ============================================================================
# Client Implementations 
# ============================================================================

@flare.client
class FrontClient:
    """
    Front client holds the bottom layers of the split model.
    Performs forward pass and receives gradients for backward pass.
    """
    
    def __init__(self, input_dim=10, hidden_dim=20, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Create front part of the model (bottom layers)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        
        # Store activations for backward pass
        self.saved_activations = None
        self.total_batches = 0
    
    @flare.collab
    def forward_pass(self, round_idx, batch_idx):
        """
        Perform forward pass through front layers and return activations.
        """
        # Generate synthetic input data
        x = torch.randn(32, self.input_dim)  # Batch size of 32
        
        # Forward pass
        self.model.train()
        activations = self.model(x)
        
        # Save activations for backward pass (need to keep gradients)
        self.saved_activations = activations.clone().detach().requires_grad_(True)
        
        return {
            'activations': self.saved_activations.detach()
        }
    
    @flare.collab
    def backward_pass(self, round_idx, batch_idx, gradients):
        """
        Receive gradients from back client and perform backward pass.
        """
        if self.saved_activations is None:
            print(f"Warning: No saved activations for backward pass")
            return {}
        
        # Backward pass through front layers
        self.optimizer.zero_grad()
        self.saved_activations.backward(gradients)
        self.optimizer.step()
        
        self.total_batches += 1
        self.saved_activations = None  # Clear saved activations
        
        return {"status": "completed"}
    
    @flare.collab
    def get_stats(self):
        """Return training statistics."""
        return {
            "total_batches_processed": self.total_batches,
            "model_type": "front_layers",
            "learning_rate": self.learning_rate
        }


@flare.client  
class BackClient:
    """
    Back client holds the top layers of the split model.
    Receives activations, computes loss, and sends gradients back.
    """
    
    def __init__(self, hidden_dim=20, output_dim=5, learning_rate=0.01):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # Create back part of the model (top layers)
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.total_loss = 0.0
        self.total_batches = 0
    
    @flare.collab
    def forward_and_backward(self, round_idx, batch_idx, activations):
        """
        Receive activations from front client, complete forward pass,
        compute loss, and return gradients.
        """
        # Prepare activations (they need to have gradients)
        activations = activations.requires_grad_(True)
        
        # Generate synthetic labels
        labels = torch.randint(0, self.output_dim, (activations.size(0),))
        
        # Forward pass through back layers
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(activations)
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Get gradients w.r.t. activations (to send back to front client)
        gradients = activations.grad.clone()
        
        # Update back client's model
        self.optimizer.step()
        
        self.total_loss += loss.item()
        self.total_batches += 1
        
        return {
            'gradients': gradients.detach(),
            'loss': loss.item()
        }
    
    @flare.collab
    def get_stats(self):
        """Return training statistics."""
        avg_loss = self.total_loss / max(1, self.total_batches)
        return {
            "total_batches_processed": self.total_batches,
            "average_loss": avg_loss,
            "model_type": "back_layers",
            "learning_rate": self.learning_rate
        }


# ============================================================================
# Main execution
# ============================================================================

def main():
    """
    Setup and execute split learning between two clients.
    """
    recipe = FlareRecipe(
        name="pt_split_learning",
        server=SplitLearningServer(num_rounds=3, num_batches=5),
        clients=[
            FrontClient(input_dim=10, hidden_dim=20, learning_rate=0.01),
            BackClient(hidden_dim=20, output_dim=5, learning_rate=0.01),
        ],
    )
    
    # Use 2 clients and 2 threads for split learning
    env = SimEnv(num_clients=2, num_threads=2)
    run = recipe.execute(env)
    
    print(f"\n=== Final Result ===")
    print(run.get_result())


if __name__ == "__main__":
    main()

