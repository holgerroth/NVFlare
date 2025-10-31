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
PyTorch Split Learning - Peer-to-Peer Implementation

This example demonstrates PEER-TO-PEER split learning between two clients:
- Client 0 (front client) holds the front layers of the model (feature extraction)
- Client 1 (back client) holds the back layers of the model (classification/prediction)

Key Difference from pt_split_learning.py:
- **Server**: Only initiates the process, doesn't coordinate every step
- **Clients**: Directly call each other's @flare.collab methods
- **Communication**: Peer-to-peer between clients without server mediation

Training Flow (Peer-to-Peer):
1. Server tells front client to start training
2. Front client performs forward pass and DIRECTLY calls back client
3. Back client computes loss, backpropagates, and DIRECTLY calls front client
4. Front client performs backward pass
5. Repeat without server involvement

This demonstrates the flexibility of FOX's @flare.collab decorator for 
truly decentralized communication patterns.
"""
import torch
import torch.nn as nn

from nvflare.fox.api import flare
from nvflare.fox.sys.recipe import FlareRecipe
from nvflare.fox.sim.simulator import SimEnv


# ============================================================================
# Server Implementation - Minimal, Just Initiates
# ============================================================================

class SplitLearningP2PServer():
    def __init__(self, num_rounds=3, num_batches=5):
        self.num_rounds = num_rounds
        self.num_batches = num_batches

    @flare.algo
    def run_split_learning_p2p(self):
        """
        Initiate peer-to-peer split learning.
        
        Unlike traditional split learning where the server coordinates every step,
        here the server only:
        1. Initiates the training process
        2. Collects final statistics
        
        All forward/backward communication happens DIRECTLY between clients.
        """
        print(f"System info: {flare.sys_info()}")
        print(f"\n=== Starting Peer-to-Peer Split Learning ===")
        print(f"Training for {self.num_rounds} rounds with {self.num_batches} batches per round")
        print(f"Server role: Minimal - only initiating and collecting stats")
        
        assert len(flare.clients) == 2, "Split learning requires 2 clients"
        front_client = flare.clients[0]
        
        # Tell the front client to start training
        # The front client will then coordinate with the back client directly
        print(f"\n=== Server: Initiating training on front client ===")
        front_client.start_training(
            back_client_idx=1,  # Tell front client which client is the back client
            num_rounds=self.num_rounds,
            num_batches=self.num_batches
        )
        
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
# Client Implementations - Direct Peer-to-Peer Communication
# ============================================================================


class FrontClientP2P:
    """
    Front client holds the bottom layers of the split model.
    
    Key difference from traditional approach:
    - DIRECTLY calls back client's methods (no server mediation)
    - Coordinates its own training loop
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
    def start_training(self, back_client_idx, num_rounds, num_batches):
        """
        Start the training process and coordinate DIRECTLY with back client.
        
        This method controls the entire training loop and calls the back client
        directly without going through the server for each step.
        """
        print(f"\n[Front Client] Starting peer-to-peer training")
        print(f"[Front Client] Will communicate directly with client {back_client_idx}")
        
        # Get reference to back client
        
        for round_idx in range(num_rounds):
            print(f"\n=== Round {round_idx} ===")
            
            for batch_idx in range(num_batches):
                # Step 1: Perform forward pass
                activations = self._forward_pass(round_idx, batch_idx)
                
                # Step 2: DIRECTLY call back client (no server mediation!)
                back_results = flare.other_clients[0].forward_and_backward(
                    round_idx, 
                    batch_idx, 
                    activations
                )
                
                gradients = back_results['gradients']
                loss = back_results['loss']
                
                # Step 3: Perform backward pass with received gradients
                self._backward_pass(gradients)
                
                if batch_idx % 2 == 0:
                    print(f"[Front Client] Round {round_idx}, Batch {batch_idx}: Loss = {loss:.4f}")
        
        print(f"\n[Front Client] Training complete!")
        return {"status": "completed"}
    
    def _forward_pass(self, round_idx, batch_idx):
        """
        Internal method: Perform forward pass through front layers.
        """
        # Generate synthetic input data
        x = torch.randn(32, self.input_dim)  # Batch size of 32
        
        # Forward pass
        self.model.train()
        activations = self.model(x)
        
        # Save activations for backward pass (need to keep gradients)
        self.saved_activations = activations.clone().detach().requires_grad_(True)
        
        return self.saved_activations.detach()
    
    def _backward_pass(self, gradients):
        """
        Internal method: Perform backward pass with received gradients.
        """
        if self.saved_activations is None:
            print(f"[Front Client] Warning: No saved activations for backward pass")
            return
        
        # Backward pass through front layers
        self.optimizer.zero_grad()
        self.saved_activations.backward(gradients)
        self.optimizer.step()
        
        self.total_batches += 1
        self.saved_activations = None  # Clear saved activations
    
    @flare.collab
    def get_stats(self):
        """Return training statistics."""
        return {
            "total_batches_processed": self.total_batches,
            "model_type": "front_layers",
            "learning_rate": self.learning_rate,
            "communication_pattern": "peer-to-peer"
        }


  
class BackClientP2P:
    """
    Back client holds the top layers of the split model.
    
    Key difference from traditional approach:
    - Called DIRECTLY by front client (no server mediation)
    - Can optionally call front client directly if needed
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
        Receive activations DIRECTLY from front client, complete forward pass,
        compute loss, and return gradients.
        
        This is called directly by the front client without server involvement.
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
            "learning_rate": self.learning_rate,
            "communication_pattern": "peer-to-peer"
        }


# ============================================================================
# Main execution
# ============================================================================

def main():
    """
    Setup and execute peer-to-peer split learning between two clients.
    
    This demonstrates how clients can directly communicate with each other
    using @flare.collab decorators, without server mediation for every step.
    """
    recipe = FlareRecipe(
        name="pt_split_learning_p2p",
        server=SplitLearningP2PServer(num_rounds=3, num_batches=5),
        clients=[
            FrontClientP2P(input_dim=10, hidden_dim=20, learning_rate=0.01),
            BackClientP2P(hidden_dim=20, output_dim=5, learning_rate=0.01),
        ],
    )
    
    # Use 2 clients and 2 threads for split learning
    env = SimEnv(num_clients=2, num_threads=2)
    run = recipe.execute(env)
    
    print(f"\n=== Final Result ===")
    print(run.get_result())
    
    print(f"\n=== Key Takeaway ===")
    print("In this peer-to-peer version:")
    print("- Server ONLY initiated training (1 call)")
    print("- Front client DIRECTLY called back client methods")
    print("- No server mediation for forward/backward passes")
    print("- Demonstrates true peer-to-peer @flare.collab communication")


if __name__ == "__main__":
    main()

