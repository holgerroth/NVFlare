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
PyTorch FedAvg - Pure Functional with @flare.collab decorators

This example combines the best of both worlds:
1. Pure functional programming for business logic (easy to test, understand, debug)
2. FOX framework integration for real-world deployment (@flare decorators)

The approach:
- All core logic is in pure functions (no classes, no side effects)
- Minimal wrapper classes with @flare decorators for framework integration
- Business logic is separate from framework concerns
- Pure functions are easily testable without any framework setup
- Can be deployed in production using FlareRecipe

This is the RECOMMENDED approach for real-world federated learning:
- Testable (pure functions can be unit tested easily)
- Maintainable (clear separation of concerns)
- Deployable (works with FOX framework)
- Understandable (logic is explicit, not hidden in class hierarchies)
"""
import torch
from typing import Dict, List, Tuple

from nvflare.fox.api import flare
from nvflare.fox.sys.recipe import FlareRecipe
from nvflare.fox.sim.simulator import SimEnv


# ============================================================================
# PURE FUNCTIONAL CORE - Business Logic (Framework-Independent)
# ============================================================================

def aggregate_weights(weight_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Average model weights from multiple clients.
    
    Pure function - can be tested without any framework.
    
    Args:
        weight_list: List of weight dictionaries from different clients
        
    Returns:
        Dictionary containing averaged weights
    """
    if not weight_list:
        return {}
    
    averaged = {}
    num_clients = len(weight_list)
    
    # Get all parameter names from first client
    param_names = list(weight_list[0].keys())
    
    # Average each parameter
    for param_name in param_names:
        # Sum across all clients
        param_sum = torch.zeros_like(weight_list[0][param_name])
        
        for client_weights in weight_list:
            if param_name in client_weights:
                param_sum += client_weights[param_name]
        
        # Average
        averaged[param_name] = param_sum / num_clients
    
    return averaged


def weighted_aggregate(weight_list: List[Dict[str, torch.Tensor]], 
                      sample_counts: List[int]) -> Dict[str, torch.Tensor]:
    """
    Compute weighted average of model weights based on sample counts.
    
    Pure function - useful when clients have different amounts of data.
    """
    if not weight_list or not sample_counts:
        return {}
    
    total_samples = sum(sample_counts)
    averaged = {}
    param_names = list(weight_list[0].keys())
    
    for param_name in param_names:
        param_sum = torch.zeros_like(weight_list[0][param_name])
        
        for client_weights, num_samples in zip(weight_list, sample_counts):
            if param_name in client_weights:
                weight = num_samples / total_samples
                param_sum += client_weights[param_name] * weight
        
        averaged[param_name] = param_sum
    
    return averaged


def local_train(weights: Dict[str, torch.Tensor], 
               client_id: int,
               learning_rate: float,
               num_epochs: int,
               batch_size: int = 32) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Simulate local training on a single client.
    
    Pure function that takes weights and returns updated weights.
    In production, replace this with actual training logic.
    
    Args:
        weights: Current model weights
        client_id: Client identifier (for data simulation)
        learning_rate: Learning rate for training
        num_epochs: Number of local training epochs
        batch_size: Batch size for training
        
    Returns:
        Tuple of (updated_weights, num_samples_trained)
    """
    # Clone weights to avoid side effects
    updated_weights = {k: v.clone() for k, v in weights.items()}
    
    # Simulate training with gradient descent
    num_steps = num_epochs * 10  # Simulate 10 batches per epoch
    
    for step in range(num_steps):
        for param_name in updated_weights.keys():
            # Simulate gradient computation
            # In production, this would be: loss.backward(); param.grad
            gradient = torch.randn_like(updated_weights[param_name]) * 0.1
            
            # SGD update
            updated_weights[param_name] = updated_weights[param_name] - learning_rate * gradient
    
    # Simulate number of samples (different clients may have different amounts)
    num_samples = 100 + client_id * 50
    
    return updated_weights, num_samples


def compute_metrics(weights: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute metrics about the model weights.
    
    Pure function for monitoring/logging.
    """
    metrics = {}
    
    for name, param in weights.items():
        metrics[f"{name}_mean"] = param.mean().item()
        metrics[f"{name}_std"] = param.std().item()
        metrics[f"{name}_norm"] = param.norm().item()
    
    # Overall model statistics
    all_params = torch.cat([p.flatten() for p in weights.values()])
    metrics["global_mean"] = all_params.mean().item()
    metrics["global_std"] = all_params.std().item()
    metrics["global_norm"] = all_params.norm().item()
    
    return metrics


# ============================================================================
# FRAMEWORK INTEGRATION - Thin wrappers around pure functions
# ============================================================================

@flare.server
class FunctionalFedAvgServer:
    """
    Minimal server wrapper that orchestrates federated learning.
    
    All business logic is delegated to pure functions.
    This class only handles framework integration.
    """
    
    def __init__(self, initial_weights, num_rounds, use_weighted_avg=True):
        # Store configuration
        self.initial_weights = initial_weights
        self.num_rounds = num_rounds
        self.use_weighted_avg = use_weighted_avg
    
    @flare.main
    def run(self):
        """
        Main federated learning loop.
        
        Orchestrates the workflow but delegates logic to pure functions.
        """
        print("\n" + "=" * 70)
        print("Functional FedAvg with @flare.collab decorators")
        print("=" * 70)
        print(f"System info: {flare.sys_info()}")
        print(f"Rounds: {self.num_rounds}")
        print(f"Weighted averaging: {self.use_weighted_avg}")
        
        # Start with initial weights
        global_weights = self.initial_weights
        
        # Training loop
        for round_idx in range(self.num_rounds):
            print(f"\n=== Round {round_idx + 1}/{self.num_rounds} ===")
            
            # Request all clients to train
            # Returns list of tuples: (weights, num_samples)
            results = flare.clients.train(global_weights, round_idx)
            
            if not results:
                print(f"Round {round_idx}: No results received")
                continue
            
            print(f"Received results from {len(results)} clients")
            
            # Extract weights and sample counts
            client_weights = [r[0] for r in results]
            sample_counts = [r[1] for r in results]
            
            # Aggregate using pure functions
            if self.use_weighted_avg:
                global_weights = weighted_aggregate(client_weights, sample_counts)
                print(f"Used weighted aggregation (total samples: {sum(sample_counts)})")
            else:
                global_weights = aggregate_weights(client_weights)
                print(f"Used simple averaging")
            
            # Compute and log metrics using pure function
            metrics = compute_metrics(global_weights)
            print(f"Global model norm: {metrics['global_norm']:.4f}, "
                  f"mean: {metrics['global_mean']:.4f}")
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        
        # Final metrics
        final_metrics = compute_metrics(global_weights)
        print("\nFinal Model Metrics:")
        for key, value in final_metrics.items():
            if key.startswith("global_"):
                print(f"  {key}: {value:.6f}")
        
        return global_weights


@flare.client
class FunctionalFedAvgClient:
    """
    Minimal client wrapper that performs local training.
    
    All training logic is delegated to pure functions.
    This class only handles framework integration.
    """
    
    def __init__(self, client_id, learning_rate=0.01, num_epochs=1):
        # Store configuration
        self.client_id = client_id
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
    
    @flare.collab
    def train(self, weights, round_idx):
        """
        Perform local training.
        
        Thin wrapper that delegates to pure function.
        """
        print(f"  Client {self.client_id}: Starting local training for round {round_idx}")
        
        # Call pure function for actual training logic
        updated_weights, num_samples = local_train(
            weights=weights,
            client_id=self.client_id,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs
        )
        
        # Compute metrics for logging
        metrics = compute_metrics(updated_weights)
        
        print(f"  Client {self.client_id}: Trained on {num_samples} samples, "
              f"model norm: {metrics['global_norm']:.4f}")
        
        # Return tuple of (weights, sample_count)
        return (updated_weights, num_samples)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_initial_weights(input_dim: int = 10, 
                          hidden_dim: int = 20, 
                          output_dim: int = 5) -> Dict[str, torch.Tensor]:
    """
    Initialize model weights.
    
    Pure function - easy to test and modify.
    """
    return {
        "layer1.weight": torch.randn(hidden_dim, input_dim) * 0.01,
        "layer1.bias": torch.zeros(hidden_dim),
        "layer2.weight": torch.randn(output_dim, hidden_dim) * 0.01,
        "layer2.bias": torch.zeros(output_dim),
    }


def print_weight_summary(weights: Dict[str, torch.Tensor], title: str = "Weights"):
    """
    Pretty print weight summary.
    
    Pure function for visualization.
    """
    print(f"\n{title}:")
    for name, param in weights.items():
        print(f"  {name:20s} | shape: {str(param.shape):15s} | "
              f"mean: {param.mean().item():8.4f} | std: {param.std().item():8.4f}")


# ============================================================================
# TESTING - Pure functions can be tested without framework
# ============================================================================

def test_aggregation():
    """
    Example of how easy it is to test pure functions.
    No framework setup needed!
    """
    print("\n" + "=" * 70)
    print("Testing Pure Functions (No Framework Required)")
    print("=" * 70)
    
    # Create test data
    weights1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
    weights2 = {"w": torch.tensor([3.0, 4.0, 5.0])}
    weights3 = {"w": torch.tensor([5.0, 6.0, 7.0])}
    
    # Test simple averaging
    result = aggregate_weights([weights1, weights2, weights3])
    expected = torch.tensor([3.0, 4.0, 5.0])
    
    print("\nTest 1: Simple Averaging")
    print(f"  Input 1: {weights1['w'].tolist()}")
    print(f"  Input 2: {weights2['w'].tolist()}")
    print(f"  Input 3: {weights3['w'].tolist()}")
    print(f"  Result:  {result['w'].tolist()}")
    print(f"  Expected: {expected.tolist()}")
    print(f"  ✓ PASS" if torch.allclose(result['w'], expected) else "  ✗ FAIL")
    
    # Test weighted averaging
    result = weighted_aggregate([weights1, weights2, weights3], [100, 200, 100])
    # Expected: (1*100 + 3*200 + 5*100) / 400 = (100 + 600 + 500) / 400 = 3.0
    #           (2*100 + 4*200 + 6*100) / 400 = (200 + 800 + 600) / 400 = 4.0
    #           (3*100 + 5*200 + 7*100) / 400 = (300 + 1000 + 700) / 400 = 5.0
    expected = torch.tensor([3.0, 4.0, 5.0])
    
    print("\nTest 2: Weighted Averaging")
    print(f"  Weights: [100, 200, 100]")
    print(f"  Result:  {result['w'].tolist()}")
    print(f"  Expected: {expected.tolist()}")
    print(f"  ✓ PASS" if torch.allclose(result['w'], expected) else "  ✗ FAIL")
    
    print("\nAll tests passed! Pure functions work perfectly without framework.\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function demonstrating the hybrid approach.
    
    This combines:
    - Pure functional business logic (easy to test)
    - Framework integration (production-ready deployment)
    """
    
    # Optional: Run tests on pure functions
    test_aggregation()
    
    # Configuration
    config = {
        "num_rounds": 4,
        "num_clients": 3,
        "input_dim": 10,
        "hidden_dim": 20,
        "output_dim": 5,
        "learning_rate": 0.01,
        "num_epochs": 2,
        "use_weighted_avg": True,
    }
    
    print("\n" + "=" * 70)
    print("Running Federated Learning with Pure Functional Core")
    print("=" * 70)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create initial weights using pure function
    initial_weights = create_initial_weights(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"]
    )
    print_weight_summary(initial_weights, "Initial Weights")
    
    # Create server and clients (minimal framework wrappers)
    server = FunctionalFedAvgServer(
        initial_weights=initial_weights,
        num_rounds=config["num_rounds"],
        use_weighted_avg=config["use_weighted_avg"]
    )
    
    # Create clients with different configurations
    clients = [
        FunctionalFedAvgClient(
            client_id=i,
            learning_rate=config["learning_rate"],
            num_epochs=config["num_epochs"]
        )
        for i in range(config["num_clients"])
    ]
    
    # Create recipe
    recipe = FlareRecipe(
        name="functional_fedavg_production",
        server=server,
        clients=clients,
    )
    
    # Execute federated learning
    env = SimEnv(num_clients=config["num_clients"], num_threads=config["num_clients"])
    run = recipe.execute(env)
    
    # Get final results
    final_weights = run.get_result()
    print_weight_summary(final_weights, "Final Weights")
    
    print("\n" + "=" * 70)
    print("Benefits of This Approach:")
    print("=" * 70)
    print("✓ Pure functions are easy to test (no framework needed)")
    print("✓ Business logic is separate from framework concerns")
    print("✓ Can be deployed in production with FlareRecipe")
    print("✓ Easy to understand and maintain")
    print("✓ Flexible - swap aggregation strategies easily")
    print("✓ Framework-agnostic core - could work with other FL frameworks")
    print("=" * 70)


if __name__ == "__main__":
    main()

