# Custom Aggregator Example

This directory demonstrates how to use custom aggregators with NVFlare's `FedAvgRecipe`.

## Overview

The `job.py` file provides a complete example of running federated learning with custom aggregation strategies. Two custom aggregators are implemented in `custom_aggregators.py`:

### 1. **WeightedAggregator**
Weights each client's contribution by their number of training steps (or dataset size). This is more fair when clients have different amounts of data.

**Use case**: When clients have heterogeneous dataset sizes and you want to weight their contributions proportionally.

### 2. **MedianAggregator**
Computes the element-wise median of all client models instead of averaging. This provides robustness against Byzantine (malicious) clients.

**Use case**: When you need protection against adversarial clients who might send malicious model updates.

## Usage

### Basic Usage

Run with weighted aggregator (default):
```bash
python job.py --aggregator weighted --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

Run with median aggregator:
```bash
python job.py --aggregator median --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

Run with default FedAvg aggregator (for comparison):
```bash
python job.py --aggregator default --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

**Important**: Use the same `--seed` value to ensure identical model initialization across experiments!

### Command-line Arguments

#### Aggregator Selection
- `--aggregator {weighted,median,default}` - Choose aggregation strategy (default: `weighted`)

#### Federated Learning Parameters
- `--n_clients` - Number of federated learning clients (default: `8`)
- `--num_rounds` - Number of FL rounds (default: `50`)
- `--alpha` - Data heterogeneity parameter (default: `0.5`)
  - Higher values (e.g., 1.0) = more uniform data distribution
  - Lower values (e.g., 0.1) = more heterogeneous/non-IID distribution
- `--seed` - Random seed for model initialization and reproducibility (default: `0`)
  - Sets random seeds for Python, NumPy, PyTorch (CPU & CUDA), and makes CUDNN deterministic
  - **Important**: Use the same seed across experiments for fair comparison!

#### Training Parameters
- `--aggregation_epochs` - Local epochs per round (default: `4`)
- `--lr` - Learning rate (default: `0.05`)
- `--batch_size` - Training batch size (default: `64`)
- `--num_workers` - Data loading workers (default: `2`)

#### Other Options
- `--name` - Custom job name (default: auto-generated based on aggregator and alpha)

## Examples

### Compare All Three Aggregators

Run these commands to compare the performance of different aggregators on highly heterogeneous data (alpha=0.1):

```bash
# Default FedAvg aggregator (baseline)
python job.py --aggregator default --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0

# Weighted aggregator
python job.py --aggregator weighted --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0

# Median aggregator (Byzantine-robust)
python job.py --aggregator median --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

All three runs will use the same seed, ensuring identical model initialization and data splits for fair comparison.

### Run on Different GPUs (Parallel Execution)

You can run experiments in parallel on different GPUs:

Terminal 1:
```bash
export CUDA_VISIBLE_DEVICES=0
python job.py --aggregator weighted --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

Terminal 2:
```bash
export CUDA_VISIBLE_DEVICES=1
python job.py --aggregator median --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

Terminal 3:
```bash
export CUDA_VISIBLE_DEVICES=2
python job.py --aggregator default --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

## Implementation Details

### Custom Aggregator Structure

Each custom aggregator must implement three key methods:

```python
class CustomAggregator(Aggregator):
    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Accept and accumulate client model updates"""
        pass
    
    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Perform aggregation and return aggregated model"""
        pass
    
    def reset(self, fl_ctx: FLContext):
        """Reset state for next round"""
        pass
```

### Key Concepts

1. **Shareable & DXO**: NVFlare uses these objects to exchange data between clients and server
   - `Shareable`: Container for data exchange
   - `DXO` (Data eXchange Object): Structured data with metadata

2. **DataKind**: Specifies what type of data is being exchanged
   - `DataKind.WEIGHTS`: Full model weights
   - `DataKind.WEIGHT_DIFF`: Model weight differences (used in this example)

3. **FLContext**: Provides context information about the federated learning process

## Viewing Results

After running a job, view the training curves with TensorBoard:

```bash
tensorboard --logdir=/tmp/nvflare/simulation
```

Then open http://localhost:6006 in your browser.

## Extending This Example

To create your own custom aggregator:

1. Create a new class inheriting from `Aggregator` in `custom_aggregators.py`
2. Implement the three required methods: `accept()`, `aggregate()`, `reset()`
3. Add your aggregator to the `get_aggregator()` function in `job.py`
4. Add it to the `--aggregator` choices in `define_parser()`

Example skeleton:

```python
class MyCustomAggregator(Aggregator):
    def __init__(self, custom_param=1.0):
        super().__init__()
        self.custom_param = custom_param
        # Initialize your state variables
    
    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        # Process incoming client model
        dxo = from_shareable(shareable)
        if dxo.data_kind in [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]:
            # Your custom logic here
            return True
        return False
    
    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        # Your custom aggregation logic
        aggregated_params = {}  # Compute this
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=aggregated_params)
        return dxo.to_shareable()
    
    def reset(self, fl_ctx: FLContext):
        # Reset your state variables
        pass
```

## References

- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [FedAvg Paper](https://arxiv.org/abs/1602.05629)
- [Byzantine-Robust Aggregation](https://arxiv.org/abs/1803.01498)

