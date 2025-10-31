# Federated Learning Approaches - Examples Guide

This directory contains examples of different federated learning algorithms and programming styles.

---

## FL Algorithms

### 1. Federated Averaging (FedAvg) - `pt_fedavg.py`

**What it is:** Standard centralized federated learning

**How it works:**
```
Server broadcasts model → Clients train locally → Clients send updates back → 
Server aggregates → Repeat
```

**Key Features:**
- Central server coordinates everything
- Most common FL approach
- Simple and efficient

**Use Cases:**
- Standard FL scenarios
- Cross-silo FL (hospitals, organizations)
- When you trust the central server


### 2. Split Learning - `pt_split_learning.py`

**What it is:** Model split between two parties for privacy

**How it works:**
```
Client 1 (bottom layers) → forward pass → sends activations →
Client 2 (top layers) → forward pass → computes loss → backprop → sends gradients →
Client 1 → backprop → updates weights
```

**Key Features:**
- Model never fully shared between parties
- Strong privacy guarantees
- Sequential processing (slower)

**Use Cases:**
- High privacy requirements
- Medical imaging (raw data stays local)
- When parties can't share full model

---

### 3. Swarm Learning - `pt_swarm_learning.py`

**What it is:** Decentralized P2P learning without trusted server

**How it works:**
```
Round 1: Client A is aggregator → All clients train → Send to Client A → 
         Client A aggregates
Round 2: Client B becomes aggregator → All clients train → Send to Client B → 
         Client B aggregates
(Aggregator rotates each round)
```

**Key Features:**
- No central aggregator needed
- Aggregator role rotates among clients
- Server never sees model weights
- Fully decentralized

**Use Cases:**
- Untrusted central server
- Blockchain/distributed systems
- Privacy-sensitive applications

---

## Programming Styles

### Class-Based (`pt_fedavg.py`)

Simple OOP approach - good for learning:

```python
class MyFedAvg:
    def __init__(self, initial_model, num_rounds):
        self.initial_model = initial_model
        self.num_rounds = num_rounds
    
    @flare.algo
    def run_fedavg(self):
        # Logic here
        pass
```

**Best for:** Learning, prototyping, simple examples

---

### Functional + @flare.collab (`pt_fedavg_functional_collab.py`) ⭐ RECOMMENDED

Pure functions + thin wrappers - best for production:

```python
# Pure function - easy to test
def aggregate_weights(weight_list):
    """No framework dependencies"""
    # Business logic here
    return averaged

# Thin wrapper - FOX integration
class FunctionalFedAvgServer:
    @flare.algo
    def run(self):
        # Delegates to pure function
        global_weights = aggregate_weights(client_weights)
```

**Best for:** Production, testing, maintainability

**Why it's better:**
- ✓ Test pure functions without framework
- ✓ Business logic is reusable
- ✓ Easy to swap algorithms
- ✓ Clear separation of concerns

---

## Quick Comparison

| Feature | FedAvg | Split Learning | Swarm Learning |
|---------|--------|----------------|----------------|
| **Architecture** | Centralized | 2-party | P2P |
| **Server Trust** | Required | Low | None needed |
| **Privacy** | Medium | High | High |
| **Complexity** | Low | Medium | High |
| **Best For** | Standard FL | Privacy-critical | Decentralized |

---

## Which Should I Use?

### For Algorithm:

**Choose FedAvg** if:
- You trust the central server
- You want simplicity
- Standard FL is enough

**Choose Split Learning** if:
- Maximum privacy needed
- Working with 2 parties
- Can't share full model

**Choose Swarm Learning** if:
- Can't trust central server
- Want decentralization
- Need fault tolerance

### For Programming Style:

**Choose Class-Based** if:
- Learning the framework
- Building quick prototypes

**Choose Functional** if: ⭐
- Building production systems
- Testing is important
- Want maintainable code

---

## Running Examples

```bash
# FedAvg - class-based
python pt_fedavg.py

# FedAvg - functional (RECOMMENDED for production)
python pt_fedavg_functional_collab.py

# FedAvg - async variant
python pt_fedavg_async.py

# Split Learning
python pt_split_learning.py

# Swarm Learning
python pt_swarm_learning.py
```

---

## Testing Example

With functional approach, testing is trivial:

```python
from pt_fedavg_functional_collab import aggregate_weights
import torch

# No framework setup needed!
weights1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
weights2 = {"w": torch.tensor([3.0, 4.0, 5.0])}

result = aggregate_weights([weights1, weights2])
# result["w"] == tensor([2.0, 3.0, 4.0])
```

---

## More Information

- **Architecture details:** See `ARCHITECTURE_GUIDE.md`
- **Full overview:** See `README.md`

