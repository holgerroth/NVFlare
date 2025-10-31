# Research Examples Summary

This directory contains comprehensive examples of federated learning implementations using different approaches and algorithms.

## Quick Start

**Recommended for Production** ⭐:
```bash
python pt_fedavg_functional_collab.py
```

---

## Files Overview

### 1. Federated Learning Algorithms

| File | Algorithm | Description |
|------|-----------|-------------|
| `pt_fedavg.py` (or `pt_avg_research_api.py`) | FedAvg | Standard federated averaging - centralized server aggregates client models |
| `pt_split_learning.py` | Split Learning | Model split between 2 clients - sequential training with activations/gradients |
| `pt_swarm_learning.py` | Swarm Learning | Decentralized P2P learning - rotating aggregators, no trusted server |

### 2. Programming Approaches

| File | Approach | Framework | Best For |
|------|----------|-----------|----------|
| `pt_fedavg_functional_collab.py` ⭐ | Pure functions + @flare.collab | FOX | **Production** (Recommended) |
| `pt_fedavg.py` | Traditional class-based | FOX | Learning FOX framework |

### 3. Documentation

| File | Description |
|------|-------------|
| `README_LEARNING_APPROACHES.md` | Comprehensive guide to all FL approaches and programming styles |
| `ARCHITECTURE_GUIDE.md` | Detailed explanation of why functional+collab is recommended |

---

## Which File Should I Use?

### I want to...

**Build a production FL system:**
→ `pt_fedavg_functional_collab.py` ⭐
- Best practices
- Testable business logic
- Production-ready deployment

**Learn the FOX framework:**
→ `pt_fedavg.py`
- Shows framework patterns
- Traditional class-based approach

**Implement split learning:**
→ `pt_split_learning.py`
- Vertical model partitioning
- Privacy-preserving training

**Implement swarm learning:**
→ `pt_swarm_learning.py`
- Decentralized P2P approach
- Rotating aggregators

---

## Architecture Philosophy

### The Functional + @flare.collab Approach

```
Pure Functions              Thin Framework Wrappers
(Business Logic)            (Deployment)
───────────────────         ───────────────────────
aggregate_weights()    →    
local_train()          →    
compute_metrics()           @flare.collab
```

**Why this is best:**
- ✅ Business logic is framework-independent
- ✅ Easy to test without setup
- ✅ Clear separation of concerns
- ✅ Production-ready deployment
- ✅ Can swap FL frameworks easily

See `ARCHITECTURE_GUIDE.md` for detailed explanation.

---

## Running Examples

### FedAvg (Different Styles)
```bash
# Recommended for production
python pt_fedavg_functional_collab.py

# Traditional class-based approach
python pt_fedavg.py
```

### Other Algorithms
```bash
# Split Learning
python pt_split_learning.py

# Swarm Learning  
python pt_swarm_learning.py
```

---

## Code Examples

### Testing with Functional Approach

The functional approach makes testing trivial:

```python
from pt_fedavg_functional_collab import aggregate_weights
import torch

def test_aggregation():
    """Test without ANY framework setup!"""
    weights1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
    weights2 = {"w": torch.tensor([3.0, 4.0, 5.0])}
    
    result = aggregate_weights([weights1, weights2])
    expected = torch.tensor([2.0, 3.0, 4.0])
    
    assert torch.allclose(result["w"], expected)
    print("✓ Test passed!")

test_aggregation()
```

### Swapping Aggregation Strategies

Pure functions make it easy to swap algorithms:

```python
# Easy to swap between strategies
if use_weighted:
    global_weights = weighted_aggregate(client_weights, sample_counts)
else:
    global_weights = aggregate_weights(client_weights)

# Or use a completely different aggregation
from my_custom_algos import median_aggregate
global_weights = median_aggregate(client_weights)
```

---

## Key Takeaways

1. **For Production**: Use `pt_fedavg_functional_collab.py` pattern ⭐
   - Pure business logic functions
   - Thin framework wrappers
   - Easy to test and maintain

2. **Algorithm Choice**: Depends on your requirements
   - **FedAvg**: Standard, trusted server
   - **Split Learning**: High privacy, 2 parties
   - **Swarm Learning**: Decentralized, untrusted server

3. **Read the Docs**: 
   - `README_LEARNING_APPROACHES.md` - Algorithm comparison
   - `ARCHITECTURE_GUIDE.md` - Why functional approach is best

---

## Contributing

When adding new examples:

1. **Extract pure functions first**
   - Put business logic in pure functions
   - No framework dependencies
   - Easy to test

2. **Create thin wrappers**
   - Minimal classes with @flare decorators
   - Delegate to pure functions
   - Only orchestration logic

3. **Document**
   - Update README_LEARNING_APPROACHES.md
   - Add code examples
   - Explain use cases

---

## Questions?

- **What is federated learning?** See `README_LEARNING_APPROACHES.md`
- **Why use functional programming?** See `ARCHITECTURE_GUIDE.md`
- **How do I deploy?** Use `pt_fedavg_functional_collab.py` as template
- **How do I test?** Import pure functions and test directly

Happy federated learning! 🎉

