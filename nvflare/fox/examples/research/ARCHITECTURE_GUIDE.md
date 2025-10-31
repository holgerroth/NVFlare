# FOX Architecture Guide: Class-Based vs Functional Approaches

## Overview

This guide compares two programming styles for implementing federated learning algorithms in FOX. Both styles work, but each has different trade-offs.

---

## The Two Approaches

### 1. Class-Based Approach (`pt_fedavg.py`)

**Structure:**
- Business logic in class methods
- State stored in instance variables (`self.*`)
- All code in one class

**Example:**
```python
class MyFedAvg:
    def __init__(self, initial_model, num_rounds=3):
        self.initial_model = initial_model
        self.num_rounds = num_rounds
    
    @flare.algo
    def run_fedavg(self):
        current_model = self.initial_model
        for i in range(self.num_rounds):
            results = flare.clients.train(i, current_model)
            current_model = self.aggregate_results(current_model, results)
        return current_model
    
    def aggregate_results(self, current_model, results):
        # Aggregation logic uses self
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
```

**Pros:**
- ✓ Familiar to OOP developers
- ✓ All related code in one place
- ✓ Simple for small examples

**Cons:**
- ✗ Hard to test methods without framework setup
- ✗ Business logic tightly coupled to FOX
- ✗ Can't easily reuse logic in other contexts

---

### 2. Functional Approach (`pt_fedavg_functional_collab.py`) ⭐ RECOMMENDED

**Structure:**
- Business logic in pure functions (no `self`, no framework)
- Thin wrapper classes for FOX integration
- Clear separation of concerns

**Example:**
```python
# ============================================================================
# PURE FUNCTIONS - No framework dependencies
# ============================================================================

def aggregate_weights(weight_list):
    """Pure function - easy to test, easy to reuse."""
    averaged = {}
    num_clients = len(weight_list)
    
    for param_name in weight_list[0].keys():
        param_sum = torch.zeros_like(weight_list[0][param_name])
        for client_weights in weight_list:
            param_sum += client_weights[param_name]
        averaged[param_name] = param_sum / num_clients
    
    return averaged


# ============================================================================
# THIN WRAPPERS - FOX integration only
# ============================================================================

class FunctionalFedAvgServer:
    def __init__(self, initial_weights, num_rounds):
        self.initial_weights = initial_weights
        self.num_rounds = num_rounds
    
    @flare.algo
    def run(self):
        global_weights = self.initial_weights
        for round_idx in range(self.num_rounds):
            results = flare.clients.train(global_weights, round_idx)
            client_weights = [r[0] for r in results]
            # Delegate to pure function
            global_weights = aggregate_weights(client_weights)
        return global_weights
```

**Pros:**
- ✓ Easy to test (just call the function)
- ✓ Business logic is framework-independent
- ✓ Easy to reuse in other projects
- ✓ Clear separation of concerns
- ✓ Easy to swap algorithms

**Cons:**
- ✗ More files/functions to organize
- ✗ Less familiar to OOP-only developers

---

## Testing Comparison

### Class-Based - Harder to Test
```python
# Need to create instance and mock framework
def test_aggregation():
    server = MyFedAvg(initial_model={...}, num_rounds=1)
    # aggregate_results needs self and framework context
    # Requires more setup
```

### Functional - Easy to Test
```python
# Just call the function
def test_aggregation():
    weights1 = {"layer1": torch.tensor([1.0, 2.0])}
    weights2 = {"layer1": torch.tensor([3.0, 4.0])}
    
    result = aggregate_weights([weights1, weights2])
    
    expected = torch.tensor([2.0, 3.0])
    assert torch.allclose(result["layer1"], expected)
    # No framework setup needed!
```

---

## When to Use Each

### Use Class-Based When:
- Learning the FOX framework
- Building quick prototypes
- Simple logic that doesn't need independent testing
- You prefer OOP style

### Use Functional When: ⭐ RECOMMENDED
- **Production deployments**
- **Testing is important**
- **Maintainability matters**
- You want to reuse logic across projects
- You want framework-independent business logic

---

## Summary

Both approaches work in FOX. The **functional approach is recommended for production** because it provides better testability, maintainability, and portability. The class-based approach is simpler for learning and prototyping.

**Key Insight:** Separate your business logic (pure functions) from framework integration (thin wrappers).

