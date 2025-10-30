# Architecture Comparison: Functional vs Class-Based Approaches

## Recommended Production Approach: Pure Functions + @flare.collab

This document explains why the hybrid functional approach (`pt_fedavg_functional_collab.py`) is recommended for production federated learning.

---

## Architecture Diagram

### Traditional Class-Based Approach
```
┌────────────────────────────────────────┐
│   @flare.server                        │
│   class FedAvgServer:                  │
│   ┌──────────────────────────────────┐ │
│   │  __init__(self, ...)             │ │
│   │  ├─ self.initial_model           │ │
│   │  ├─ self.num_rounds              │ │
│   │  └─ self.aggregated_results      │ │
│   │                                  │ │
│   │  @flare.main                     │ │
│   │  run_fedavg(self):               │ │
│   │  ├─ training loop                │ │
│   │  ├─ call clients                 │ │
│   │  └─ aggregate inline             │ │    ← Hard to test!
│   │                                  │ │
│   │  aggregate_results(self, ...):   │ │
│   │  └─ uses self.state              │ │    ← Tightly coupled
│   └──────────────────────────────────┘ │
└────────────────────────────────────────┘

Issues:
❌ Logic mixed with framework
❌ Hard to unit test (need framework setup)
❌ State in self.* variables
❌ Tightly coupled to FOX
```

### Recommended: Functional Core + Thin Wrappers
```
┌──────────────────────────────────────────────────────────────┐
│  PURE FUNCTIONS (Business Logic)                             │
│  ════════════════════════════════════════                     │
│                                                               │
│  def aggregate_weights(weight_list) -> weights               │
│      """Pure function - easy to test!"""                     │
│      # No self, no framework, no side effects                │
│                                                               │
│  def local_train(weights, client_id, lr, ...) -> weights     │
│      """Pure function - framework independent!"""            │
│                                                               │
│  def compute_metrics(weights) -> metrics                     │
│      """Pure function - returns data only!"""                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                        ▲
                        │ delegates to
                        │
┌──────────────────────────────────────────────────────────────┐
│  THIN WRAPPERS (Framework Integration)                       │
│  ═══════════════════════════════════════                     │
│                                                               │
│  @flare.server                                               │
│  class FunctionalFedAvgServer:                               │
│      @flare.main                                             │
│      def run(self):                                          │
│          results = flare.clients.train(...)                  │
│          weights = aggregate_weights(results)  ← Pure func!  │
│                                                               │
│  @flare.client                                               │
│  class FunctionalFedAvgClient:                               │
│      @flare.collab                                           │
│      def train(self, weights, round_idx):                    │
│          return local_train(weights, ...)  ← Pure func!      │
│                                                               │
└──────────────────────────────────────────────────────────────┘

Benefits:
✓ Business logic is pure & testable
✓ Framework is just a thin wrapper
✓ Easy to swap FL frameworks
✓ Can test without any setup
✓ Clear separation of concerns
```

---

## Code Comparison

### Traditional Approach (Class-Based)
```python
@flare.server
class FedAvgServer:
    def __init__(self, initial_model, num_rounds):
        self.initial_model = initial_model
        self.num_rounds = num_rounds
    
    @flare.main
    def run_fedavg(self):
        current_model = self.initial_model
        for i in range(self.num_rounds):
            results = flare.clients.train(i, current_model)
            current_model = self.aggregate_results(current_model, results)
        return current_model
    
    def aggregate_results(self, current_model, results):
        # Aggregation logic mixed with class state
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

# ❌ Can't test aggregate_results without creating server instance!
# ❌ Can't test without FlareRecipe setup
```

### Recommended Approach (Functional + @flare.collab)
```python
# ============================================================================
# PURE FUNCTIONS - Can be imported and tested anywhere!
# ============================================================================

def aggregate_weights(weight_list: List[Dict]) -> Dict:
    """Pure function - no dependencies, no side effects."""
    averaged = {}
    num_clients = len(weight_list)
    param_names = list(weight_list[0].keys())
    
    for param_name in param_names:
        param_sum = torch.zeros_like(weight_list[0][param_name])
        for client_weights in weight_list:
            param_sum += client_weights[param_name]
        averaged[param_name] = param_sum / num_clients
    
    return averaged


def local_train(weights: Dict, client_id: int, lr: float) -> Dict:
    """Pure function - completely framework independent."""
    updated_weights = {k: v.clone() for k, v in weights.items()}
    # Training logic here...
    return updated_weights


# ============================================================================
# THIN WRAPPERS - Just orchestration, no business logic
# ============================================================================

@flare.server
class FunctionalFedAvgServer:
    def __init__(self, initial_weights, num_rounds):
        self.initial_weights = initial_weights
        self.num_rounds = num_rounds
    
    @flare.main
    def run(self):
        global_weights = self.initial_weights
        for round_idx in range(self.num_rounds):
            results = flare.clients.train(global_weights, round_idx)
            client_weights = [r[0] for r in results]
            # Delegate to pure function!
            global_weights = aggregate_weights(client_weights)
        return global_weights


@flare.client
class FunctionalFedAvgClient:
    def __init__(self, client_id, learning_rate):
        self.client_id = client_id
        self.learning_rate = learning_rate
    
    @flare.collab
    def train(self, weights, round_idx):
        # Delegate to pure function!
        return local_train(weights, self.client_id, self.learning_rate)


# ✅ Can test aggregate_weights() instantly - just call it!
# ✅ Can test local_train() instantly - no framework needed!
# ✅ Business logic is framework-agnostic
```

---

## Testing Comparison

### Traditional Approach - Difficult to Test
```python
# ❌ Need to create instances, mock framework
def test_aggregation():
    # Need to instantiate server
    server = FedAvgServer(initial_model={...}, num_rounds=1)
    
    # Can't call aggregate_results directly without setup
    # Need to mock self.initial_model and other state
    # Tightly coupled to class structure
```

### Functional Approach - Easy to Test
```python
# ✅ Just call the function!
def test_aggregation():
    # Arrange
    weights1 = {"layer1": torch.tensor([1.0, 2.0])}
    weights2 = {"layer1": torch.tensor([3.0, 4.0])}
    
    # Act
    result = aggregate_weights([weights1, weights2])
    
    # Assert
    expected = torch.tensor([2.0, 3.0])
    assert torch.allclose(result["layer1"], expected)
    
    # ✅ No framework setup required!
    # ✅ No mocking needed!
    # ✅ Pure function is predictable!
```

---

## When to Use Each Approach

### Use Traditional Class-Based When:
- You're already familiar with OOP patterns
- Your team prefers class-based code
- You have simple logic that doesn't need testing

### Use Functional + @flare.collab When: ⭐ **RECOMMENDED**
- **Production deployments** (most reliable)
- **Testing is important** (always!)
- **Maintainability matters** (always!)
- You want to **swap algorithms easily**
- You might **migrate frameworks** in the future
- You want to **understand what's happening** clearly

### Use Pure Functional When:
- **Learning** federated learning concepts
- **Prototyping** new algorithms quickly
- **Teaching** FL to others
- You don't need production deployment yet

---

## Migration Path

If you have existing class-based code, here's how to refactor:

```python
# BEFORE: Mixed business logic and framework
@flare.server
class MyServer:
    def process_data(self, data):
        # Business logic mixed in...
        result = complicated_computation(data, self.state)
        return result

# AFTER: Extract pure function
def process_data_pure(data, config):
    """Pure function - easy to test!"""
    result = complicated_computation(data, config)
    return result

@flare.server
class MyServer:
    def process_data(self, data):
        # Thin wrapper delegates to pure function
        return process_data_pure(data, self.config)
```

---

## Summary

**The functional + @flare.collab approach (`pt_fedavg_functional_collab.py`) provides:**

1. ✅ **Testability** - Test business logic without framework
2. ✅ **Maintainability** - Clear separation of concerns
3. ✅ **Deployability** - Works with FlareRecipe for production
4. ✅ **Flexibility** - Easy to modify and extend
5. ✅ **Portability** - Core logic is framework-agnostic
6. ✅ **Debuggability** - Pure functions are easy to trace
7. ✅ **Understandability** - Explicit data flow

**This is the recommended approach for real-world federated learning projects!** ⭐

