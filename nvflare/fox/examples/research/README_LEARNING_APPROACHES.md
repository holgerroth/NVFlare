# Federated Learning Approaches - Examples Comparison

This directory contains three different approaches to federated learning, each with unique characteristics:

## 1. Federated Averaging (FedAvg) - `pt_avg_research_api.py`

**Architecture:** Centralized star topology

**How it works:**
- **Server Role:** Central coordinator that aggregates all model updates
- **Client Role:** Train on local data and send updates to server
- **Communication:** Hub-and-spoke (clients → server → clients)

**Flow:**
```
Round 1:  Server → All Clients (broadcast model)
          All Clients → Server (send trained models)
          Server (aggregates) → All Clients (broadcast aggregated model)
Round 2:  (repeat)
```

**Key Features:**
- Central aggregation point
- Server has access to all model weights
- Simple and efficient
- Most common FL approach

**Use Cases:**
- Standard federated learning scenarios
- When central coordination is acceptable
- Cross-silo FL (hospitals, organizations)

---

## 2. Split Learning - `pt_split_learning.py`

**Architecture:** Sequential vertical partitioning

**How it works:**
- **Two Clients:** Model is split vertically between two parties
- **Client 1 (Front):** Holds bottom layers, processes raw data
- **Client 2 (Back):** Holds top layers, computes loss
- **Server Role:** Minimal coordinator

**Flow (per batch):**
```
Client 1: Forward pass → activations
          ↓
Client 2: Receives activations → completes forward → computes loss → backprop
          ↓ (sends gradients)
Client 1: Receives gradients → completes backprop → updates weights
```

**Key Features:**
- Model is never fully shared
- No party sees complete model architecture
- Sequential processing (slower)
- Strong privacy guarantees

**Use Cases:**
- High privacy requirements
- When parties can't share full model
- Resource-constrained edge devices
- Medical imaging (raw data stays with provider)

---

## 3. Swarm Learning - `pt_swarm_learning.py`

**Architecture:** Decentralized peer-to-peer

**How it works:**
- **Server Role:** Minimal - only initiates process
- **Client Role:** Both trainer AND aggregator (rotating)
- **Communication:** Peer-to-peer between clients

**Flow:**
```
Round 1:  Client A (chosen as aggregator)
          Client A → All Clients (request training)
          All Clients → Client A (send trained models)
          Client A (aggregates locally)
          
Round 2:  Client A → Client B (pass aggregated model)
          Client B (becomes new aggregator)
          Client B → All Clients (request training)
          ... (repeat)
```

**Key Features:**
- No central aggregator
- Aggregator role rotates randomly
- Server never sees model weights
- Fully decentralized after initialization
- No single point of failure/trust

**Use Cases:**
- Untrusted central server
- Blockchain/distributed ledger integration
- Cross-device FL at scale
- Privacy-sensitive applications
- Resilient to server failures

---

## 4. Functional Programming Implementation

In addition to the class-based examples above, we provide a functional programming approach that is **RECOMMENDED for production use**:

### **RECOMMENDED** Production Approach - `pt_fedavg_functional_collab.py`

**Approach:** Pure functional core with `@flare.collab` decorators

**Why this is the BEST approach for real-world use:**
- **Business logic** in pure functions (no framework dependencies)
- **Framework integration** through thin wrapper classes
- **Easily testable** - test pure functions without any setup
- **Production-ready** - works with FlareRecipe and FOX deployment
- **Maintainable** - clear separation between logic and framework
- **Flexible** - easy to swap strategies or adapt to other frameworks

**Architecture:**
```
Pure Functions (Core Logic)        Framework Wrappers
━━━━━━━━━━━━━━━━━━━━━━━━━━        ━━━━━━━━━━━━━━━━━━━━
aggregate_weights()          ←───  
weighted_aggregate()              FunctionalFedAvgServer
local_train()               ←───  
compute_metrics()                 FunctionalFedAvgClient
create_initial_weights()            with @flare.collab
```

**Key Components:**
```python
# Pure Functions (100% testable without framework)
aggregate_weights(weight_list)                      # Simple aggregation
weighted_aggregate(weight_list, sample_counts)      # Weighted aggregation
local_train(weights, client_id, learning_rate, ...)  # Training logic
compute_metrics(weights)                            # Monitoring

# Thin Framework Wrappers

class FunctionalFedAvgServer:
    @flare.algo
    def run(self):
        # Orchestrate workflow, delegate to pure functions
        

class FunctionalFedAvgClient:
    @flare.collab
    def train(self, weights, round_idx):
        # Delegate to pure function
        return local_train(weights, ...)
```

**Testing Example:**
```python
# Test pure functions without ANY framework setup!
def test_aggregation():
    weights1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
    weights2 = {"w": torch.tensor([3.0, 4.0, 5.0])}
    
    result = aggregate_weights([weights1, weights2])
    # Assert result is correct - no FlareRecipe needed!
```

**Benefits:**
- ✓ **Testable:** Unit test pure functions instantly
- ✓ **Debuggable:** Step through pure functions easily
- ✓ **Maintainable:** Logic separated from framework
- ✓ **Deployable:** Works with FlareRecipe for production
- ✓ **Flexible:** Swap aggregation algorithms easily
- ✓ **Portable:** Core logic could work with other FL frameworks

---

## Comparison Table

| Feature | FedAvg | Split Learning | Swarm Learning |
|---------|--------|----------------|----------------|
| **Server Trust** | Required | Low | None |
| **Privacy** | Medium | High | High |
| **Communication** | Centralized | Sequential | P2P |
| **Model Sharing** | Full weights | Activations/gradients only | Full weights (peer-to-peer) |
| **Aggregation** | Server | N/A | Rotating client |
| **Complexity** | Low | Medium | High |
| **Latency** | Low | High | Medium |
| **Scalability** | High | Limited (2 parties) | High |
| **Fault Tolerance** | Single point of failure | Single point of failure | Distributed |

---

## Which Approach to Choose?

### Choose **FedAvg** when:
- ✓ Central server is trusted
- ✓ Simplicity is important
- ✓ Many clients need coordination
- ✓ Low latency is required

### Choose **Split Learning** when:
- ✓ Maximum privacy is critical
- ✓ Parties can't share full model
- ✓ One party has computational limits
- ✓ Working with 2-3 parties only

### Choose **Swarm Learning** when:
- ✓ Central server cannot be trusted
- ✓ Decentralization is important
- ✓ Resilience to failures needed
- ✓ Blockchain/ledger integration wanted

---

## Running the Examples

### Class-Based Examples (FOX Framework)

All class-based examples use the research API style with decorators:

```bash
# Run FedAvg (class-based)
python pt_avg_research_api.py
python pt_fedavg.py  # Alternative name

# Run Split Learning
python pt_split_learning.py

# Run Swarm Learning
python pt_swarm_learning.py
```

### Functional Programming Example

```bash
# RECOMMENDED: Pure functional core with @flare.collab (production-ready)
python pt_fedavg_functional_collab.py
```

Each example demonstrates the core concepts with synthetic data and simple PyTorch models.

---

## Programming Style Comparison

| Style | Examples | Framework | Classes | State Management | Testing | Best For |
|-------|----------|-----------|---------|------------------|---------|----------|
| **Class-Based** | pt_fedavg.py, pt_split_learning.py, pt_swarm_learning.py | FOX | Yes | Instance variables | Framework-dependent | Traditional OOP approach |
| **Functional + Collab** ⭐ | pt_fedavg_functional_collab.py | FOX | Thin wrappers | Pure functions | Independent of framework | **PRODUCTION USE** ⭐ |

**⭐ Recommended:** `pt_fedavg_functional_collab.py` provides the best approach for real-world federated learning projects by combining testable pure functions with production-ready framework integration.

