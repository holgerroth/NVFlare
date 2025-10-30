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

All examples use the same research API style with decorators:

```python
# Run FedAvg
python pt_avg_research_api.py

# Run Split Learning
python pt_split_learning.py

# Run Swarm Learning
python pt_swarm_learning.py
```

Each example demonstrates the core concepts with synthetic data and simple PyTorch models.

