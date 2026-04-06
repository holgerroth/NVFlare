# Collab API examples

This folder has two small demos that use the NVFlare **Collab API** with **standalone functions** only (`@collab.main`, `@collab.publish`), **`CollabRecipe`**, and **`SimEnv`** for local simulation. No custom classes are required for the federated workflow itself.

| Script | What it shows |
| --- | --- |
| **`simple_fedavg.py`** | Federated averaging: clients train a tiny PyTorch model; the server aggregates weights. |
| **`simple_split_learning.py`** | Split learning on MNIST: the client holds images and a **bottom** network; the server holds labels and a **top** network; activations and cut-layer gradients cross the wire. |

## `simple_fedavg.py`

- **Server (`@collab.main`)**: `fed_avg` runs several rounds, calls `collab.clients.train(...)` so all simulated clients train in parallel, then aggregates with `simple_avg`.
- **Client (`@collab.publish`)**: `train` runs local SGD on synthetic data and returns updated weights and loss.
- **Simulation**: `SimEnv(num_clients=5)`.

## `simple_split_learning.py`

- **Server (`@collab.main`)**: `split_learning_flow` loops over MNIST batches (aligned by step index), pulls activations from the first client with `collab.clients[0].forward(step)`, computes loss and accuracy on the server labels, then calls `collab.clients[0].backward(...)`.
- **Client (`@collab.publish`)**: `forward` runs the bottom network on image batches; `backward` applies cut-layer gradients to the bottom parameters.
- **Simulation**: `SimEnv(num_clients=1)` (single data holder; labels live on the server).

Dependencies are listed in **`requirements.txt`** (PyTorch, torchvision for MNIST, and `nvflare` when installing from PyPI).

## Prerequisites

- Python 3.9 or newer

## Install

Create and use a **virtual environment** in this folder so dependencies stay isolated:

```bash
cd examples/collab_api
python3 -m venv .venv
source .venv/bin/activate
```

Then continue with one of the options below (the `pip` commands apply inside this environment).

**Option A — add the repository root to `PYTHONPATH` (no editable install)**

Point Python at the `nvflare` package in your checkout by prepending the **repository root** to `PYTHONPATH`. You still need **PyTorch** and **torchvision** (`pip install torch torchvision`) and NVFlare’s other dependencies; if something is missing, install it from the root `setup.cfg` or use **Option B** for a one-shot setup.

From **`examples/collab_api`** (this folder):

```bash
export PYTHONPATH="$(cd ../.. && pwd):$PYTHONPATH"
```

**Option B — from this repository (recommended when developing Collab or using a branch that is not on PyPI yet)**

From the **repository root** (not this folder):

```bash
pip install -e .
```

**Install requirements**

```bash
pip install -r requirements.txt
```

Use this when a suitable `nvflare` release that includes the Collab API is available on your index.

## Run

From this directory:

```bash
python simple_fedavg.py
```

```bash
python simple_split_learning.py
```

`simple_fedavg.py` runs a short FedAvg job with five simulated clients, trains a small linear model for several rounds, and prints job status and result paths.

`simple_split_learning.py` downloads MNIST on first run (into `./data` by default), runs split-learning steps with logged batch loss and accuracy, then prints job status and result paths.

## What to look for in the code

**FedAvg (`simple_fedavg.py`)**

- **`@collab.publish`** — client-side `train` returns updated weights and loss.
- **`@collab.main`** — server-side `fed_avg` calls `collab.clients.train` and aggregates with `simple_avg`.
- **`CollabRecipe`** / **`SimEnv`** — recipe wiring and local multi-client simulation.

**Split learning (`simple_split_learning.py`)**

- **`@collab.publish`** — `forward` / `backward` for the bottom model; **`@collab.main`** — `split_learning_flow` for the top model and orchestration.
- **`collab.clients[0]`** — single-client proxy calls (direct result, not a group list).
- **`compute_loss_and_grads`** — server-side cut-layer forward, loss, backward, and batch accuracy.
