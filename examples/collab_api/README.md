# Collab API: simple FedAvg example

This folder contains **`simple_fedavg.py`**, a minimal federated averaging demo using the NVFlare Collab API with **standalone functions** only: a `@collab.main` entry point, a `@collab.publish` client training function, and `CollabRecipe` with `SimEnv` for local simulation.

## Prerequisites

- Python 3.9 or newer

## Install

Create and use a **virtual environment** in this folder so dependencies stay isolated:

```bash
cd examples/collab_api
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (Command Prompt): `.venv\Scripts\activate.bat` — or PowerShell: `.venv\Scripts\Activate.ps1`.

Then continue with one of the options below (the `pip` commands apply inside this environment).

**Option A — add the repository root to `PYTHONPATH` (no editable install)**

Point Python at the `nvflare` package in your checkout by prepending the **repository root** to `PYTHONPATH`. You still need **PyTorch** (`pip install torch`) and NVFlare’s other dependencies; if something is missing, install it from the root `setup.cfg` or use **Option B** for a one-shot setup.

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

The script runs a short FedAvg job in simulation (`SimEnv` with five clients), trains a tiny linear model for several rounds, and prints job status and result paths.

## What to look for in the code

- **`@collab.publish`** — client-side `train` function that returns updated weights and loss.
- **`@collab.main`** — server-side `fed_avg` loop that calls `collab.clients.train` and aggregates with `simple_avg`.
- **`CollabRecipe`** — wires the current module; **`SimEnv`** drives a local multi-client simulation.
