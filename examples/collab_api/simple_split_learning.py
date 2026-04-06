"""Split learning on MNIST with the Collab API (no classes required for the workflow).

The client holds **images** and a **bottom** network; the server holds **labels** and a **top**
network. Each training step: client forward → server loss and backward on the cut layer → client
backward. The server logs batch loss and batch accuracy each step. Batches are aligned by a shared
step index and `batch_size` with `shuffle=False` on MNIST.

CollabRecipe automatically uses the current module when server/client are not specified.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import backward as autograd_backward
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from nvflare.collab import collab
from nvflare.collab.sim import SimEnv
from nvflare.collab.sys.recipe import CollabRecipe

# =============================================================================
# Hyperparameters & MNIST batching
# =============================================================================

BATCH_SIZE = 64
HIDDEN_DIM = 256
MNIST_ROOT = "./data"


def _load_mnist_batches():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST(root=MNIST_ROOT, train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return xs, ys


# =============================================================================
# Client-side: Bottom model (forward / backward)
# =============================================================================

_bottom: Optional[nn.Module] = None
_opt_bottom: Optional[optim.Optimizer] = None
_batches_x: Optional[List[torch.Tensor]] = None
_last_activations: Optional[torch.Tensor] = None


@collab.publish
def forward(step: int):
    """Local forward: one MNIST batch indexed by ``step`` (same index as server labels)."""
    global _last_activations, _bottom, _opt_bottom, _batches_x
    if _bottom is None:
        batches_x, _ = _load_mnist_batches()
        _batches_x = batches_x
        _bottom = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, HIDDEN_DIM),
            nn.ReLU(),
        )
        _opt_bottom = optim.SGD(_bottom.parameters(), lr=0.05)
    bi = step % len(_batches_x)
    x = _batches_x[bi]
    acts = _bottom(x)
    _last_activations = acts
    return acts.detach()


@collab.publish
def backward(grads: torch.Tensor):
    """Apply cut-layer gradients to the bottom model."""
    global _last_activations
    if isinstance(grads, Exception):
        raise grads
    _opt_bottom.zero_grad(set_to_none=True)
    g = grads.to(dtype=_last_activations.dtype, device=_last_activations.device)
    autograd_backward(_last_activations, g)
    _opt_bottom.step()


# =============================================================================
# Server-side: Top model, loss, and cut-layer gradients
# =============================================================================

_top: Optional[nn.Module] = None
_opt_top: Optional[optim.Optimizer] = None
_batches_y: Optional[List[torch.Tensor]] = None
_criterion: Optional[nn.Module] = None


def compute_loss_and_grads(activations: torch.Tensor, labels: torch.Tensor):
    """Server forward on the cut layer, loss, backward; return grad w.r.t. activations, loss, batch accuracy."""
    z = activations.clone().requires_grad_(True)
    logits = _top(z)
    loss = _criterion(logits, labels)
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        batch_acc = (preds == labels).float().mean().item()
    _opt_top.zero_grad(set_to_none=True)
    loss.backward()
    return z.grad.detach(), loss.item(), batch_acc


# =============================================================================
# Server-side: Split learning algorithm
# =============================================================================


@collab.main
def split_learning_flow(num_steps: int = 200):
    """Orchestrate split learning: fetch activations, compute loss on labels, push grads to client."""
    global _top, _opt_top, _batches_y, _criterion
    if _top is None:
        _, batches_y = _load_mnist_batches()
        _batches_y = batches_y
        _top = nn.Linear(HIDDEN_DIM, 10)
        _opt_top = optim.SGD(_top.parameters(), lr=0.05)
        _criterion = nn.CrossEntropyLoss()

    num_batches = len(_batches_y)
    for step in range(num_steps):
        bi = step % num_batches
        labels = _batches_y[bi]

        acts = collab.clients[0].forward(step)
        if isinstance(acts, Exception):
            raise acts

        server_grads, loss, batch_acc = compute_loss_and_grads(acts, labels)
        collab.clients[0].backward(server_grads)
        _opt_top.step()

        print(f"  step {step}/{num_steps}  loss={loss:.4f}  batch_acc={batch_acc:.4f}")

    print("Split learning finished.")
    return None


# =============================================================================
# Execute - CollabRecipe auto-detects this module!
# =============================================================================

if __name__ == "__main__":
    recipe = CollabRecipe(name="split_learning_flow")
    env = SimEnv(num_clients=1)
    run = recipe.execute(env)
    print()
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())
