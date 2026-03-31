# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .task import DEVICE, Net, get_weights, load_data, set_weights, test, train

# Module-level cache for model and data to avoid reloading every round.
# Flower's supernode runs as a long-lived subprocess (not re-imported per round),
# so module-level state persists across FL rounds.
net = None
trainloader = None
testloader = None
_init_lock = threading.Lock()


def _ensure_data_loaded():
    """Load model and data once, reusing cached values on subsequent calls."""
    global net, trainloader, testloader
    with _init_lock:
        if net is None:
            net = Net().to(DEVICE)
        if trainloader is None or testloader is None:
            trainloader, testloader = load_data()


# Define FlowerClient and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, learning_rate, momentum):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum

    def fit(self, parameters, config):
        set_weights(net, parameters)
        results = train(
            net,
            trainloader,
            testloader,
            epochs=1,
            device=DEVICE,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
        )
        return get_weights(net), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(net, parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def _get_required_hyperparameters(context: Context):
    missing = [key for key in ("learning-rate", "momentum") if key not in context.run_config]
    if missing:
        missing_args = ", ".join(missing)
        raise ValueError(
            f"missing required run_config value(s): {missing_args}. "
            "Define them in [tool.flwr.app.config] in pyproject.toml or pass them via run_config."
        )

    return context.run_config["learning-rate"], context.run_config["momentum"]


def client_fn(context: Context):
    """Create and return an instance of Flower `Client`."""
    _ensure_data_loaded()
    learning_rate, momentum = _get_required_hyperparameters(context)
    return FlowerClient(learning_rate, momentum).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
