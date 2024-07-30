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
from flwr.client import ClientApp, NumPyClient
from task import DEVICE, Net, get_weights, load_data, set_weights, test, train

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()

# initializes NVFlare interface
from nvflare.client.tracking import SummaryWriter
import nvflare.client as flare
flare.init()


# Define FlowerClient and client_fn
class FlowerClient(NumPyClient):
    def __init__(self):
        self.writer = SummaryWriter()
        self.step = 0

    def fit(self, parameters, config):
        set_weights(net, parameters)
        results = train(net, trainloader, testloader, epochs=1, device=DEVICE)

        self.writer.add_scalar("train_loss", results["train_loss"], self.step)
        self.writer.add_scalar("train_accuracy", results["train_accuracy"], self.step)
        self.writer.add_scalar("val_loss", results["val_loss"], self.step)
        self.writer.add_scalar("val_accuracy", results["val_accuracy"], self.step)

        print("%%%%%%%%%%%%%%%STEP", self.step)

        self.step += 1

        return get_weights(net), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(net, parameters)
        loss, accuracy = test(net, testloader)

        self.writer.add_scalar("test_loss", loss, self.step)
        self.writer.add_scalar("test_accuracy", accuracy, self.step)

        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
