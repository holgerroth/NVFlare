import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
import numpy as np

SEED = 0
LOG_FILE = "/tmp/nvflare_reproduce/flwr_in_flare_client1.csv"
#LOG_FILE = "/tmp/nvflare_reproduce/flwr_alone_client1.csv"

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

from flwr.client import ClientApp, NumPyClient

from task import (
    Net,
    DEVICE,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define FlowerClient and client_fn
class FlowerClient(NumPyClient):
    def fit(self, parameters, config):
        set_weights(net, parameters)
        results = train(net, trainloader, testloader, epochs=1, device=DEVICE, log_file=LOG_FILE)
        return get_weights(net), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(net, parameters)
        loss, accuracy = test(net, testloader)
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
