import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net

# (1) import nvflare client API
import nvflare.client as flare
from nvflare import Job
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg

# (optional) set a fix place so we don't need to download everytime
dataset_path = "/tmp/nvflare/data"
# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change device="cpu"


""" SERVER CODE """


class FedAvg(BaseFedAvg):
    """Controller for FedAvg Workflow."""

    def run(self) -> None:
        self.info("Start FedAvg.")

        for self._current_round in range(self._num_rounds):
            self.info(f"Round {self._current_round} started.")

            clients = self.sample_clients(self._min_clients)

            results = self.send_model_and_wait(targets=clients, data=self.model)

            aggregate_results = self.aggregate(
                results, aggregate_fn=None
            )  # if no `aggregate_fn` provided, default `WeightedAggregationHelper` is used

            self.update_model(aggregate_results)

            self.save_model()

        self.info("Finished FedAvg.")


""" CLIENT CODE """


def client_train(dataset_path="/tmp/nvflare/data", device="cuda:0"):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    epochs = 2

    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net()

    # (2) initializes NVFlare client API
    flare.init()

    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # (4) loads model from NVFlare
        net.load_state_dict(input_model.params)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # (optional) use GPU to speed things up
        net.to(device)
        # (optional) calculate total steps
        steps = epochs * len(trainloader)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # (optional) use GPU to speed things up
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

        print("Finished Training")

        PATH = "./cifar_net.pth"
        torch.save(net.state_dict(), PATH)

        # (5) wraps evaluation logic into a method to re-use for
        #       evaluation on both trained and received model
        def evaluate(input_weights):
            net = Net()
            net.load_state_dict(input_weights)
            # (optional) use GPU to speed things up
            net.to(device)

            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in testloader:
                    # (optional) use GPU to speed things up
                    images, labels = data[0].to(device), data[1].to(device)
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")
            return 100 * correct // total

        # (6) evaluate on received model for model selection
        accuracy = evaluate(input_model.params)
        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=net.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        # (8) send model back to NVFlare
        flare.send(output_model)


def main():
    job = Job(name="cifar10_fedavg")

    wf = FedAvg(min_clients=n_clients, num_rounds=10)
    job.to(wf, "server")

    for i in range(n_clients):
        client_name = f"site-{i}"
        job.to(client_train(dataset_path=f"/tmp/data/{client_name}"), client_name, gpu=...)

    # run simulator
    job.simulate(threads=n_clients)


if __name__ == "__main__":
    main()
