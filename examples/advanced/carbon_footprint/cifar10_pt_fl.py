import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from codecarbon import OfflineEmissionsTracker, EmissionsTracker
import argparse
import pickle


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# (1) import nvflare client API
import nvflare.client as flare

# (optional) metrics
from nvflare.client.tracking import SummaryWriter

# (optional) set a fix place so we don't need to download everytime
DATASET_PATH = "/groups/lingurarugrp/atapp/NVFlare/examples/advanced/carbon_footprint/data"
# If available, we use GPU to speed things up.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CODECARBON_API_TOKEN = os.getenv("CODECARBON_API_TOKEN")


def main(tracker=None):

    # (2) initializes NVFlare client API
    flare.init()
    client_name = flare.get_site_name()

    # Initialize the tracker
    #tracker = OfflineEmissionsTracker(country_iso_code=args.country_iso_code, measure_power_secs=1, experiment_id=f"{client_name}")  
    project_name = f"{flare.get_job_id}--{client_name}"
    print(f"Project name: {project_name}")
    tracker = EmissionsTracker(project_name="Test", experiment_id="8e1112c9-3f9c-49f3-ad3a-005504885005", measure_power_secs=1, api_key=CODECARBON_API_TOKEN, tracking_mode="process")
    
    tracker.start_task("init")

    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 256
    epochs = 1

    # See README.md for how to download the dataset
    trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net()

    summary_writer = SummaryWriter()

    init_emissions_data = tracker.stop_task()

    while flare.is_running():
        tracker.start_task(f"idle_time")
        idle_emissions_data = tracker.stop_task()
        print(f"idle_emissions_data: {idle_emissions_data}")

        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")
        tracker.start_task(f"round_{input_model.current_round}")

        net.load_state_dict(input_model.params)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        net.to(DEVICE)
        steps = epochs * len(trainloader)
        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print(f"{client_name} [Round {input_model.current_round}, Epoch: {epoch + 1}, Step: {i + 1:5d}] loss: {running_loss / 100:.3f}")
                    global_step = input_model.current_round * steps + epoch * len(trainloader) + i

                    summary_writer.add_scalar(
                        tag="loss_for_each_batch",
                        scalar=running_loss, 
                        global_step=global_step
                    )
                    running_loss = 0.0

        print("Finished Training")

        PATH = "./cifar_net.pth"
        torch.save(net.state_dict(), PATH)

        train_emissions_data = tracker.stop_task()
        print(f"train_emissions_data: {train_emissions_data}")

        def evaluate(input_weights):
            net = Net()
            net.load_state_dict(input_weights)
            net.to(DEVICE)

            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")
            return 100 * correct // total

        tracker.start_task("evaluate")
        accuracy = evaluate(input_model.params)
        evaluate_emissions_data = tracker.stop_task()
        print(f"evaluate_emissions_data: {evaluate_emissions_data}")

        emissions_data = {
            "idle": idle_emissions_data,
            "init": init_emissions_data if input_model.current_round == 0 else None,
            "train": train_emissions_data,
            "evaluate": evaluate_emissions_data
        }

        params = net.cpu().state_dict()
        model_params_bytes = pickle.dumps(params)
        model_params_size = len(model_params_bytes)
        print(f"Model parameters size: {model_params_size} bytes ({model_params_size / 1024:.2f} KB)")

        emissions_data["model_params_size"] = model_params_size

        output_model = flare.FLModel(
            params=params,
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps, "EMISSIONS_DATA": emissions_data},
        )
        
        flare.send(output_model)

    tracker.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR10 FL Training with Carbon Footprint Tracking')
    parser.add_argument('--country_iso_code', type=str, default='USA',
                      help='3-letter ISO code for the country to use for carbon emissions calculation')
    args = parser.parse_args()

    main(args)
    