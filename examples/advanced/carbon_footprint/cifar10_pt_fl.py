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
    
    # Log slowdown args if available
    try:
        print(f"[slowdown] sleep_ms_mean={args.sleep_ms_mean}, sleep_ms_std={args.sleep_ms_std}, "
              f"extra_no_update_iters={args.extra_no_update_iters}, max_steps_per_epoch={args.max_steps_per_epoch}")
    except Exception:
        print("[slowdown] args not available at tracker init")
    # attach CLI args for use in training loop
    try:
        tracker._args = args
    except Exception:
        pass
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
        for epoch in range(epochs):

            _args = getattr(tracker, "_args", None)
            sleep_ms_mean = getattr(_args, "sleep_ms_mean", 0.0) if _args else 0.0
            sleep_ms_std  = getattr(_args, "sleep_ms_std", 0.0) if _args else 0.0
            extra_no_update_iters = max(0, getattr(_args, "extra_no_update_iters", 0)) if _args else 0
            max_steps_per_epoch = getattr(_args, "max_steps_per_epoch", -1) if _args else -1
            train_step_count = 0

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_step_count += 1
                # extra forward-only iterations (no optimizer/backward)
                for _ in range(int(extra_no_update_iters)):
                    _ = net(inputs)
                    if DEVICE == "cuda":
                        torch.cuda.synchronize()
                # Gaussian sleep delay clipped at 0
                if (sleep_ms_mean or sleep_ms_std):
                    import random, time as _time
                    delay_ms = random.gauss(float(sleep_ms_mean), float(sleep_ms_std))
                    if delay_ms < 0: delay_ms = 0
                    _time.sleep(delay_ms/1000)
                # Optional cap on real optimizer steps per epoch
                if isinstance(max_steps_per_epoch, int) and max_steps_per_epoch > 0 and train_step_count >= max_steps_per_epoch:
                    break
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

        # derive durations (sec) for train/eval and round total
        def _dur(x):
            try:
                return getattr(x, "duration")
            except Exception:
                try:
                    return x.get("duration") or x.get("duration_sec") or x.get("duration_seconds")
                except Exception:
                    return None
        _train_sec = _dur(train_emissions_data) or 0
        _eval_sec = _dur(evaluate_emissions_data) or 0
        _round_total_seconds = float(_train_sec) + float(_eval_sec)

        emissions_data = {
            "idle": idle_emissions_data,
            "init": init_emissions_data if input_model.current_round == 0 else None,
            "train": train_emissions_data,
            "evaluate": evaluate_emissions_data,
            "round_total_seconds": _round_total_seconds,
            "train_seconds": _train_sec,
            "eval_seconds": _eval_sec
        }

        params = net.cpu().state_dict()
        model_params_bytes = pickle.dumps(params)
        model_params_size = len(model_params_bytes)
        print(f"Model parameters size: {model_params_size} bytes ({model_params_size / 1024:.2f} KB)")

        emissions_data["model_params_size"] = model_params_size

        output_model = flare.FLModel(
            params=params,
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps, "EMISSIONS_DATA": emissions_data, "ROUND_TOTAL_SECONDS": _round_total_seconds, "TRAIN_SECONDS": _train_sec, "EVAL_SECONDS": _eval_sec},
        )
        
        flare.send(output_model)

    tracker.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR10 FL Training with Carbon Footprint Tracking')
    parser.add_argument('--country_iso_code', type=str, default='USA', help='3-letter ISO code for the country to use for carbon emissions calculation')
    parser.add_argument('--max_steps_per_epoch', type=int, default=-1, help='If >0, cap real optimizer steps per epoch to this number.')
    parser.add_argument('--sleep_ms_mean', type=float, default=0.0, help='Mean per-step sleep (ms) to simulate device delay.')
    parser.add_argument('--sleep_ms_std', type=float, default=0.0, help='Std dev of per-step sleep (ms); Gaussian, clipped at 0.')
    parser.add_argument('--extra_no_update_iters', type=int, default=0, help='Extra forward-only iterations per batch (no backward/step).')
    args = parser.parse_args()

    main(args)
    