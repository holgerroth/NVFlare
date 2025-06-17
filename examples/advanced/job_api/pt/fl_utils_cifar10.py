from src.cifar10_fl import Net

from nvflare.app_common.workflows.fedavg import FedAvg
from fl_utils import Server, Client, FLRunner


if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_fl.py"

    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=Net()
    )
    server = Server(controller)
    clients = [Client(train_script, script_args="--dataset /tmp/nvflare/cifar10/cifar10_data_site-{i}.pkl") for i in range(n_clients)]

    # Run experiment
    runner = FLRunner(server=server, clients=clients)
    runner.simulate(workdir="/tmp/nvflare/cifar10", gpu="0")
    # runner.export("/tmp/nvflare/jobs")
    # runner.deploy(admindir="/tmp/startup_kits/admin")

