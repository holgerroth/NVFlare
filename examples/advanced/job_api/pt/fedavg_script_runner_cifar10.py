from src.cifar10_fl import Net

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner


def run_federated_simulation(server, clients, workdir, gpu=None, job_name="fed_sim_job"):
    """
    Simplified API to run a federated learning simulation.
    Args:
        server: Server object containing the controller.
        clients: List of Client objects with training scripts.
        workdir: Directory for simulation output.
        gpu: GPU id as string, or None for CPU.
        job_name: Name for the job (optional).
    """
    job = BaseFedJob(
        name=job_name,
        initial_model=server.initial_model,
    )
    job.to_server(server.controller)

    # Add clients to the job
    for i, client in enumerate(clients):
        job.to(client.runner, f"site-{i}")
    job.simulator_run(workdir, gpu=gpu)


class Server:
    def __init__(self, controller, initial_model=None):
        self.controller = controller
        self.initial_model = initial_model

class Client:
    def __init__(self, train_script):
        self.train_script = train_script
        self.runner = ScriptRunner(script=train_script)

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_fl.py"

    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    server = Server(controller, initial_model=Net())
    clients = [Client(train_script) for _ in range(n_clients)]

    run_federated_simulation(
        server=server,
        clients=clients,
        workdir="/tmp/nvflare/jobs/cifar10",
        gpu="0",
        job_name="cifar10_pt_fedavg",
    )
