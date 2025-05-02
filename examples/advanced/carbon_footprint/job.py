
from cifar10_pt_fl import Net
from fedavg_carbon import FedAvg

from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 30
    train_script = "cifar10_pt_fl.py"

    # Create BaseFedJob with initial model
    job = BaseFedJob(
      name="carbon_footprint",
      initial_model=Net(),
    )

    # Define the controller and send to server
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to_server(controller)

    # Add clients
    for i in range(n_clients):
        runner = ScriptRunner(script=train_script, script_args=f"--country_iso_code=USA")  # ISO code for the country to use for carbon emissions calculation
        job.to(runner, f"site-{i}")

    job.export_job("./job_configs")
    job.simulator_run("/tmp/nvflare/carbon_footprint", gpu="0,1")  # runs each client on a different GPU
    