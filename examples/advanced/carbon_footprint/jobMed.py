"""NVFlare job.py (SLOW via extra forward-only iterations):

- sleep_ms_mean=0, sleep_ms_std=0
- extra_no_update_iters=100  (adds 100 extra forward passes per batch for ALL clients)
"""
from cifar10_pt_fl import Net
from fedavg_carbon import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 6
    num_rounds = 10
    train_script = "cifar10_pt_fl.py"

    job = BaseFedJob(
        name="carbon_footprint_iters100",
        initial_model=Net(),
    )

    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to_server(controller)

    for i in range(n_clients):
        sim = "--sleep_ms_mean=0 --sleep_ms_std=0 --extra_no_update_iters=100"
        runner = ScriptRunner(script=train_script, script_args=f"--country_iso_code=USA {sim}")
        job.to(runner, f"site-{i}")

    job.export_job("./job_configs")
    job.simulator_run("runMed", gpu="0")
