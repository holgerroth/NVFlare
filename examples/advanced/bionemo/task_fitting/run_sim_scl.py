# DEBUG
import os
os.environ["SIM_LOCAL"] = "False"
from nvflare import SimulatorRunner    
n_clients = 3
split_alpha = 1.0

simulator = SimulatorRunner(
    #job_folder="jobs/local_finetune_esm2nv",
    #workspace=f"./results_scl/local_finetune_esm2nv_alpha{split_alpha}_freeze_encoder_100epochs_512d",
    job_folder="jobs/fedavg_finetune_esm2nv",
    workspace=f"./results_scl/fedavg_finetune_esm2nv_alpha{split_alpha}_freeze_encoder_100epochs_512d",    
    n_clients=n_clients,
    threads=n_clients
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)

