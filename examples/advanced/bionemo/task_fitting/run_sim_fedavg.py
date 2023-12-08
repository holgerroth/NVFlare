# FedAvg
import os
os.environ["SIM_LOCAL"] = "False"
from nvflare import SimulatorRunner    
n_clients = 3
split_alpha = 1.0

simulator = SimulatorRunner(
    #job_folder="jobs/fedavg_finetune_esm1nv",
    #workspace=f"/tmp/nvflare/bionemo/fedavg_finetune_esm1nv_alpha{split_alpha}",
    #workspace=f"/tmp/nvflare/bionemo/local_site1_finetune_esm1nv_alpha{split_alpha}_unfreeze_encoder4",
    job_folder="jobs/fedavg_finetune_esm2nv",
    #workspace=f"/tmp/nvflare/bionemo/local_site1_finetune_esm2nv_alpha{split_alpha}_freeze_encoder2",
    #workspace=f"/tmp/nvflare/bionemo/local_site1_finetune_esm2nv_alpha{split_alpha}_unfreeze_encoder1_large_ds",
    workspace=f"/tmp/nvflare/bionemo/3clients_FedAvg_finetune_esm2nv_alpha{split_alpha}_freeze_encoder2",
    #workspace=f"/tmp/nvflare/bionemo/3clients_Local_finetune_esm2nv_alpha{split_alpha}_freeze_encoder2",
    n_clients=n_clients,
    threads=n_clients
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
