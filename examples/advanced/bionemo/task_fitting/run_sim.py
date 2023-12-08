# DEBUG
import os
os.environ["SIM_LOCAL"] = "False"
from nvflare import SimulatorRunner    
n_clients = 1
split_alpha = 100.0

simulator = SimulatorRunner(
    #job_folder="jobs/local_finetune_esm1nv",
    #workspace=f"/tmp/nvflare/bionemo/local_site1_finetune_esm1nv_alpha{split_alpha}_freeze_encoder_large_ds",
    #workspace=f"/tmp/nvflare/bionemo/DEBUG5_local_site1_finetune_esm1nv_alpha{split_alpha}_unfreeze_encoder_large_ds_dropout-0.25_hidden_dim-64_LR-0.00001_PARAMGROUPS_val_optim2_enclr1e-7",
    job_folder="jobs/local_finetune_esm2nv",
    #workspace=f"/tmp/nvflare/bionemo/local_site1_finetune_esm2nv_alpha{split_alpha}_freeze_encoder3",
    #workspace=f"/tmp/nvflare/bionemo/local_site1_finetune_esm2nv_alpha{split_alpha}_freeze_encoder_large_ds_finetune",
    workspace=f"/tmp/nvflare/bionemo/local_site1_finetune_esm2nv_alpha{split_alpha}_unfreeze_encoder_large_ds_dropout-0.25_hidden_dim-64_LR-0.00001_PARAMGROUPS_val_optim2_enclr1e-4",
    n_clients=n_clients,
    threads=n_clients
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)

