import os
os.environ["SIM_LOCAL"] = "False"
from nvflare import SimulatorRunner    
n_clients = 8

simulator = SimulatorRunner(
    job_folder="jobs/local_sabdab_esm1nv",
    workspace=f"./results/local_finetune_esm1nv_enclr1e-6_maxepochs300_JoinedChains_alpha1.0",
    #job_folder="jobs/central_sabdab_esm1nv",
    #workspace=f"./results/central_finetune_esm1nv_enclr1e-6_maxepochs200_JoinedChains",
    #job_folder="jobs/fedavg_sabdab_esm1nv",
    #workspace=f"./results/fedavg_finetune_esm1nv_enclr1e-6_maxepochs300_JoinedChains_alpha1.0",
    n_clients=n_clients,
    threads=n_clients
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
