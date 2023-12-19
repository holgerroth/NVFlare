import os
os.environ["SIM_LOCAL"] = "False"
from nvflare import SimulatorRunner    
n_clients = 6

simulator = SimulatorRunner(
    #job_folder="jobs/central_sabdab_esm1nv",
    #workspace=f"./results/central_finetune_esm1nv_enclr1e-5_maxepochs500_JoinedChains_bs32",
    job_folder="jobs/local_sabdab_esm1nv",
    workspace=f"./results/local_finetune_esm1nv_enclr1e-5_maxepochs500_JoinedChains_alpha100_bs32",
    #job_folder="jobs/fedavg_sabdab_esm1nv",
    #workspace=f"./results/fedavg_finetune_esm1nv_enclr1e-5_maxepochs500_JoinedChains_alpha100_bs32",
    n_clients=n_clients,
    threads=n_clients
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
