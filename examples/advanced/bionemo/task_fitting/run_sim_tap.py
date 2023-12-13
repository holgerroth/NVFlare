# DEBUG
import os
os.environ["SIM_LOCAL"] = "False"
from nvflare import SimulatorRunner    
n_clients = 5

simulator = SimulatorRunner(
    job_folder="jobs/local_tap_esm1nv",
    workspace=f"./results/local_finetune_esm1nv_enclr1e-6_maxepochs500_JoinedChains",
    #job_folder="jobs/fedavg_tap_esm1nv",
    #workspace=f"./results/fedavg_finetune_esm1nv_enclr1e-6_maxepochs10_JoinedChains",
    n_clients=n_clients,
    threads=n_clients
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)

