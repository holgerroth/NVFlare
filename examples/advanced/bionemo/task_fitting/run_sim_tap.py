# DEBUG
import os
os.environ["SIM_LOCAL"] = "False"
from nvflare import SimulatorRunner    
n_clients = 5

simulator = SimulatorRunner(
    #job_folder="jobs/local_tap_esm1nv",
    #workspace=f"/tmp/nvflare/bionemo/local_finetune_esm1nv_enclr1e-6_maxepochs500_SplitChained",
    job_folder="jobs/fedavg_tap_esm1nv",
    workspace=f"/tmp/nvflare/bionemo/fedavg_finetune_esm1nv_enclr1e-6_maxepochs10_SplitChained_1",
    n_clients=n_clients,
    threads=n_clients
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)

