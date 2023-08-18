import os
import glob
import subprocess
from nvflare import SimulatorRunner

data_root = "/tmp/data"

def clean_files(data_root, ext):
    files = glob.glob(os.path.join(data_root, "*", ext), recursive=True)
    for file in files:
        print(file)
        os.remove(file)

def clean_memmap(data_root):
    clean_files(data_root, "*.npy")
    clean_files(data_root, "*.info")

# Clean temporary data
clean_memmap(data_root)

# Start FL simulation
simulator = SimulatorRunner(
    job_folder="jobs/peft",
    workspace="/tmp/nvflare/nemo/peft_fedavg_345M",
    n_clients=1,
    threads=1
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
