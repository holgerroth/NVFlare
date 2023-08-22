#!/usr/bin/python3

import os
import glob
import subprocess
from nvflare import SimulatorRunner

n_clients=3
peft_scheme="lora"
job_name=f"peft_{peft_scheme}_fedavg_345M"

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

# Create configurations
subprocess.run(["python3", "create_configs.py", "--job_folder", f"jobs/{job_name}", 
                "--num_clients", str(n_clients), 
                "--max_steps", "200", 
                "--val_check_interval", "100",
                "--num_rounds", "50",
                "--peft_scheme", peft_scheme])

# Start FL simulation
simulator = SimulatorRunner(
    job_folder=f"jobs/{job_name}",
    workspace=f"/tmp/nvflare/nemo/{job_name}",
    n_clients=n_clients,
    threads=n_clients,
    gpu="0,1,2"
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
