#!/usr/bin/python3

import os
import glob
import subprocess
from nvflare import SimulatorRunner

n_clients=1
peft_scheme="lora"
max_steps=10
val_check_interval=5
num_rounds=3
lr=1e-4
job_name=f"peft_{peft_scheme}_fedavg_345M_lr{lr}_steps{max_steps}_val{val_check_interval}_rounds{num_rounds}_{n_clients}clients"

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
try:
    subprocess.run(["python3", "create_configs.py", "--job_folder", f"jobs/{job_name}", 
                    "--num_clients", str(n_clients), 
                    "--max_steps", str(max_steps), 
                    "--val_check_interval", str(val_check_interval), #str(max_steps),
                    "--num_rounds", str(num_rounds),
                    "--lr", str(lr),
                    "--peft_scheme", peft_scheme])
except subprocess.CalledProcessError as e:
    raise RuntimeError(f"Create_configs failed with {e.output}")

# Start FL simulation
simulator = SimulatorRunner(
    job_folder=f"jobs/{job_name}",
    workspace=f"/tmp/nvflare/nemo/{job_name}",
    n_clients=n_clients,
    threads=n_clients,
    #gpu="0,1,2"
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
