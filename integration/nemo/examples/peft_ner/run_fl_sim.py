#!/usr/bin/python3

import os
import glob
import subprocess
from nvflare import SimulatorRunner

n_clients=4
peft_scheme="ptuning"
peft_scheme="adapter"
peft_scheme="lora"
max_steps=2000
val_check_interval=100
num_rounds=50
lr=5e-3
job_name=f"peft_{peft_scheme}_fedavg_345M_lr{lr}_steps{max_steps}_val10_rounds{num_rounds}_{n_clients}clients_f1fg_5"


data_root = f"/workspace/Data/NLP/NCBI-disease/NCBI-disease-20230831T023848Z-001/NCBI-disease/{n_clients}_split"

def clean_files(data_root, ext):
    files = glob.glob(os.path.join(data_root, "**", ext), recursive=True)
    for file in files:
        print(file)
        os.remove(file)

def clean_memmap(data_root):
    clean_files(data_root, "*.npy")
    clean_files(data_root, "*.info")

# Clean temporary data
clean_memmap(data_root)

# Create configurations
sp = subprocess.run(["python3", "create_configs.py", "--job_folder", f"jobs/{job_name}", 
                    "--num_clients", str(n_clients), 
                    "--max_steps", str(max_steps), 
                    "--val_check_interval", str(val_check_interval),
                    "--num_rounds", str(num_rounds),
                    "--root_dir", data_root,
                    "--lr", str(lr),
                    "--peft_scheme", peft_scheme])
if sp.returncode != 0:
    raise RuntimeError(f"Create_configs failed!")

# DEBUG
#n_clients = 1

# Start FL simulation
simulator = SimulatorRunner(
    job_folder=f"jobs/{job_name}",
    workspace=f"./results/F1_launch_once/{job_name}",
    n_clients=n_clients,
    threads=n_clients,
    gpu="0,1,2,3"
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
