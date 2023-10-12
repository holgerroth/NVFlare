#!/usr/bin/python3

import argparse
import os
import glob
import subprocess
from nvflare import SimulatorRunner

n_clients=3
max_steps=100
val_check_interval=50
num_rounds=100
lr=1e-4

data_root = "/workspace/Data/NLP"

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--peft_scheme", type=str, help="PEFT scheme. Choose from 'ptuning', 'adapter', or 'lora'.")
args = parser.parse_args()
job_name=f"peft_{args.peft_scheme}_fedavg_345M_lr{lr}_steps{max_steps}_val10_rounds{num_rounds}_{n_clients}clients_2"

assert args.peft_scheme in ["ptuning", "adapter", "lora"], f"PEFT scheme {args.peft_scheme} not supported!"

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
sp = subprocess.run(["python3", "create_configs.py", "--job_folder", f"jobs/{job_name}", 
                    "--num_clients", str(n_clients), 
                    "--max_steps", str(max_steps), 
                    "--val_check_interval", str(val_check_interval),
                    "--num_rounds", str(num_rounds),
                    "--root_dir", data_root, 
                    "--lr", str(lr),
                    "--peft_scheme", args.peft_scheme])
if sp.returncode != 0:
    raise RuntimeError(f"Create_configs failed!")

# Start FL simulation
simulator = SimulatorRunner(
    job_folder=f"jobs/{job_name}",
    workspace=f"./results/F1_launch_once/{job_name}",
    n_clients=n_clients,
    threads=n_clients,
    gpu="0,1,2"
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
