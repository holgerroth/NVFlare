#!/usr/bin/python3

import os
import glob
import sys
import subprocess
from nvflare import SimulatorRunner

n_clients=3
peft_scheme="ptuning"
max_steps=20
num_rounds=100
lr=1e-4
nemo_ckpt="megatron_gpt_345m_tp2.nemo"
devices=2

job_name=f"peft_{peft_scheme}_fedavg_345M_lr{lr}_steps{max_steps}_val10_rounds{num_rounds}_{n_clients}clients_devices{devices}"

data_root = "/home/hroth/Data/NLP"

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
                    "--val_check_interval", str(max_steps),
                    "--num_rounds", str(num_rounds),
                    "--lr", str(lr),
                    "--peft_scheme", peft_scheme,
                    "--root_dir", data_root,                      
                    "--nemo_ckpt", nemo_ckpt,
                    "--devices", str(devices)])
if sp.returncode != 0:
    raise RuntimeError(f"Create_configs failed!")


# Start FL simulation
simulator = SimulatorRunner(
    job_folder=f"jobs/{job_name}",
    workspace=f"./results/F1_launch_once_tp2/{job_name}",
    n_clients=n_clients,
    threads=n_clients,
    gpu="[6,7],[6,7],[6,7]"
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
