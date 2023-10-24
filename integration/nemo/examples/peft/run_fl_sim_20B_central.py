#!/usr/bin/python3

import os
import glob
import sys
import subprocess
from nvflare import SimulatorRunner

# TODO: combine with other script

n_clients=1
peft_scheme="ptuning"
max_steps=100
val_check_interval=100
num_rounds=10
lr=1e-4
nemo_ckpt="/data/Models/nemo-megatron-gpt-20B/nemo_gpt20B_bf16_tp2.nemo"
devices=2

job_name=f"peft_{peft_scheme}_central_20B_lr{lr}_steps{max_steps}_val10_rounds{num_rounds}_{n_clients}clients_devices{devices}_newsched_rounds"

data_root = "/home/hroth/Data/NLP"
train_ds_files_prefix = "FinancialPhraseBank-v1.0/financial_phrase_bank_train.jsonl"

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
                    "--lr", str(lr),
                    "--peft_scheme", peft_scheme,
                    "--root_dir", data_root,                      
                    "--nemo_ckpt", nemo_ckpt,
                    "--train_ds_files_prefix", train_ds_files_prefix,
                    "--devices", str(devices)])
if sp.returncode != 0:
    raise RuntimeError(f"Create_configs failed!")


# Start FL simulation
simulator = SimulatorRunner(
    job_folder=f"jobs/{job_name}",
    workspace=f"./results/F1_launch_once_tp2/{job_name}",
    n_clients=n_clients,
    threads=n_clients,
    gpu="[6,7]"
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
