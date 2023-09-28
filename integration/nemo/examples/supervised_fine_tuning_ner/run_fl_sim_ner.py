#!/usr/bin/python3

import os
import glob
import subprocess
from nvflare import SimulatorRunner

n_clients=4
#max_steps=2000
#val_check_interval=100
#num_rounds=50
#lr=5e-3
#job_name=f"peft_sft_fedavg_345M_lr{lr}_steps{max_steps}_val10_rounds{num_rounds}_{n_clients}clients_f1fg_1"
job_name=f"peft_sft_fedavg_345M_{n_clients}clients_f1fg_1"


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

config_cmd = ["python3", "utils/create_configs.py", "--job_folder", f"jobs/{job_name}", 
                    "--num_clients", str(n_clients), 
                    "--devices", "1"]
config_cmd.append("--train_ds_files")
for i in range(n_clients):
    train_file = os.path.join(data_root,f"site-{i+1}_train.jsonl")
    assert os.path.isfile(train_file)
    config_cmd.append(train_file)
config_cmd.append("--validation_ds_files")
for i in range(n_clients):
    val_file = os.path.join(data_root,f"site-{i+1}_val.jsonl")
    assert os.path.isfile(val_file)    
    config_cmd.append(val_file)    
# Create configurations
sp = subprocess.run(config_cmd)
if sp.returncode != 0:
    raise RuntimeError(f"Create_configs failed!")

# DEBUG
#n_clients = 1

# Start FL simulation
simulator = SimulatorRunner(
    job_folder=f"jobs/{job_name}",
    workspace=f"../peft_ner/results/F1/{job_name}",
    n_clients=n_clients,
    threads=n_clients,
    gpu="0,1,2,3"
)
run_status = simulator.run()
print("Simulator finished with run_status", run_status)
