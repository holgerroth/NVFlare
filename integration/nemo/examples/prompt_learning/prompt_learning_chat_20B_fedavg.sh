#!/usr/bin/env bash
NVFLARE_ROOT=/home/hroth/Code/nvflare/nemo_nvflare
####export PYTHONPATH=$NVFLARE_ROOT/integration/nemo
export PYTHONPATH=$NVFLARE_ROOT:$NVFLARE_ROOT/integration/nemo

n_clients=3

data_dir="data/SFT_Data/Data_ptuning/alpaca-cleaned/alpaca_data_cleaned/3-clients"
#model_root="/home/hroth/llm_models"

# create configs
job_name="gpt_p-tuning_fedavg_20B_chat"

job_path="jobs/$job_name"
rm -r "$job_path"
python3 create_configs.py --job_folder="$job_path" --num_clients="$n_clients" --devices 2 --aggregation_epochs=1 --num_rounds=50 --train_ds_files_prefix="$data_dir/site-" --train_ds_files_suffix="-training.jsonl" --val_ds_files="$data_dir/site-0-validation.jsonl"

# create gpus ids
#let "max_id = $n_clients - 1"
#gpu_ids="0"
#for id in $(eval echo "{1..$max_id}")
#do
#    gpu_ids="$gpu_ids,$id"
#done

# a100 
gpu_ids="[0,3],[4,5],[6,7]"
#gpu_ids="[0,3]"
#gpu_ids="0"

# simulate
python -m nvflare.private.fed.app.simulator.simulator "$job_path" -w "/tmp/nvflare/nemo/$job_name" -n $n_clients -t $n_clients --gpu "$gpu_ids"
