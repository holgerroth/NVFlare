#!/usr/bin/env bash

n_clients=3

data_dir="data/SFT_Data/Data_ptuning/alpaca-cleaned/alpaca_data_cleaned/3-clients"

# create configs
python3 create_configs.py --job_folder="jobs/gpt_p-tuning_local_345M_chat" --num_clients="$n_clients" --aggregation_epochs=50 --num_rounds=1 --train_ds_files_prefix="$data_dir/site-" --train_ds_files_suffix="-training.jsonl" --val_ds_files="$data_dir/site-0-validation.jsonl"

# create gpus ids
let "max_id = $n_clients - 1"
gpu_ids="0"
for id in $(eval echo "{1..$max_id}")
do
    gpu_ids="$gpu_ids,$id"
done

# simulate
nvflare simulator "jobs/gpt_p-tuning_local_345M_chat" -w "/tmp/nvflare/nemo/gpt_p-tuning_local_345M_chat" -n $n_clients -t $n_clients --gpu "$gpu_ids"
