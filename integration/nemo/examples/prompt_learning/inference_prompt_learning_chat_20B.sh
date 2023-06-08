#!/usr/bin/env bash
NVFLARE_ROOT=/home/hroth/Code/nvflare/nemo_nvflare
####export PYTHONPATH=$NVFLARE_ROOT/integration/nemo
export PYTHONPATH=$NVFLARE_ROOT:$NVFLARE_ROOT/integration/nemo

n_clients=3

#data_list="data/SFT_Data/Data_ptuning/alpaca-cleaned/alpaca_data_cleaned/3-clients"
#model_root="/home/hroth/llm_models"


export CUDA_VISIBLE_DEVICES="1,2"
python3 -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12345 inference.py 2>&1 | tee inference.log
