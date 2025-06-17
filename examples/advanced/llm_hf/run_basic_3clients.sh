python3 llm_hf_fl_job_multigpu.py \
    --client_ids dolly alpaca oasst1 \
    --data_path ${PWD}/dataset \
    --workspace_dir ${PWD}/workspace/hf_sft_multi_3clients \
    --job_dir ${PWD}/workspace/jobs/hf_sft_multi_3clients \
    --train_mode SFT \
    --threads 1 \
    --model_name_or_path allenai/OLMo-2-0425-1B \
    --gpu "[0,1],[2,3],[0,1]" 

# use cached snapshot if HF rate limit hits
#     --model_name_or_path allenai/OLMo-2-0425-1B \
#     --model_name_or_path ~/.cache/huggingface/hub/models--meta-llama--llama-3.2-1b/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/ \
#     --model_name_or_path ~/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf/ \
#     --quantize_mode float16 \