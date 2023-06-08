taskname = "chat"

test_examples = [
    {"taskname": "chat", "input": "Tell me an interesting fact about space travel."},
    {"taskname": "chat", "input": "Write a description of the presidential palace in Bolivia."},
    {"taskname": "chat", "input": "Analyze the relationship between economic status and crime in the United States."},
    {"taskname": "chat", "input": "Compose a story about an adventure on the high seas."},
    {"taskname": "chat", "input": "Describe the basic steps in painting a room."},
    {"taskname": "chat", "input": "Describe the traditional art of origami."},
]

config_filename = "/home/hroth/Results/poc_20B_chat/admin/transfer/45c73b8f-9e17-4d60-8d5e-f7af328e494c/job/gpt_p-tuning_fedavg_20B_chat/server/config/megatron_gpt_prompt_learning_config.yaml"
task_templates_filename = "/home/hroth/Results/poc_20B_chat/admin/transfer/45c73b8f-9e17-4d60-8d5e-f7af328e494c/job/gpt_p-tuning_fedavg_20B_chat/server/config/task_templates.json"
devices = 2

gpt_file_name = "/home/hroth/Code/nvflare/nemo_nvflare/integration/nemo/examples/prompt_learning/nemo-megatron-gpt-20B/nemo_gpt20B_bf16_tp2.nemo"
prompt_encoder_file_name = "/home/hroth/Results/poc_20B_chat/admin/transfer/45c73b8f-9e17-4d60-8d5e-f7af328e494c/workspace/app_server/best_FL_global_model.pt"

import os
import torch
import pytorch_lightning as pl
import pynvml
from nemo_nvflare.fed_megatron_gpt_prompt_learning_model import FedMegatronGPTPromptLearningModel
from nemo_nvflare.utils import load_weights
from omegaconf import OmegaConf
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

import subprocess as sp
import os

print(f"Running with distributed environment: LOCAL_RANK: {os.environ['LOCAL_RANK']}", 
      f"RANK: {os.environ['RANK']}, WORLD_SIZE {os.environ['WORLD_SIZE']}", 
      f"MASTER_ADDR: {os.environ['MASTER_ADDR']}, and MASTER_PORT: {os.environ['MASTER_PORT']}")

#def get_memory_free_MiB(gpu_index):
#    pynvml.nvmlInit()
#    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
#    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#    return mem_info.free // 1024 ** 2

#def get_gpu_memory():
#    memory = []
#    for gpu_index in range(pynvml.nvmlDeviceGetCount()):
#        memory.append(get_memory_free_MiB(gpu_index))
#    return memory

#print(f"LOCAL_RANK: {os.environ['LOCAL_RANK']} free GPU memory: {get_gpu_memory()}")

# Load model configuration used by one of the clients
print(f"Loading config from {config_filename}")
config = OmegaConf.load(config_filename)

# Set GPT model path
config.model.language_model_path = gpt_file_name

# Load task templates
print(f"Loading task templates from {task_templates_filename}")
config.model.task_templates = OmegaConf.load(task_templates_filename)

config.model.existing_tasks = []
# Set task that were learned
config.model.new_tasks = [taskname]

# Setup cluster environment parameters
# use torch elastic cluster environment so `create_process_externally` is True
# the launcher is set to None. It will not try to spawn new processes.
# It won't create the misconfiguration error because of the `interactive session`
#os.environ["LOCAL_RANK"] = '0'
#os.environ["RANK"] = '0'
#os.environ["WORLD_SIZE"] = '1'

print(f"Inference with global_batch_size {config.model.global_batch_size} and micro_batch_size {config.model.micro_batch_size}")

config.trainer.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
config.trainer.devices = devices
config.model.tensor_model_parallel_size = devices

##strategy = NLPDDPStrategy(parallel_devices=[torch.device("cuda:0"), torch.device("cuda:1")], find_unused_parameters=False, no_ddp_communication_hook=True)
strategy = NLPDDPStrategy(find_unused_parameters=False, no_ddp_communication_hook=True)
plugins = [TorchElasticEnvironment()]

# Set up the trainer and load the model that was used for p-tuning
trainer = pl.Trainer(plugins=plugins, strategy=strategy, **config.trainer)
#trainer = pl.Trainer(**config.trainer)
config.model.precision = config.trainer.precision

model = FedMegatronGPTPromptLearningModel(cfg=config.model, trainer=trainer)
model.init_prompt_encoder()

print("Model initialized", type(model))


# Overwrite the prompt encoder with the trained checkpoint
print(f"Restore checkpoint from {prompt_encoder_file_name}")
ckpt = torch.load(prompt_encoder_file_name)
global_weights = ckpt["model"]

n_loaded = load_weights(model, global_weights, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
print(f"Loaded {n_loaded} of {len(global_weights)} weights")

# Run the model
response = model.generate(inputs=test_examples, length_params=None)

print('The prediction results of some sample queries with the trained model:')
for result in response['sentences']:
    print(result)
    print("-" * 30)
