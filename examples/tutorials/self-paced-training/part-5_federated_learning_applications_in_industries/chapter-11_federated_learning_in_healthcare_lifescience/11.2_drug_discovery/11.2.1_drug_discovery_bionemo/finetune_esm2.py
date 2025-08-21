# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copied and adapted for NVFlare from https://github.com/NVIDIA/bionemo-framework/blob/main/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/finetune_esm2.py

import shutil
from pathlib import Path
from typing import List, Optional, Tuple, get_args

from lightning.pytorch.callbacks import Callback, LearningRateMonitor, RichModelSummary
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.esm2.model.finetune.peft import ESM2LoRA
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.config import TorchmetricsConfig
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger

from bionemo.esm2.scripts.finetune_esm2 import get_parser, SUPPORTED_DATASETS, SUPPORTED_CONFIGS

# (1) import nvflare lightning client API
import nvflare.client.lightning as flare


def train_model(
    train_data_path: Path,
    valid_data_path: Path,
    num_nodes: int = 1,
    num_gpus: int = 1,
    min_seq_length: Optional[int] = None,
    max_seq_length: int = 512,
    result_dir: Path = Path("./results"),
    num_steps: int = 500_000,
    max_epochs: int = 500_000,
    limit_val_batches: int = 1000,
    limit_test_batches: int = 1000,
    val_check_interval: int = 20,
    log_every_n_steps: int = 1,
    num_dataset_workers: int = 8,
    no_persistent_workers: bool = False,
    no_pin_memory: bool = False,
    lr: float = 4e-4,
    micro_batch_size: int = 64,
    accumulate_grad_batches: int = 1,
    experiment_name: str = "esm2-finetune",
    resume_if_exists: bool = False,
    precision: str = "bf16-mixed",
    task_type: str = "regression",
    encoder_frozen: bool = False,
    scale_lr_layer: Optional[str] = None,
    lr_multiplier: float = 1.0,
    mlp_ft_dropout: float = 0.25,
    mlp_hidden_size: int = 256,
    mlp_target_size: int = 1,
    cnn_dropout: float = 0.25,
    cnn_hidden_size: int = 32,
    cnn_num_classes: int = 3,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_offline: bool = False,
    wandb_tags: Optional[List[str]] = None,
    wandb_group: Optional[str] = None,
    wandb_id: Optional[str] = None,
    wandb_anonymous: bool = False,
    wandb_log_model: bool = False,
    pipeline_model_parallel_size: int = 1,
    tensor_model_parallel_size: int = 1,
    create_tensorboard_logger: bool = False,
    restore_from_checkpoint_path: Optional[Path] = None,
    save_last_checkpoint: bool = True,
    metric_to_monitor_for_checkpoints: str = "val_loss",
    save_top_k: int = 2,
    nsys_profiling: bool = False,
    nsys_start_step: int = 0,
    nsys_end_step: Optional[int] = None,
    nsys_ranks: List[int] = [0],
    dataset_class: str = "InMemorySingleValueDataset",
    config_class: str = "ESM2FineTuneSeqConfig",
    metric_tracker=None,
    overlap_grad_reduce: bool = False,
    no_overlap_param_gather: bool = False,
    no_average_in_collective: bool = False,
    grad_reduce_in_fp32: bool = False,
    no_ckpt_async_save: bool = True,  # Keep this True for FL
    label_column: str = "labels",
    labels_mask_column: Optional[str] = None,
    lora_checkpoint_path: Optional[Path] = None,
    lora_finetune: bool = False,
    classes: List[str] = None,
) -> Tuple[Path, Callback | None, nl.Trainer]:
    config_class = SUPPORTED_CONFIGS[config_class]
    dataset_class = SUPPORTED_DATASETS[dataset_class]

    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup the strategy and trainer
    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=num_gpus,
        accumulate_grad_batches=accumulate_grad_batches,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    # Convert lora_checkpoint_path to string if it's a Path object
    if lora_checkpoint_path is not None:
        lora_checkpoint_path = str(lora_checkpoint_path)

    # Initialize LoRA adapter first if needed
    peft = None
    if lora_finetune:
        peft = ESM2LoRA(peft_ckpt_path=lora_checkpoint_path)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=False,  # do not use `ckpt_async_save=True` as the checkpoint might still be saved while the next round already removed that saving directory
        ckpt_parallel_load=True,
        ckpt_load_strictness=StrictHandling.LOG_UNEXPECTED,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=overlap_grad_reduce,
            overlap_param_gather=not no_overlap_param_gather,
            average_in_collective=not no_average_in_collective,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            use_distributed_optimizer=False,
        ),
    )

    # for wandb integration
    # Please refer to https://pytorch-lightning.readthedocs.io/en/0.7.6/api/lightning.pytorch.loggers.html"
    wandb_config: Optional[WandbConfig] = (
        None
        if wandb_project is None
        else WandbConfig(
            offline=wandb_offline,
            project=wandb_project,
            entity=wandb_entity,
            tags=wandb_tags,
            group=wandb_group,
            id=wandb_id,
            anonymous=wandb_anonymous,
            log_model=wandb_log_model,
        )
    )

    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        nl_callbacks.PreemptionCallback(),
    ]
    if metric_tracker is not None:
        callbacks.append(metric_tracker)
    if nsys_profiling:
        if nsys_end_step is None:
            nsys_end_step = num_steps
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=nsys_start_step, end_step=nsys_end_step, ranks=nsys_ranks, gen_shape=True
            )
        )
    if peft is not None:
        callbacks.append(peft)

    tokenizer = get_tokenizer()

    # Initialize the data module.
    train_dataset = dataset_class.from_csv(
        train_data_path, task_type=task_type, label_column=label_column, labels_mask_column=labels_mask_column
    )
    valid_dataset = dataset_class.from_csv(
        valid_data_path, task_type=task_type, label_column=label_column, labels_mask_column=labels_mask_column
    )
    if task_type == "classification":
        if classes:
            if not isinstance(classes, List):
                raise ValueError(f"classes is expected to be list of strings but received {type(classes)}: {classes}")
            train_dataset.label_tokenizer.build_vocab([classes])
            print(f"Build custom label tokenizer based on label classes: {classes}")

    data_module = ESM2FineTuneDataModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        num_workers=num_dataset_workers,
        persistent_workers=not no_persistent_workers,
        pin_memory=not no_pin_memory,
        tokenizer=tokenizer,
    )
    # Configure the model
    train_metric = None
    is_model_parallel = tensor_model_parallel_size * pipeline_model_parallel_size > 1
    if is_model_parallel:
        valid_metric = None  # metric logging under model parallelism is not supported yet
    elif task_type == "regression":
        valid_metric = TorchmetricsConfig(class_path="MeanSquaredError", task="regression", metric_name="val_mse")
    elif task_type == "classification":
        valid_metric = TorchmetricsConfig(
            class_path="Accuracy",
            task="classification",
            kwargs={
                "task": "multiclass",
                "threshold": 0.5,
                "num_classes": data_module.train_dataset.label_tokenizer.vocab_size,
            },
            metric_name="val_acc",
        )
    else:
        raise ValueError(f"Task type {task_type} not supported. Supported task types are: classification, regression")
    config = config_class(
        task_type=task_type,
        encoder_frozen=encoder_frozen,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        initial_ckpt_path=str(restore_from_checkpoint_path),
        initial_ckpt_skip_keys_with_these_prefixes=[f"{task_type}_head"],
        train_metric=train_metric,
        valid_metric=valid_metric,
    )
    # Mapping of task-dependent config attributes to their new values
    task_dependent_attr = {
        "mlp_ft_dropout": mlp_ft_dropout,
        "mlp_hidden_size": mlp_hidden_size,
        "mlp_target_size": mlp_target_size,
        "cnn_dropout": cnn_dropout,
        "cnn_hidden_size": cnn_hidden_size,
        "cnn_num_classes": cnn_num_classes,
    }
    # Update attributes only if they exist in the config
    for attr, value in task_dependent_attr.items():
        if hasattr(config, attr):
            config.set_hparam(attr, value)

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_config=wandb_config,
    )

    # If client should save best local checkpoints, set to `save_local_ckpt=True`,
    save_local_ckpt = False
    if save_local_ckpt:    
        # Configure our custom Checkpointer
        checkpoint_path = str(Path(nemo_logger.save_dir) / "checkpoints")
        checkpoint_callback = nl_callbacks.ModelCheckpoint(
            dirpath=checkpoint_path,
            save_last=save_last_checkpoint,
            monitor=metric_to_monitor_for_checkpoints,  # "val_loss",
            save_top_k=save_top_k,
            every_n_train_steps=val_check_interval,
            always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
            filename="checkpoint-{step}-{consumed_samples}",  # Including step and consumed_samples in the checkpoint filename prevents duplicate filenames and bugs related to this.
            save_weights_only=False,
            save_optim_on_train_end=True,
        )
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None

    trainer = nl.Trainer(
        devices=num_gpus,
        max_steps=num_steps,
        max_epochs=max_epochs,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=1.0,  # frac of validation set.
        limit_test_batches=limit_test_batches,  # This controls upsampling and downsampling
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        num_nodes=num_nodes,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(
            precision=precision,
            params_dtype=get_autocast_dtype(precision),
            pipeline_dtype=get_autocast_dtype(precision),
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            autocast_enabled=False,
        ),
        enable_checkpointing=True,
    )

    # (2) patch the lightning trainer
    flare.patch(trainer, restore_state=False, load_state_dict_strict=False)

    # (3) receives FLModel from NVFlare
    # Note that we don't need to pass this input_model to trainer
    # because after flare.patch the trainer.fit/validate will get the
    # global model internally
    input_model = flare.receive()
    print(
        f"\n[Current Round={input_model.current_round}, Site = {flare.get_site_name()}, Global model = {input_model} ({len(input_model.params)} params)]\n"
    )


    # add NVFlare metric streamer to capture continues tensorboard output on the server.
    from bionemo_tb_streamer import BioNeMoTBStreamer

    trainer.callbacks.append(BioNeMoTBStreamer(start_step=input_model.current_round * num_steps))

    # use a unique result directory for each round
    # Remove previous checkpoints to preserve disk space
    keep_last_ckpt_only = True  # TODO: make configurable
    if keep_last_ckpt_only:
        previous_ckpt_dir = (
            result_dir / f"round{input_model.current_round - 1}" / experiment_name / "dev" / "checkpoints"
        )
        if previous_ckpt_dir.is_dir():
            print(f"Removing previous checkpoint directory {previous_ckpt_dir}")
            shutil.rmtree(previous_ckpt_dir)

    # create output folder for this round
    result_dir = result_dir / f"round{input_model.current_round}"

    # add a learning rate decay for each round
    if input_model.current_round > 0:
        lr_step_reduce = 1.05  # TODO: make lr_step_reduce configurable
        new_lr = lr / (input_model.current_round * lr_step_reduce)
        new_lr_multiplier = lr_multiplier / (input_model.current_round * lr_step_reduce)
        print(f"Reduce lr {lr} by {input_model.current_round * lr_step_reduce}: {new_lr}")
    else:
        new_lr = lr
        new_lr_multiplier = lr_multiplier

    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=new_lr,
            optimizer="adam",  # fused_adam not supported
            use_distributed_optimizer=True,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.98,
            clip_grad=1.0,
        ),
    )
    # fiddle is not serializing lambda fn
    # to bypass serialization of lambda fn scale_lr_condition as part of optimizer configuration
    if scale_lr_layer:
        optimizer.scale_lr_cond = lambda name, param: scale_lr_layer in name
        optimizer.lr_mult = new_lr_multiplier

    if peft is not None:
        module = biobert_lightning_module(
            config=config, tokenizer=tokenizer, optimizer=optimizer, model_transform=peft
        )
    else:
        module = biobert_lightning_module(config=config, tokenizer=tokenizer, optimizer=optimizer)

    llm.train(
        model=module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=None,  # No resume for FL
    )

    if checkpoint_callback:
        ckpt_path = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
    else:
        ckpt_path = None
    return ckpt_path, metric_tracker, trainer


def finetune_esm2_entrypoint() -> Tuple[Path, Callback | None, nl.Trainer]:
    """Train an ESM2 model on UR data."""
    parser = get_parser()

    # Add some FL specific arguments
    parser.add_argument(
        "--classes",
        type=str,
        required=False,
        default=None,
        help="Unique strings describing the classes for classification. Used to build the same label vocabulary on each client. Should be comma separate list of strings, e.g. 'pos,neg'",
    )    
    args = parser.parse_args()

    if args.classes:
        if args.task_type != "classification":
            parser.error("Use --classes argument only with --task-type 'classification'")
        args.classes = args.classes.split(",")
    else:
        args.classes = None    

    # Validate arguments
    if args.lora_checkpoint_path and not args.lora_finetune:
        raise ValueError("Arguments --lora-checkpoint-path cannot be set when not using lora-finetune.")
    if args.precision not in get_args(PrecisionTypes):
        raise ValueError(f"Precision {args.precision} not supported. Supported precisions are: {PrecisionTypes}")
    if args.task_type not in ["classification", "regression"]:
        raise ValueError(
            f"Task type {args.task_type} not supported. Supported task types are: classification, regression"
        )
    if args.dataset_class not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset class {args.dataset_class} not supported. Supported dataset classes are: {SUPPORTED_DATASETS.keys()}"
        )
    if args.config_class not in SUPPORTED_CONFIGS:
        raise ValueError(
            f"Config class {args.config_class} not supported. Supported config classes are: {SUPPORTED_CONFIGS.keys()}"
        )
    if args.min_seq_length is not None and args.dataset_class == "InMemorySingleValueDataset":
        raise ValueError("Arguments --min-seq-length cannot be set when using InMemorySingleValueDataset.")

    train_model(**vars(args))


if __name__ == "__main__":
    finetune_esm2_entrypoint()
    flare.shutdown()
