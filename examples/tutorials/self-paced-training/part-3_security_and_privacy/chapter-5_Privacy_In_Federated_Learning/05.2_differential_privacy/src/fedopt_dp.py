# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


from typing import Union

import torch
from copy import deepcopy

from nvflare.app_opt.pt.fedopt_ctl import FedOpt
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.security.logging import secure_format_exception


class FedOptDP(FedOpt):
    def __init__(
        self,
        *args,
        source_model: Union[str, torch.nn.Module],
        optimizer_args: dict = {
            "path": "torch.optim.SGD",
            "args": {"lr": 1.0, "momentum": 0.6},
        },
        lr_scheduler_args: dict = {
            "path": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "args": {"T_max": 3, "eta_min": 0.9},
        },
        device=None,
        target_epsilon=None,
        target_delta: float=1e-5,
        max_grad_norm: float=1.0,
        **kwargs,
    ):
        """Implement the FedOpt algorithm. Based on FedAvg ModelController.

        The algorithm is proposed in Reddi, Sashank, et al. "Adaptive federated optimization." arXiv preprint arXiv:2003.00295 (2020).
        After each round, update the global model using the specified PyTorch optimizer and learning rate scheduler.
        Note: This class will use FedOpt to optimize the global trainable parameters (i.e. `self.torch_model.named_parameters()`)
        but use FedAvg to update any other layers such as batch norm statistics.

        Args:
            source_model: component id of torch model object or a valid torch model object
            optimizer_args: dictionary of optimizer arguments, with keys of 'optimizer_path' and 'args.
            lr_scheduler_args: dictionary of server-side learning rate scheduler arguments, with keys of 'lr_scheduler_path' and 'args.
            device: specify the device to run server-side optimization, e.g. "cpu" or "cuda:0"
                (will default to cuda if available and no device is specified).
            target_epsilon: Target epsilon to be achieved, a metric of privacy loss at differential changes in data. "The target δ of the (ϵ,δ)-differential privacy guarantee. Generally, it should be set to be less than the inverse of the size of the training dataset" (from https://opacus.ai/tutorials/building_image_classifier).
            target_delta: Target delta to be achieved. Probability of information being leaked.
            max_grad_norm: The maximum norm of the per-sample gradients. Any gradient with norm
                higher than this will be clipped to this value.
        Raises:
            TypeError: when any of input arguments does not have correct type
        """
        FedAvg.__init__(self, *args, **kwargs)

        self.source_model = source_model
        self.optimizer_args = optimizer_args
        self.lr_scheduler_args = lr_scheduler_args
        self.device = device

        self.torch_model = None
        self.optimizer = None
        self.lr_scheduler = None

        # privacy args
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm

    def run(self):
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device)

        # set up source model
        if isinstance(self.source_model, str):
            self.torch_model = self.get_component(self.source_model)
        else:
            self.torch_model = self.source_model

        if self.torch_model is None:
            self.panic("Model is not available")
            return
        elif not isinstance(self.torch_model, torch.nn.Module):
            self.panic(f"expect model to be torch.nn.Module but got {type(self.torch_model)}")
            return
        else:
            print("server model", self.torch_model)
        self.torch_model.to(self.device)

        # set up optimizer
        try:
            if "args" not in self.optimizer_args:
                self.optimizer_args["args"] = {}
            self.optimizer_args["args"]["params"] = self.torch_model.parameters()
            self.optimizer = self.build_component(self.optimizer_args)
        except Exception as e:
            error_msg = f"Exception while constructing optimizer: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)
            return

        # set up lr scheduler
        try:
            if "args" not in self.lr_scheduler_args:
                self.lr_scheduler_args["args"] = {}
            self.lr_scheduler_args["args"]["optimizer"] = self.optimizer
            self.lr_scheduler = self.build_component(self.lr_scheduler_args)
        except Exception as e:
            error_msg = f"Exception while constructing lr_scheduler: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)
            return
        
        # add privacy
        if self.target_epsilon:
            try:
                # Following https://github.com/pytorch/opacus/blob/main/Migration_Guide.md for using Opacus without data loader
                self.target_delta = 1e-5  # TODO: make configurable

                print(f"Adding privacy engine with epsilon={self.target_epsilon}, delta={self.target_delta}")
                # initialize privacy accountant
                from opacus.accountants import RDPAccountant
                self.accountant = RDPAccountant()

                # wrap model
                from opacus import GradSampleModule
                self.torch_model = GradSampleModule(self.torch_model)

                # wrap optimizer
                from opacus.optimizers import DPOptimizer
                self.optimizer = DPOptimizer(
                optimizer=self.optimizer,
                noise_multiplier=1.0, # same as make_private arguments
                max_grad_norm=1.0, # same as make_private arguments
                expected_batch_size=1 # if you're averaging your gradients, you need to know the denominator
                )

                # attach accountant to track privacy for an optimizer
                self.optimizer.attach_step_hook(
                    self.accountant.get_optimizer_hook_fn(
                    # this is an important parameter for privacy accounting. Should be equal to batch_size / len(dataset)
                    sample_rate=self.target_delta
                    )
                )
                print("accountant added")
            except Exception as e:
                error_msg = f"Exception while adding privacy: {secure_format_exception(e)}"
                self.exception(error_msg)
                self.panic(error_msg)
                return
        
        FedAvg.run(self)  # run the standard FedAvg run routine

    def update_model(self, global_model: FLModel, aggr_result: FLModel):
        # Opacus adds "_module." prefix to state dict. Add it here to match optimizer's state dict
        dp_aggr_result = deepcopy(aggr_result)
        for k, v in aggr_result.params.items():
            dp_aggr_result.params[f"_module.{k}"] = v

        global_model = super().update_model(global_model, dp_aggr_result)
        epsilon = self.accountant.get_epsilon(self.target_delta)
        print(f"Training with privacy (ε = {epsilon:.2f}, δ = {self.target_delta})")
        return global_model
    