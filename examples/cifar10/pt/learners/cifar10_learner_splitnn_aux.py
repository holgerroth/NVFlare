# Copyright (c) 2021, NVIDIA CORPORATION.
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

import os
import pickle
import numpy as np
from timeit import default_timer as timer
import torch
import torch.optim as optim
from pt.networks.cifar10_nets import ModerateCNN
from pt.utils.cifar10_dataset import CIFAR10_Idx, CIFAR10SplitNN
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from nvflare.app_common.pt.pt_fedproxloss import PTFedProxLoss

from pt.learners.cifar10_learner import CIFAR10Learner
from nvflare.fuel.utils import fobs
from nvflare.app_common.pt.pt_decomposers import TensorDecomposer


def print_grads(net):
    for name, param in net.named_parameters():
        if param.grad is not None:
            print(name, "grad", param.grad.shape, torch.sum(param.grad).item())
        else:
            print(name, "grad", None)


class SplitNNConstants(object):
    BATCH_INDICES = "_splitnn_batch_indices_"
    ACTIVATIONS = "_splitnn_activations_"
    GRADIENT = "_splitnn_gradient_"

    TASK_INIT_MODEL = "_splitnn_task_init_model_"
    TASK_DATA_STEP = "_splitnn_task_data_step_"
    TASK_LABEL_STEP = "_splitnn_task_label_step_"
    TASK_BACKWARD_STEP = "_splitnn_task_backward_step_"
    TASK_TRAIN = "_splitnn_task_train_"

    TASK_RESULT = "_splitnn_task_result_"
    TIMEOUT = 60.0  # timeout for waiting for reply from aux message request


class CIFAR10LearnerSplitNNAux(Learner):
    def __init__(
        self,
        dataset_root: str = "./dataset",
        init_model_task=SplitNNConstants.TASK_INIT_MODEL,
        data_step_task=SplitNNConstants.TASK_DATA_STEP,
        label_step_task=SplitNNConstants.TASK_LABEL_STEP,
        data_backward_step_task=SplitNNConstants.TASK_BACKWARD_STEP,
        lr: float = 1e-2,
        analytic_sender_id: str = "analytic_sender",
        model: dict = None,
        timeit: bool = False
    ):
        """Simple CIFAR-10 Trainer.

        Args:
            dataset_root: directory with CIFAR-10 data.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            train_task_name: name of the task to train the model.
            submit_model_task_name: name of the task to submit the best local model.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component. If configured, TensorBoard events will be fired. Defaults to "analytic_sender".

        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.init_model_task = init_model_task
        self.data_step_task = data_step_task
        self.label_step_task = label_step_task
        self.data_backward_step_task = data_backward_step_task
        self.lr = lr
        self.model = model
        self.analytic_sender_id = analytic_sender_id

        self.app_root = None
        self.current_round = None
        self.num_rounds = None
        self.writer = None
        self.client_name = None
        self.device = None
        self.optimizer = None
        self.criterion = None
        self.transform_train = None
        self.train_dataset = None
        self.split_id = None
        self.activations = None

        # use FOBS serializing/deserializing PyTorch tensors
        fobs.register(TensorDecomposer)

        self.timeit = timeit
        self.times = {}
        if self.timeit:
            self.times["learner_start_data_step"] = []
            self.times["learner_end_data_step"] = []
            self.times["learner_start_label_step"] = []
            self.times["learner_end_label_step"] = []
            self.times["learner_start_backward_step"] = []
            self.times["learner_end_backward_step"] = []
            self.times["aux_hdl_learner_start_data_train_back_step"] = []
            self.times["aux_hdl_learner_end_data_train_back_step"] = []
            self.times["aux_hdl_learner_start_data_train_step"] = []
            self.times["aux_hdl_learner_end_data_train_step"] = []
            self.times["aux_hdl_learner_start_label_train_step"] = []
            self.times["aux_hdl_learner_end_label_train_step"] = []
            self.times["aux_hdl_learner_start_data_backward_step"] = []
            self.times["aux_hdl_learner_end_data_backward_step"] = []

    def _get_model(self, fl_ctx: FLContext):
        # TODO: is this whole logic needed?
        if isinstance(self.model, str):
            # treat it as model component ID
            model_component_id = self.model
            engine = fl_ctx.get_engine()
            self.model = engine.get_component(model_component_id)
            if not self.model:
                self.log_error(fl_ctx, f"cannot find model component '{model_component_id}'")
                return
        if self.model and isinstance(self.model, dict):
            # try building the model
            try:
                engine = fl_ctx.get_engine()
                # use provided or default optimizer arguments and add the model parameters
                if "args" not in self.model:
                    self.model["args"] = {}
                self.model = engine.build_component(self.model)
            except BaseException as e:
                self.system_panic(
                    f"Exception while parsing `model`: " f"{self.model} with Exception {e}",
                    fl_ctx,
                )
                return
        if self.model and not isinstance(self.model, torch.nn.Module):
            print("@@@@@@@@@@@ self.model", self.model)
            self.log_error(fl_ctx, f"expect model to be torch.nn.Module but got {type(self.model)}: {self.model}")
            return
        if self.model is None:
            self.log_error(fl_ctx, f"Model wasn't built correctly! It is {self.model}")
            return
        self.log_info(fl_ctx, "Running model", self.model)

    def initialize(self, parts: dict, fl_ctx: FLContext):
        self._get_model(fl_ctx=fl_ctx)
        #print("@@@@@@@@@@@@ self.model", self.model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )

        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.client_name = fl_ctx.get_identity_name()
        self.split_id = self.model.get_split_id()
        self.log_info(fl_ctx, f"Running `split_id` {self.split_id} on site `{self.client_name}`")

        if self.split_id == 0:  # data side
            data_returns = "image"
        elif self.split_id == 1:  # label side
            data_returns = "label"
        else:
            raise ValueError(f"Expected split_id to be '0' or '1' but was {self.split_id}")

        self.train_dataset = CIFAR10SplitNN(
            root=self.dataset_root,
            train=True,
            download=True,
            transform=self.transform_train,
            returns=data_returns
        )

        # Select local TensorBoard writer or event-based writer for streaming
        self.writer = parts.get(self.analytic_sender_id)  # user configured config_fed_client.json for streaming
        if not self.writer:  # use local TensorBoard writer only
            self.writer = SummaryWriter(self.app_root)

        # register aux message handlers
        engine = fl_ctx.get_engine()

        engine.register_aux_message_handler(topic=self.data_step_task, message_handle_func=self.train_backward_data_side)
        engine.register_aux_message_handler(topic=self.label_step_task, message_handle_func=self.train_label_side)
        self.log_info(fl_ctx, "Registered aux message handlers")

    """ training steps """
    def train_step_data_side(self, batch_indices):
        if self.timeit:
            self.times["learner_start_data_step"].append(timer())
        self.model.train()

        inputs = self.train_dataset.get_batch(batch_indices)
        inputs = inputs.to(self.device)

        self.activations = self.model.forward(inputs)  # keep on site-1
        if self.timeit:
            self.times["learner_end_data_step"].append(timer())
        return self.activations.detach().requires_grad_()  # x to be sent to other client

    def train_step_label_side(self, batch_indices, activations, fl_ctx: FLContext):
        if self.timeit:
            self.times["learner_start_label_step"].append(timer())
        self.model.train()
        self.optimizer.zero_grad()
        print("=============================================")
        print("222###### activations", "requires_grad", activations.requires_grad, "is_leaf", activations.is_leaf)
        print("=============================================")

        labels = self.train_dataset.get_batch(batch_indices)
        labels = labels.to(self.device)
        activations = activations.to(self.device)
        activations.requires_grad_(True)

        print("=============================================")
        print("333###### activations", "requires_grad", activations.requires_grad, "is_leaf", activations.is_leaf)
        print("=============================================")

        pred = self.model.forward(activations)
        loss = self.criterion(pred, labels)
        loss.backward()

        self.log_info(fl_ctx, f"Round {self.current_round}/{self.num_rounds} train_loss: {loss.item():.4f}")
        if self.writer:
            self.writer.add_scalar("train_loss", loss.item(), self.current_round)

        print(f"====== {self.client_name} Model with `split_id` {self.split_id} train_step_label_side grad: ======")
        #print_grads(self.model)

        self.optimizer.step()
        if self.timeit:
            self.times["learner_end_label_step"].append(timer())

        print("%%%%%%%%%%%32434322344 activations.grad", type(activations.grad))
        if not isinstance(activations.grad, torch.Tensor):
            raise ValueError("No valid gradients available!")
        return activations.grad  # gradient to be returned to other client

    def backward_step_data_side(self, gradient):
        if self.timeit:
            self.times["learner_start_backward_step"].append(timer())
        self.optimizer.zero_grad()
        print("!!!!!!!!!!!!!!!!!!gradient", type(gradient))
        gradient = gradient.to(self.device)
        self.activations.backward(gradient=gradient)
        self.optimizer.step()

        print(f"====== {self.client_name} Model with `split_id` {self.split_id} backward_step_data_side grad: ======")
        #print_grads(self.model)
        if self.timeit:
            self.times["learner_end_backward_step"].append(timer())

    """ message_handle_func functions """
    def train_backward_data_side(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        if self.timeit:
            self.times["aux_hdl_learner_start_data_train_back_step"].append(timer())
        # combine forward and backward on data client
        # 1. perform backward step if gradients provided
        dxo = from_shareable(request)
        gradient = dxo.get_meta_prop(SplitNNConstants.GRADIENT)
        if gradient is not None:
            result_backward = self.backward_data_side(topic=topic, request=request, fl_ctx=fl_ctx)
            assert result_backward.get_return_code() == ReturnCode.OK, \
                f"Backward step failed with return code {result_backward.get_return_code()}"
        # 2. compute activations
        results_activations = self.train_data_side(topic=topic, request=request, fl_ctx=fl_ctx)
        if self.timeit:
            self.times["aux_hdl_learner_end_data_train_back_step"].append(timer())
        return results_activations

    def train_data_side(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        if self.timeit:
            self.times["aux_hdl_learner_start_data_train_step"].append(timer())
        if self.split_id != 0:
            raise ValueError(f"Expected `split_id` 0. It doesn't make sense to run `train_data_side` with `split_id` {self.split_id}")

        self.current_round = request.get_header(AppConstants.CURRENT_ROUND)
        self.num_rounds = request.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Train data side in round {self.current_round} of {self.num_rounds} rounds.")

        dxo = from_shareable(request)
        batch_indices = dxo.get_meta_prop(SplitNNConstants.BATCH_INDICES)
        if batch_indices is None:
            raise ValueError("No batch indices in DXO!")

        activations = self.train_step_data_side(batch_indices=batch_indices)

        print(f"====== {self.client_name} Model with `split_id` {self.split_id} train_data_side finished:")
        self.log_info(fl_ctx, "train_data_side finished.")

        return_shareable = DXO(data={}, data_kind=DataKind.WEIGHT_DIFF, meta={SplitNNConstants.ACTIVATIONS: fobs.dumps(activations)}).to_shareable()
        if self.timeit:
            self.times["aux_hdl_learner_end_data_train_step"].append(timer())
        self.log_info(fl_ctx, f"Sending train data return_shareable: {type(return_shareable)}")
        return return_shareable

    def train_label_side(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        if self.timeit:
            self.times["aux_hdl_learner_start_label_train_step"].append(timer())
        if self.split_id != 1:
            raise ValueError(f"Expected `split_id` 1. It doesn't make sense to run `train_label_side` with `split_id` {self.split_id}")

        self.current_round = request.get_header(AppConstants.CURRENT_ROUND)
        self.num_rounds = request.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Train label in round {self.current_round} of {self.num_rounds} rounds.")

        dxo = from_shareable(request)
        batch_indices = dxo.get_meta_prop(SplitNNConstants.BATCH_INDICES)
        if batch_indices is None:
            raise ValueError("No batch indices in DXO!")

        activations = dxo.get_meta_prop(SplitNNConstants.ACTIVATIONS)
        if activations is None:
            raise ValueError("No activations in DXO!")

        gradient = self.train_step_label_side(batch_indices=batch_indices, activations=fobs.loads(activations), fl_ctx=fl_ctx)

        self.log_info(fl_ctx, "train_label_side finished.")
        return_shareable = DXO(data={}, data_kind=DataKind.WEIGHT_DIFF, meta={SplitNNConstants.GRADIENT: fobs.dumps(gradient)}).to_shareable()
        if self.timeit:
            self.times["aux_hdl_learner_end_label_train_step"].append(timer())

        self.log_info(fl_ctx, f"Sending train label return_shareable: {type(return_shareable)}")
        return return_shareable

    def backward_data_side(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        if self.timeit:
            self.times["aux_hdl_learner_start_data_backward_step"].append(timer())
        if self.split_id != 0:
            raise ValueError(f"Expected `split_id` 0. It doesn't make sense to run `backward_data_side` with `split_id` {self.split_id}")

        dxo = from_shareable(request)
        gradient = dxo.get_meta_prop(SplitNNConstants.GRADIENT)
        if gradient is None:
            raise ValueError("No gradient in DXO!")
        self.backward_step_data_side(gradient=fobs.loads(gradient))

        self.log_info(fl_ctx, "backward_data_side finished.")
        if self.timeit:
            self.times["aux_hdl_learner_end_data_backward_step"].append(timer())
        return make_reply(ReturnCode.OK)

    # Task function (one time only in beginning)
    # TODO: is model init really needed?  -> Probably helps
    def init_model(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data
        print("##########!!!!!!! global_weights", global_weights.keys())

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        print("##########!!!!!!! local_var_dict", local_var_dict.keys())
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)  # TODO: check if this is needed for SplitNN
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                    n_loaded += 1
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError("No global weights loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # TODO: this doesn't work anymore for cifar10!

        self.log_info(fl_ctx, "init_model finished.")
        return make_reply(ReturnCode.OK)

    def finalize(self, fl_ctx: FLContext):
        if self.timeit:
            app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
            with open(os.path.join(app_root, "learner_times.pkl"), "wb") as f:
                pickle.dump(self.times, f)
