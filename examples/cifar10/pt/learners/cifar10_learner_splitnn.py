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
import pickle  # TODO: used for benchmarking, remove later
from timeit import default_timer as timer

import numpy as np
import torch
import torch.optim as optim
from pt.utils.cifar10_dataset import CIFAR10SplitNN
from pt.workflows.splitnn_workflow import SplitNNConstants, SplitNNDataKind
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs


class CIFAR10LearnerSplitNN(Learner):
    def __init__(
        self,
        dataset_root: str = "./dataset",
        intersection_file: str = None,
        lr: float = 1e-2,
        model: dict = None,
        timeit: bool = False,  # TODO: remove option; only used for benchmarking
        analytic_sender_id: str = "analytic_sender",
        fp16: bool = True,
    ):
        """Simple CIFAR-10 Trainer for split learning.

        Args:
            dataset_root: directory with CIFAR-10 data.
            intersection_file: Optional. intersection file specifying overlapping indices between both clients.
                Defaults to `None`, i.e. the whole training dataset is used.
            lr: learning rate.
            model: Split learning model.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            fp16: If `True`, convert activations and gradients send between clients to `torch.float16`.
                Reduces bandwidth needed for communication but might impact model accuracy.
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.intersection_file = intersection_file
        self.lr = lr
        self.model = model
        self.analytic_sender_id = analytic_sender_id
        self.fp16 = fp16
        self.target_names = None

        self.app_root = None
        self.current_round = None
        self.num_rounds = None
        self.batch_size = None
        self.writer = None
        self.client_name = None
        self.other_client = None
        self.device = None
        self.optimizer = None
        self.criterion = None
        self.transform_train = None
        self.train_dataset = None
        self.split_id = None
        self.activations = None
        self.batch_indices = None
        self.train_size = 0

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
        """Get model from client config. Modelled after `PTFileModelPersistor`."""
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
            self.log_error(fl_ctx, f"expect model to be torch.nn.Module but got {type(self.model)}: {self.model}")
            return
        if self.model is None:
            self.log_error(fl_ctx, f"Model wasn't built correctly! It is {self.model}")
            return
        self.log_info(fl_ctx, f"Running model {self.model}")

    def initialize(self, parts: dict, fl_ctx: FLContext):
        self._get_model(fl_ctx=fl_ctx)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
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

        if self.intersection_file is not None:
            _intersect_indices = np.loadtxt(self.intersection_file)
        else:
            _intersect_indices = None
        self.train_dataset = CIFAR10SplitNN(
            root=self.dataset_root,
            train=True,
            download=True,
            transform=self.transform_train,
            returns=data_returns,
            intersect_idx=_intersect_indices,
        )
        self.train_size = len(self.train_dataset)
        if self.train_size <= 0:
            raise ValueError(f"Expected train dataset size to be larger zero but got {self.train_size}")
        self.log_info(fl_ctx, f"Training with {self.train_size} overlapping indices of {self.train_dataset.orig_size}.")

        # Select local TensorBoard writer or event-based writer for streaming
        if self.split_id == 1:  # metrics can only be computed for client with labels
            self.writer = parts.get(self.analytic_sender_id)  # user configured config_fed_client.json for streaming
            if not self.writer:  # use local TensorBoard writer only
                self.writer = SummaryWriter(self.app_root)

        # register aux message handlers
        engine = fl_ctx.get_engine()

        if self.split_id == 1:
            engine.register_aux_message_handler(
                topic=SplitNNConstants.TASK_LABEL_STEP, message_handle_func=self.train_label_side
            )
            self.log_debug(fl_ctx, f"Registered aux message handlers for split_id {self.split_id}")

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

        labels = self.train_dataset.get_batch(batch_indices)
        labels = labels.to(self.device)

        if self.fp16:
            activations = activations.type(torch.float32)  # return to default pytorch precision

        activations = activations.to(self.device)
        activations.requires_grad_(True)

        pred = self.model.forward(activations)
        loss = self.criterion(pred, labels)
        loss.backward()

        _, pred_labels = torch.max(pred, 1)
        acc = (pred_labels == labels).sum() / len(labels)

        self.log_info(
            fl_ctx,
            f"Round {self.current_round}/{self.num_rounds} train_loss: {loss.item():.4f}, accuracy: {acc.item():.4f}",
        )
        if self.writer:
            self.writer.add_scalar("train_loss", loss, self.current_round)
            self.writer.add_scalar("train_accuracy", acc, self.current_round)

        self.optimizer.step()
        if self.timeit:
            self.times["learner_end_label_step"].append(timer())

        if not isinstance(activations.grad, torch.Tensor):
            raise ValueError("No valid gradients available!")
        # gradient to be returned to other client
        if self.fp16:
            return activations.grad.type(torch.float16)
        else:
            return activations.grad

    def backward_step_data_side(self, gradient, fl_ctx: FLContext):
        if self.timeit:
            self.times["learner_start_backward_step"].append(timer())
        self.optimizer.zero_grad()

        if self.fp16:
            gradient = gradient.type(torch.float32)  # return to default pytorch precision

        gradient = gradient.to(self.device)
        self.activations.backward(gradient=gradient)
        self.optimizer.step()

        self.log_debug(
            fl_ctx, f"{self.client_name} runs model with `split_id` {self.split_id} for backward step on data side."
        )
        if self.timeit:
            self.times["learner_end_backward_step"].append(timer())

    def train_backward_data_side(self, fl_ctx: FLContext, gradient=None) -> Shareable:
        if self.timeit:
            self.times["aux_hdl_learner_start_data_train_back_step"].append(timer())
        # combine forward and backward on data client
        # 1. perform backward step if gradients provided
        if gradient is not None:
            result_backward = self.backward_data_side(gradient, fl_ctx=fl_ctx)
            assert (
                result_backward.get_return_code() == ReturnCode.OK
            ), f"Backward step failed with return code {result_backward.get_return_code()}"
        # 2. compute activations
        activations = self.train_data_side(fl_ctx=fl_ctx)
        if self.timeit:
            self.times["aux_hdl_learner_end_data_train_back_step"].append(timer())
        return activations

    def train_data_side(self, fl_ctx: FLContext) -> Shareable:
        if self.timeit:
            self.times["aux_hdl_learner_start_data_train_step"].append(timer())
        if self.split_id != 0:
            raise ValueError(
                f"Expected `split_id` 0. It doesn't make sense to run `train_data_side` with `split_id` {self.split_id}"
            )

        self.log_debug(fl_ctx, f"Train data side in round {self.current_round} of {self.num_rounds} rounds.")

        act = self.train_step_data_side(batch_indices=self.batch_indices)

        self.log_debug(
            fl_ctx, f"{self.client_name} finished model with `split_id` {self.split_id} for train on data side."
        )

        if self.timeit:
            self.times["aux_hdl_learner_end_data_train_step"].append(timer())
        self.log_debug(fl_ctx, f"Sending train data activations: {type(act)}")

        if self.fp16:
            return act.type(torch.float16)
        else:
            return act

    def train_label_side(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """aux message handler"""

        if self.timeit:
            self.times["aux_hdl_learner_start_label_train_step"].append(timer())
        if self.split_id != 1:
            raise ValueError(
                f"Expected `split_id` 1. It doesn't make sense to run `train_label_side` with `split_id` {self.split_id}"
            )

        self.current_round = request.get_header(AppConstants.CURRENT_ROUND)
        self.num_rounds = request.get_header(AppConstants.NUM_ROUNDS)
        self.log_debug(fl_ctx, f"Train label in round {self.current_round} of {self.num_rounds} rounds.")

        dxo = from_shareable(request)
        if dxo.data_kind != SplitNNDataKind.ACTIVATIONS:
            raise ValueError(f"Expected data kind {SplitNNDataKind.ACTIVATIONS} but received {dxo.data_kind}")

        batch_indices = dxo.get_meta_prop(SplitNNConstants.BATCH_INDICES)
        if batch_indices is None:
            raise ValueError("No batch indices in DXO!")

        activations = dxo.data.get(SplitNNConstants.DATA)
        if activations is None:
            raise ValueError("No activations in DXO!")

        gradient = self.train_step_label_side(
            batch_indices=batch_indices, activations=fobs.loads(activations), fl_ctx=fl_ctx
        )

        self.log_debug(fl_ctx, "train_label_side finished.")
        return_shareable = DXO(
            data={SplitNNConstants.DATA: fobs.dumps(gradient)}, data_kind=SplitNNDataKind.GRADIENT
        ).to_shareable()
        if self.timeit:
            self.times["aux_hdl_learner_end_label_train_step"].append(timer())

        self.log_debug(fl_ctx, f"Sending train label return_shareable: {type(return_shareable)}")
        return return_shareable

    def backward_data_side(self, gradient, fl_ctx: FLContext) -> Shareable:
        if self.timeit:
            self.times["aux_hdl_learner_start_data_backward_step"].append(timer())
        if self.split_id != 0:
            raise ValueError(
                f"Expected `split_id` 0. It doesn't make sense to run `backward_data_side` with `split_id` {self.split_id}"
            )

        self.backward_step_data_side(gradient=fobs.loads(gradient), fl_ctx=fl_ctx)

        self.log_debug(fl_ctx, "backward_data_side finished.")
        if self.timeit:
            self.times["aux_hdl_learner_end_data_backward_step"].append(timer())
        return make_reply(ReturnCode.OK)

    # Model initialization task (one time only in beginning)
    def init_model(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed.") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError("No global weights loaded!")

        self.log_info(fl_ctx, "init_model finished.")
        return make_reply(ReturnCode.OK)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """main training logic"""

        self.num_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        if not self.num_rounds:
            raise ValueError("No number of rounds available.")
        self.batch_size = shareable.get_header(SplitNNConstants.BATCH_SIZE)
        self.target_names = np.asarray(
            shareable.get_header(SplitNNConstants.TARGET_NAMES)
        )  # convert to array for string matching below
        self.other_client = self.target_names[self.target_names != self.client_name][0]
        self.log_info(fl_ctx, f"Starting training of {self.num_rounds} rounds with batch size {self.batch_size}")

        gradients = None  # initial gradients
        for _curr_round in range(self.num_rounds):
            self.current_round = _curr_round
            if self.split_id != 0:
                continue  # only run this logic on first site
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            engine = fl_ctx.get_engine()
            self.log_debug(fl_ctx, f"Starting current round={self.current_round} of {self.num_rounds}.")
            self.batch_indices = np.random.randint(0, self.train_size - 1, self.batch_size)

            # Site-1 image forward & backward (from 2nd round)
            fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.current_round, private=True, sticky=False)
            activations = self.train_backward_data_side(fl_ctx, gradients)

            # Site-2 label loss & backward
            dxo = DXO(data={SplitNNConstants.DATA: fobs.dumps(activations)}, data_kind=SplitNNDataKind.ACTIVATIONS)
            dxo.set_meta_prop(SplitNNConstants.BATCH_INDICES, self.batch_indices)

            data_shareable = dxo.to_shareable()
            data_shareable.set_header(AppConstants.CURRENT_ROUND, self.current_round)
            data_shareable.set_header(AppConstants.NUM_ROUNDS, self.num_rounds)
            data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self.current_round)

            # send to other side
            result = engine.send_aux_request(
                targets=self.other_client,
                topic=SplitNNConstants.TASK_LABEL_STEP,
                request=data_shareable,
                timeout=SplitNNConstants.TIMEOUT,
                fl_ctx=fl_ctx,
            )
            shareable = result.get(self.other_client)
            if shareable is not None:
                dxo = from_shareable(shareable)
                if dxo.data_kind != SplitNNDataKind.GRADIENT:
                    raise ValueError(f"Expected data kind {SplitNNDataKind.GRADIENT} but received {dxo.data_kind}")
                gradients = dxo.data.get(SplitNNConstants.DATA)
            else:
                raise ValueError(f"No message returned from {self.other_client}!")

            # TODO: remove debugging
            # _act = fobs.dumps(activations)
            # print(f"c1->c2 (activations {type(_act)}): {sys.getsizeof(_act)*1e-6:.2f} MB")
            # print(f"c2->c1 (gradients {type(gradients)}): {sys.getsizeof(gradients)*1e-6:.2f} MB")

            self.log_debug(fl_ctx, f"Ending current round={self.current_round}.")

        return make_reply(ReturnCode.OK)

    def finalize(self, fl_ctx: FLContext):
        if self.timeit:
            app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
            with open(os.path.join(app_root, "learner_times.pkl"), "wb") as f:
                pickle.dump(self.times, f)
