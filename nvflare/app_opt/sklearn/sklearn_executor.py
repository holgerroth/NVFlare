# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os.path

import joblib
import tensorboard

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_exception


def _get_global_params(shareable: Shareable, fl_ctx: FLContext):
    # retrieve current global params download from server's shareable
    dxo = from_shareable(shareable)
    current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
    fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round)
    return current_round, dxo.data


class SKLearnExecutor(Executor):
    def __init__(
        self,
        learner_id: str,
        train_task=AppConstants.TASK_TRAIN,
        submit_model_task=AppConstants.TASK_SUBMIT_MODEL,
        validate_task=AppConstants.TASK_VALIDATION,
    ):
        """An Executor interface for scikit-learn Learner.

        Args:
            learner_id (str): id pointing to the learner object
            train_task (str, optional): label to dispatch train task. Defaults to AppConstants.TASK_TRAIN.
            submit_model_task (str, optional): label to dispatch submit_model task. Defaults to AppConstants.TASK_SUBMIT_MODEL.
            validate_task (str, optional): label to dispatch validate task. Defaults to AppConstants.TASK_VALIDATION.
        """
        super().__init__()
        self.learner_id = learner_id
        self.learner = None
        self.train_task = train_task
        self.submit_model_task = submit_model_task
        self.validate_task = validate_task
        self.local_model_path = None
        self.global_model_path = None
        self.client_id = None
        self.writer = None
        self.fl_ctx = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.ABORT_TASK:
            try:
                if self.learner:
                    self.learner.abort(fl_ctx)
            except Exception as e:
                self.log_exception(fl_ctx, f"learner abort exception: {secure_format_exception(e)}")
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        self._print_configs(fl_ctx)
        self.load_log_tracker()

        try:
            engine = fl_ctx.get_engine()
            self.learner = engine.get_component(self.learner_id)
            if not isinstance(self.learner, Learner):
                raise TypeError(f"learner must be Learner type. Got: {type(self.learner)}")
            self.learner.initialize(engine.get_all_components(), fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"learner initialize exception: {secure_format_exception(e)}")

        # set the paths according to fl_ctx
        app_dir = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.local_model_path = os.path.join(app_dir, "model_local.joblib")
        self.global_model_path = os.path.join(app_dir, "model_global.joblib")

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")
        if abort_signal.triggered:
            self.finalize(fl_ctx)
            return make_reply(ReturnCode.TASK_ABORTED)

        try:
            if task_name == self.train_task:
                (current_round, global_params) = _get_global_params(shareable, fl_ctx)
                if current_round > 0:
                    # first round for parameter initialization
                    # no model evaluation
                    self.validate_model(current_round, global_params, fl_ctx)
                return self.train(current_round, global_params, fl_ctx)
            elif task_name == self.submit_model_task:
                return self.submit_model(shareable, fl_ctx)
            elif task_name == self.validate_task:
                return self.validate(shareable, fl_ctx, abort_signal)
            else:
                self.log_error(fl_ctx, f"Could not handle task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            # Task execution error, return EXECUTION_EXCEPTION Shareable
            self.log_exception(fl_ctx, f"learner execute exception: {secure_format_exception(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def train(self, current_round, global_param, fl_ctx: FLContext) -> Shareable:
        self.log_info(fl_ctx, f"Client {self.client_id} perform local train")
        # sklearn algorithms usually needs two different processing schemes
        # one for first round (generate initial centers for clustering, regular training for svm)
        # the other for following rounds (regular training for clustering, no further training for svm)
        # hence the current round is fed to learner to distinguish the two
        params, model = self.learner.train(current_round, global_param, fl_ctx)
        # save model and return dxo containing the params
        self.save_model_local(model)
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=params)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, self.learner.n_samples)
        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")

        return dxo.to_shareable()

    def submit_model(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Handle submit_model task for cross-site validation.
        
        Args:
            shareable: Shareable containing model submission request
            fl_ctx: FLContext
            
        Returns:
            Shareable containing the local model
        """
        try:
            model_name = shareable.get_header(AppConstants.SUBMIT_MODEL_NAME, "best_model")
            self.log_info(fl_ctx, f"Submitting local model: {model_name}")
            
            # Load the local model
            if os.path.exists(self.local_model_path):
                model = joblib.load(self.local_model_path)
                # Extract model parameters
                if hasattr(model, 'coef_'):
                    params = {"coef": model.coef_}
                    if hasattr(model, 'intercept_'):
                        params["intercept"] = model.intercept_
                    dxo = DXO(data_kind=DataKind.WEIGHTS, data=params)
                    return dxo.to_shareable()
                else:
                    self.log_error(fl_ctx, "Local model does not have required parameters")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            else:
                self.log_error(fl_ctx, f"Local model not found at {self.local_model_path}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        except Exception as e:
            self.log_exception(fl_ctx, f"Error submitting model: {secure_format_exception(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Handle validate task for cross-site validation.
        
        Args:
            shareable: Shareable containing model to validate
            fl_ctx: FLContext
            abort_signal: Signal to abort the task
            
        Returns:
            Shareable containing validation metrics
        """
        try:
            (current_round, global_params) = _get_global_params(shareable, fl_ctx)
            self.log_info(fl_ctx, f"Validating model from round {current_round}")
            
            metrics, model = self.learner.validate(current_round, global_params, fl_ctx)
            self.save_model_global(model)
            
            # Log metrics
            for key, value in metrics.items():
                self.log_value(key, value, current_round)
            
            # Return metrics as DXO
            dxo = DXO(data_kind=DataKind.METRICS, data=metrics)
            return dxo.to_shareable()
        except Exception as e:
            self.log_exception(fl_ctx, f"Error during validation: {secure_format_exception(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def validate_model(self, current_round, global_param, fl_ctx: FLContext):
        """Internal validation method called during training.
        
        Args:
            current_round: current training round
            global_param: global model parameters
            fl_ctx: FLContext
        """
        # retrieve current global center download from server's shareable
        self.log_info(fl_ctx, f"Client {self.client_id} perform local evaluation")
        metrics, model = self.learner.validate(current_round, global_param, fl_ctx)
        self.save_model_global(model)
        for key, value in metrics.items():
            self.log_value(key, value, current_round)

    def finalize(self, fl_ctx: FLContext):
        try:
            if self.learner:
                self.learner.finalize(fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"learner finalize exception: {secure_format_exception(e)}")

    def _print_configs(self, fl_ctx: FLContext):
        # get and print the args
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized with configs: \n {fl_args}",
        )

    def load_log_tracker(self):
        app_dir = self.fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.writer = tensorboard.summary.Writer(app_dir)

    def log_value(self, key, value, step):
        if self.writer:
            self.writer.add_scalar(key, value, step)
            self.writer.flush()

    def save_model_local(self, model: any) -> None:
        joblib.dump(model, self.local_model_path)

    def save_model_global(self, model: any) -> None:
        joblib.dump(model, self.global_model_path)
