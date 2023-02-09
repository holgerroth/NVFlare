# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from pt.learners.cifar10_learner_splitnn import SplitNNConstants

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner


# TODO: maybe move all into one executor
class SplitNNLearnerExecutor(Executor):
    def __init__(
        self, learner_id, init_model_task=SplitNNConstants.TASK_INIT_MODEL, train_task=SplitNNConstants.TASK_TRAIN
    ):
        """Key component to run learner on clients.

        Args:
            learner_id (str): id pointing to the learner object
            train_task (str, optional): label to dispatch train task. Defaults to AppConstants.TASK_TRAIN.
            submit_model_task (str, optional): label to dispatch submit model task. Defaults to AppConstants.TASK_SUBMIT_MODEL.
            validate_task (str, optional): label to dispatch validation task. Defaults to AppConstants.TASK_VALIDATION.
        """
        super().__init__()
        self.learner_id = learner_id
        self.learner = None
        self.init_model_task = init_model_task
        self.train_task = train_task

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.ABORT_TASK:
            try:
                if self.learner:
                    self.learner.abort(fl_ctx)
            except Exception as e:
                self.log_exception(fl_ctx, f"learner abort exception: {e}")
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        try:
            engine = fl_ctx.get_engine()
            self.learner = engine.get_component(self.learner_id)
            if not isinstance(self.learner, Learner):  # TODO: should we have a SplitNNLearner base class?
                raise TypeError(f"learner must be Learner type. Got: {type(self.learner)}")
            self.learner.initialize(engine.get_all_components(), fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"learner initialize exception: {e}")

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")

        self.log_info(fl_ctx, f"Executing task {task_name}...")
        try:
            if task_name == self.init_model_task:
                self.log_info(fl_ctx, "Initializing model...")
                return self.learner.init_model(shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal)
            elif task_name == self.train_task:
                self.log_info(fl_ctx, "Running training...")  # TODO: remove this? It's actually never executed.
            else:
                self.log_error(fl_ctx, f"Could not handle task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            # Task execution error, return EXECUTION_EXCEPTION Shareable
            self.log_exception(fl_ctx, f"learner execute exception: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def finalize(self, fl_ctx: FLContext):
        try:
            if self.learner:
                self.learner.finalize(fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"learner finalize exception: {e}")
