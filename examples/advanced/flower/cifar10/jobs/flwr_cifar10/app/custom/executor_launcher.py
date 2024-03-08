# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import time

from nvflare.apis.dxo import MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.security.logging import secure_format_exception
from nvflare.app_common.workflows.model_controller import ModelController
from nvflare.app_common.abstract.launcher import Launcher, LauncherRunStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.utils.validation_utils import check_object_type


class ExecutorLauncher(Executor):
    def __init__(
        self,
        launcher_id="launcher",
        task_name=AppConstants.TASK_TRAIN
    ):
        """Key component to run learner on clients.

        Args:
            learner_id (str): id of the learner object
            train_task (str, optional): task name for train. Defaults to AppConstants.TASK_TRAIN.
            submit_model_task (str, optional): task name for submit model. Defaults to AppConstants.TASK_SUBMIT_MODEL.
            validate_task (str, optional): task name for validation. Defaults to AppConstants.TASK_VALIDATION.
        """
        super().__init__()
        self._launcher_id = launcher_id
        self._task_name = task_name
        self.is_initialized = False

    def _init_launcher(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        launcher: Launcher = engine.get_component(self._launcher_id)
        if launcher is None:
            raise RuntimeError(f"Launcher can not be found using {self._launcher_id}")
        check_object_type(self._launcher_id, launcher, Launcher)
        self.launcher = launcher
        self.is_initialized = True

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            if not self.is_initialized:
                self._init_launcher(fl_ctx)

            self._launch_script(fl_ctx)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        pass

    def _launch_script(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Start Executor Launcher.")

        self.launcher.initialize(fl_ctx=fl_ctx)

        while True:
            time.sleep(10.0)
            run_status = self.launcher.check_run_status(task_name=self._task_name, fl_ctx=fl_ctx)
            if run_status == LauncherRunStatus.RUNNING:
                print(f"Running ... [{self.launcher._script}]")
            elif run_status == LauncherRunStatus.COMPLETE_SUCCESS:
                print("run success")
                break
            else:
                print(f"run failed or not start: {run_status}")
                break
        self.launcher.finalize(fl_ctx=fl_ctx)
        self.log_info(fl_ctx, "Stop Executor Launcher.")
