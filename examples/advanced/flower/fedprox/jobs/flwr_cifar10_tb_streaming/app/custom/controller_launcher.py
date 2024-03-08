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

import time

from nvflare.app_common.workflows.model_controller import ModelController
from nvflare.app_common.abstract.launcher import Launcher, LauncherRunStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.fuel.utils.validation_utils import check_object_type


class ControllerLauncher(ModelController):
    """The base controller for FedAvg Workflow. *Note*: This class is based on the experimental `ModelController`.

    Implements [FederatedAveraging](https://arxiv.org/abs/1602.05629).
    The model persistor (persistor_id) is used to load the initial global model which is sent to a list of clients.
    Each client sends it's updated weights after local training which is aggregated.
    Next, the global model is updated.
    The model_persistor also saves the model after training.

    Provides the default implementations for the follow routines:
        - def sample_clients(self, min_clients)
        - def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel
        - def update_model(self, aggr_result)

    The `run` routine needs to be implemented by the derived class:

        - def run(self)
    """

    def __init__(self,
                 launcher_id,
                 task_name=AppConstants.TASK_TRAIN
    ):
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

    def run(self):
        self.info("Start Controller Launcher.")

        if not self.is_initialized:
            self._init_launcher(self.fl_ctx)

        self.launcher.initialize(fl_ctx=self.fl_ctx)

        while True:
            time.sleep(10.0)
            run_status = self.launcher.check_run_status(task_name=self._task_name, fl_ctx=self.fl_ctx)
            if run_status == LauncherRunStatus.RUNNING:
                print(f"Running ... [{self.launcher._script}]")
            elif run_status == LauncherRunStatus.COMPLETE_SUCCESS:
                print("run success")
                break
            else:
                print(f"run failed or not start: {run_status}")
                break
        self.launcher.finalize(fl_ctx=self.fl_ctx)
        self.info("Stop Controller Launcher.")

