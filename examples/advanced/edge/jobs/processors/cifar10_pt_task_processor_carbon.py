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
import logging
import os

import torch
from codecarbon import OfflineEmissionsTracker, EmissionsTracker

from nvflare.apis.dxo import DXO, DataKind, from_dict
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse

from .models.cifar10_model import Cifar10ConvNet
from .cifar10_pt_task_processor import Cifar10PTTaskProcessor

CODECARBON_API_TOKEN = os.getenv("CODECARBON_API_TOKEN")

log = logging.getLogger(__name__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Cifar10PTTaskProcessorCarbon(Cifar10PTTaskProcessor):

    def __init__(
        self,
        data_root: str,
        subset_size: int,
        communication_delay: dict,
        device_speed: dict,
        local_batch_size: int = 4,
        local_epochs: int = 4,
        local_lr: float = 0.001,
        local_momentum: float = 0.9,
    ):
        self.tracker = None

        super().__init__(data_root, subset_size, communication_delay, device_speed, local_batch_size, local_epochs, local_lr, local_momentum)

    def setup(self, job: JobResponse) -> None:
        #client_name = "test-client" # TODO: get real client name 
        #country_iso_code = "USA"
        #self.tracker = OfflineEmissionsTracker(country_iso_code=self.country_iso_code, measure_power_secs=1, experiment_id=client_name)  
        #project_name = f"{flare.get_job_id}--{client_name}"  # TODO: get unique id from job
        
        project_name = f"flare_edge_{job.get_device_id()}"  # TODO: why is this id
        print(f"Project name: {project_name}")
        self.tracker = EmissionsTracker(project_name=project_name, experiment_id="", measure_power_secs=1, api_key=CODECARBON_API_TOKEN, tracking_mode="process")
        
    def process_task(self, task: TaskResponse) -> dict:
        # TODO: get current round
        # tracker.start_task(f"round_{input_model.current_round}")
        self.tracker.start_task("train")
        result = super().process_task(task)
        train_emissions_data = self.tracker.stop_task()
        print(f"train_emissions_data: {train_emissions_data}")        

        result["meta"]["emissions"] = train_emissions_data
         
        return result
