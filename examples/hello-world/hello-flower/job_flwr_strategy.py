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

from flwr_controller import FlwrController
from cifar10_pt_fl import Net

from flwr.server.strategy import FedAvg as FlwrFedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner

def main():
    n_clients = 2
    num_rounds = 30
    train_script = "cifar10_pt_fl.py"

    # Create BaseFedJob with initial model
    job = BaseFedJob(
      name="cifar10_pt_fedavg",
      initial_model=Net(),
    )

    # Define the controller and send to server
    controller = FlwrController(
        num_clients=n_clients,
        num_rounds=num_rounds,
        strategy=FlwrFedAvg(),
    )
    job.to_server(controller)

    # Add clients
    for i in range(n_clients):
        runner = ScriptRunner(script=train_script)
        job.to(runner, f"site-{i}")

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")



if __name__ == "__main__":
    main()
