# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare import FedAvg, FedJob
from nvflare.app_opt.flower.controller import FlowerController
from nvflare.app_opt.flower.executor import FlowerExecutor


if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2

    job = FedJob(name="cifar10_flwr")

    # Define the controller workflow and send to server
    controller = FlowerController(
        server_app="server:app"
    )
    job.to(controller, "server")

    # Add flwr server code
    job.to("server.py", "server")
    job.to("task.py", "server")

    # Add clients
    for i in range(n_clients):
        executor = FlowerExecutor(
            client_app="client:app"
        )
        job.to(executor, f"site-{i}", gpu=0)

        # Add flwr client code
        job.to("client.py", f"site-{i}")
        job.to("task.py", f"site-{i}")

    job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir")
