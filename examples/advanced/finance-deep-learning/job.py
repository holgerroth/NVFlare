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

from model import SimpleNetwork

from nvflare.app_opt.tf.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.sim_env import SimEnv
from nvflare.recipe.prod_env import ProdEnv

recipe = FedAvgRecipe(
    name="hello-tf-mlflow",
    min_clients=1,
    num_rounds=3,
    initial_model=SimpleNetwork(num_classes=2),
    train_script="client.py",
    mlflow_tracking_uri="https://rrayrz6j-nvflmlflowserver.xenon.lepton.run"
)

recipe.export("transfer")

env = SimEnv(num_clients=1, num_threads=1)
recipe.execute(env=env)
