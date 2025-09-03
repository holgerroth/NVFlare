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
from utils import ShapCollectionFilter

from nvflare.app_opt.tf.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.sim_env import SimEnv
from nvflare.recipe.prod_env import ProdEnv
from nvflare.recipe.utils import add_experiment_tracking

recipe = FedAvgRecipe(
    name="hello-tf-mlflow",
    min_clients=1,
    num_rounds=3,
    initial_model=SimpleNetwork(num_classes=2),
    train_script="src/dl/client.py",
    train_args="--dataset None" # /home/hroth/Code2/nvflare/jpm_demo/examples/advanced/finance-deep-learning/paysim1/PS_20174392719_1491204439457_log.csv
)

# Add experiment tracking
mlflow_config = {
    "tracking_uri": "https://rrayrz6j-nvflmlflowserver.xenon.lepton.run",
    "kw_args": {
        "experiment_name": "nvflare-poc-lepton-fl-experiment",
        "run_name": "nvflare-fedavgrecipe-with-mlflow-sim",
        "experiment_tags": {"mlflow.note.content": "## **NVFlare FedAvg experiment with MLflow**"},
        "run_tags": {"mlflow.note.content": "## Federated Experiment tracking with MLflow.\n"},
    },
    "artifact_location": "artifacts",
    "events": ["fed.analytix_log_stats"],
}
add_experiment_tracking(recipe, tracking_type="mlflow", tracking_config=mlflow_config)

# Add filter to collect SHAP metrics on the server
shap_filter = ShapCollectionFilter()
recipe.add_server_input_filter(shap_filter, tasks=["train"])

# Optionally export
recipe.export("job_configs")

# Run simulation
env = SimEnv(num_clients=1)
recipe.execute(env=env)

# Submit job in production environment
#env = ProdEnv(startup_kit_dir="...")
#recipe.execute(env=env)
