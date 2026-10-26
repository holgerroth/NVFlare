# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
from pathlib import Path

from nvflare.fuel.utils.constants import FrameworkType
from nvflare.recipe import FedAvgRecipe, SimEnv


def make_recipe(rounds=3):
    return FedAvgRecipe(
        name="ga4gh-numpy",
        framework=FrameworkType.NUMPY,
        model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        min_clients=2,
        num_rounds=rounds,
        train_script=str(Path(__file__).with_name("client.py")),
        launch_external_process=False,
        key_metric="weight_mean",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--job-dir", help="Export for an existing POC or production federation")
    target.add_argument("--workspace", help="Run the recipe locally with two simulator clients")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    recipe = make_recipe(args.rounds)
    if args.job_dir:
        recipe.export(args.job_dir)
    else:
        run = recipe.execute(SimEnv(num_clients=2, num_threads=2, workspace_root=args.workspace))
        print("Results:", run.get_result())
