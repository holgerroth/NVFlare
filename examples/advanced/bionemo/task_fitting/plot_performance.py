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


import os
import time
import numpy as np
import pickle
import pandas as pd
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import json


result_root = "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/bionemo_results_mlp_tuning"

def read_json(filename):
    with open(filename) as f:
        return json.load(f)

def main():
    result_files = glob.glob(os.path.join(result_root, "**", "cross_val_results.json"), recursive=True)
    assert len(result_files) > 0

    results = {
        "Accuracy": [],
        "Site": [],
        "Hiddens": [],
        "Setting": []
    }
    for result_file in result_files:
        _results = read_json(result_file)

        hiddens = result_file.replace(result_root, "").split(os.sep)[1].replace("bionemo_", "")
        assert len(hiddens) > 0

        for site, v in _results.items():
            results["Site"].append(site)
            results["Hiddens"].append(hiddens)
            if "local" in result_file:
                results["Setting"].append("Local")
                results["Accuracy"].append(v[site]["accuracy"])
            elif "fedavg" in result_file:
                results["Setting"].append("FL")
                results["Accuracy"].append(v["SRV_FL_global_model.pt"]["accuracy"])
                # best
                results["Site"].append(site)
                results["Hiddens"].append(hiddens)
                results["Setting"].append("FL (best)")
                results["Accuracy"].append(v["SRV_best_FL_global_model.pt"]["accuracy"])
            else:
                raise ValueError

    hidden_names = ['32', '64_32', '128_64', '128_64_32', '256_128_64', '512_256_128', '1024_512_256', '2048_1024_512', '512_256_128_64', '1024_512_256_128']
    hidden_names = ['32', '64_32', '128_64', '128_64_32', '256_128_64', '512_256_128', '512_256_128_64']
    unique_hiddens = list(set(results["Hiddens"]))
    print(f"Read {len(unique_hiddens)} hidden sizes: {unique_hiddens}")
    #assert len(hidden_names) == len(unique_hiddens)

    plt.figure()
    ax = sns.barplot(data=pd.DataFrame(results), x="Hiddens", y="Accuracy", hue="Setting",
                     order=hidden_names, hue_order=["Local", "FL"],# "FL (best)"],
                     estimator='mean', errorbar="sd")
    plt.ylim(0.665, 0.785)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()

if __name__ == "__main__":
    main()
