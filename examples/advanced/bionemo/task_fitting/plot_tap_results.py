# Copyright (c) 2021, NVIDIA CORPORATION.
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

import glob
import json
import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

# uniform
# results = \
#    {
#        "Local": "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/results/local_finetune_esm1nv_enclr1e-6_maxepochs200_JoinedChains_uniform",
#        "FL": "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/results/fedavg_finetune_esm1nv_enclr1e-6_maxepochs200_JoinedChains_uniform"
#    }
# out_dir = "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/plots/tap/uniform"
# out_name = "esm1_tap_uniform"

# alpha 1.0
# results = \
#     {
#         "Local": "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/results/local_finetune_esm1nv_enclr1e-6_maxepochs200_JoinedChains_alpha1.0",
#         "FL": "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/results/fedavg_finetune_esm1nv_enclr1e-6_maxepochs200_JoinedChains_alpha1.0"
#     }
# out_dir = "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/plots/tap/alpha1.0"
# out_name = "esm1_tap_alpha1.0"

# central
results = \
     {
         #"Local": "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/results/local_finetune_esm1nv_enclr1e-6_maxepochs200_JoinedChains_uniform",
         #"FL": "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/results/fedavg_finetune_esm1nv_enclr1e-6_maxepochs200_JoinedChains_uniform",
         "Central": "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/results/central_finetune_esm1nv_enclr1e-6_maxepochs200_JoinedChains"
     }
out_dir = "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/plots/tap/central"
out_name = "esm1_tap_central"


def read_eventfile(filepath, tags=["val_acc_global_model"]):
    data = {}
    for summary in tf.compat.v1.train.summary_iterator(filepath):
        for v in summary.summary.value:
            if v.tag in tags:
                # print(v.tag, summary.step, v.simple_value)
                if v.tag in data.keys():
                    data[v.tag].append([summary.step, v.simple_value])
                else:
                    data[v.tag] = [[summary.step, v.simple_value]]
    return data


def add_eventdata(data, filepath, tag="val_acc_global_model", setting="", endpoint=""):
    event_data = read_eventfile(filepath, tags=[tag])

    if tag in event_data:
        assert len(event_data[tag]) > 0, f"No data for key {tag}"
        values = []
        steps = []
        for i, e in enumerate(event_data[tag]):
            data["Setting"].append(setting)
            data["Step"].append(e[0])
            data["MSE"].append(e[1])
            rmse = np.sqrt(e[1])
            data["RMSE"].append(rmse)
            data["Endpoint"].append(endpoint)
            steps.append(e[0])
            values.append(rmse)
        print(f"added {len(event_data[tag])} entries for {tag}")


def save_figs(out_dir, out_name):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, f"{out_name}.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, f"{out_name}.svg"), dpi=300)
    plt.savefig(os.path.join(out_dir, f"{out_name}.pdf"), dpi=300)


def add_min_data(min_data, data_dict, endpoint, setting):
    data_df = pd.DataFrame(data_dict)
    sub_df = data_df[data_df["Setting"] == setting]
    sub_df = sub_df[sub_df["Endpoint"] == endpoint]
    values = np.asarray(sub_df["RMSE"])
    steps = np.asarray(sub_df["Step"])
    assert len(values) == len(steps)

    min_idx = np.argmin(values)
    assert isinstance(min_idx, np.int64)  # make sure only one minimum value is returned
    min_rmse = values[min_idx]
    min_step = steps[min_idx]
    print(f"{endpoint} {setting}: min: {min_rmse} at step: {min_step}")

    min_data["Setting"].append(setting)
    min_data["Endpoint"].append(endpoint)
    min_data["Min. Step"].append(min_step)
    min_data["Min. RMSE"].append(min_rmse)


def main():
    endpoints = {
        "site-1": "PSH",
        "site-2": "PPC",
        "site-3": "PNC",
        "site-4": "SFvCSP",
    }
    min_data = {"Setting": [], "Min. Step": [], "Min. RMSE": [], "Endpoint": []}

    for site_name, endpoint in endpoints.items():
        tag = f"val_{endpoint}_MSE"
        data = {"Step": [], "RMSE": [], "MSE": [], "Setting": [], "Endpoint": []}

        for setting, results_root in results.items():
            eventfiles = glob.glob(os.path.join(results_root, "**", f"app_{site_name}", "**", "events.out.tfevents.*"), recursive=True)
            assert len(eventfiles) > 0

            # add event files
            for eventfile in eventfiles:
                add_eventdata(data, eventfile, tag=tag, setting=setting, endpoint=endpoint)

            add_min_data(min_data, data, endpoint, setting)

        data = pd.DataFrame(data)
        print("Training TB data:")
        print("=" * 20)
        print(data)

        # FL vs Local
        plt.figure()

        line_plot = sns.lineplot(x="Step", y="RMSE", hue="Setting", data=data)
        #line_plot.set(yscale='log')
        plt.legend(loc='lower right')
        plt.title(endpoint)
        #plt.xlim(xlim)
        #plt.ylim(ylim)
        #plt.grid()
        #plt.show()

        save_figs(out_dir, f"{out_name}_{endpoint}")

    print("minimum values:")
    print("=" * 20)
    min_data = pd.DataFrame(min_data)
    print(min_data)
    min_data.to_csv(os.path.join(out_dir, f"{out_name}_min_data.csv"), index=False)


if __name__ == "__main__":
    main()

