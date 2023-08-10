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


split_root = "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/fasta"

alphas = [1, 5, 10]



def main():
    for alpha in alphas:
        labels = {
            "Site": [],
            "Target": [],
            "Count": []
        }

        split_summary_file = os.path.join(split_root, f"alpha{alpha}_summary.txt")
        assert os.path.isfile(split_summary_file)

        with open(split_summary_file, "r") as f:
            lines = f.readlines()
        label_str = lines[3]
        label_dict = json.loads(label_str)

        for site, counts in label_dict.items():
            for target, count in counts.items():
                labels["Site"].append(site)
                labels["Target"].append(target)
                labels["Count"].append(count)

        class_labels = ["Cell_membrane", "Cytoplasm", "Endoplasmic_reticulum", "Extracellular", "Golgi_apparatus",
                        "Lysosome", "Mitochondrion", "Nucleus", "Peroxisome", "Plastid"]

        plt.figure()
        ax = sns.barplot(data=pd.DataFrame(labels), x="Site", y="Count", hue="Target", hue_order=class_labels)
        plt.title(f"alpha = {alpha}")
        #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.legend([], [], frameon=False)
        plt.show()

if __name__ == "__main__":
    main()
