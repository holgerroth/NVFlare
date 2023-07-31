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


data_root = "/home/hroth/Code2/nvflare/bionemo_nvflare/examples/advanced/bionemo/task_fitting/fasta/mixed_soft"
service_result = "/home/hroth/Code2/BioNeMo/bionemo-service-public/examples/service/notebooks/fasta_esm1nv_service_embeddings.pkl"
service_result = None

def create_embedding_dict(embeddings, labels, sets):
    embeddings_dict = {
        "x": [],
        "y": [],
        "label": [],
        "set": []
    }
    for x, y, set in zip(embeddings, labels, sets):
        embeddings_dict["x"].append(x[0])
        embeddings_dict["y"].append(x[1])
        embeddings_dict["label"].append(y)
        embeddings_dict["set"].append(set)

    return embeddings_dict


def main():
    X_train = list()
    y_train = list()
    X_test = list()
    y_test = list()
    sets = list()

    if service_result:
        print(f"Reading data from {service_result}")
        _data = pickle.load(open(service_result, "rb"))
        X_train = _data["X_train"]
        y_train = _data["y_train"]
        X_test = _data["X_test"]
        y_test = _data["y_test"]
        sets = ["train"] * len(y_train) + ["test"] * len(y_test)
    else:
        print(f"Reading data from {data_root}")
        for site_name in ["site-1", "site-2", "site-3"]:
            # Read embeddings
            data_filename = os.path.join(data_root, f"data_{site_name}.pkl")
            protein_embeddings = pickle.load(open(data_filename, "rb"))
            print(f"Loaded {len(protein_embeddings)} embeddings")

            # Read labels
            labels_filename = os.path.join(data_root, f"data_{site_name}.csv")
            labels = pd.read_csv(labels_filename).astype(str)

            # Prepare the data for training
            for embedding in protein_embeddings:
                # get label entry from pandas dataframe
                label = labels.loc[labels["id"] == str(embedding["id"])]
                if label['SET'].item() == 'train':
                    X_train.append(embedding["embeddings"])
                    y_train.append(label['TARGET'].item())
                    sets.append("train")
                elif label['SET'].item() == 'test':
                    X_test.append(embedding["embeddings"])
                    y_test.append(label['TARGET'].item())
                    sets.append("test")

    assert len(X_train) > 0
    assert len(X_test) > 0
    print(f"There are {len(X_train)} training samples and {len(X_test)} testing samples.")

    # plot UMAP
    print("compute UMAP embedding...")
    reducer = umap.UMAP(transform_seed=42)
    umap_embedding = reducer.fit_transform(X_train + X_test)
    print(f"UMAP embedding train/test {umap_embedding.shape}")

    embeddings_dict = create_embedding_dict(umap_embedding, y_train + y_test, sets)

    label_names = sorted(set(y_test))
    plt.figure()
    ax = sns.scatterplot(data=pd.DataFrame(embeddings_dict), x="x", y="y", hue="label", hue_order=label_names, style="set")
    plt.title(f"{len(y_train)} train, {len(y_test)} test")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()

if __name__ == "__main__":
    main()
