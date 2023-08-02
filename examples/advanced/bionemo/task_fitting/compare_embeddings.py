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


def main():
    X_service = []
    seq_service = []

    X_local = []
    seq_local = []

    # Read service results
    print(f"Reading data from {service_result}")
    _data = pickle.load(open(service_result, "rb"))
    X_service.extend(_data["X_train"])
    X_service.extend(_data["X_test"])
    seq_service.extend(_data["seq_train"])
    seq_service.extend(_data["seq_test"])

    # Read local results
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
            X_local.append(embedding["embeddings"])
            seq_local.append(embedding["sequence"])

    assert len(X_service) > 0
    assert len(X_local) > 0
    assert len(seq_service) > 0
    assert len(seq_local) > 0
    print(f"There are {len(X_service)} service samples and {len(X_local)} local samples.")

    # Compare
    norms = {
        "L1-norm": [],
        "Type": [],
    }
    delta_l1_norms = []
    service_l1_norms = []
    local_l1_norms = []

    for s_idx, s_seq in enumerate(seq_service):
        l_idx = np.where(np.asarray(seq_local) == s_seq)
        if not np.any(l_idx):
            continue
        assert len(l_idx) == 1
        l_idx = l_idx[0].item()

        if s_idx % 1000 == 0:
            print(f"Processing {s_idx+1} of {len(seq_service)} samples")

        delta_l1 = np.linalg.norm(X_service[s_idx]-X_local[l_idx], ord=1)
        service_l1 = np.linalg.norm(X_service[s_idx], ord=1)
        local_l1 = np.linalg.norm(X_local[s_idx], ord=1)

        delta_l1_norms.append(delta_l1)
        service_l1_norms.append(service_l1)
        local_l1_norms.append(local_l1)

        for norm, type in zip([delta_l1, service_l1, local_l1], ["Delta", "Service", "Local"]):
            norms["L1-norm"].append(float(norm))
            norms["Type"].append(type)

    #sns.histplot(data=pd.DataFrame(norms), x="Type", y="L1-norm")

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.hist(service_l1_norms)
    plt.title("Service L1-norm")

    plt.subplot(1, 3, 2)
    plt.hist(local_l1_norms)
    plt.title("Local L1-norm")

    plt.subplot(1, 3, 3)
    plt.hist(delta_l1_norms)
    plt.title("Delta")

    plt.show()


if __name__ == "__main__":
    main()
