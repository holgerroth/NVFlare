# Federated Protein Embeddings and Task Model Fitting with BioNeMo

This example notebook shows how to obtain protein learned representations in the form of embeddings using the ESM-1nv pre-trained model. The model is trained with NVIDIA's BioNeMo framework for Large Language Model training and inference. For more details, please visit NVIDIA BioNeMo Service at https://www.nvidia.com/en-us/gpu-cloud/bionemo/


## 1. Install requirements

Install required packages for training
```
pip install --upgrade pip
pip install -r ./requirements.txt
```

> **_NOTE:_**  We recommend either using a containerized deployment or virtual environment, 
> please refer to [getting started](https://nvflare.readthedocs.io/en/latest/getting_started.html).

Set `PYTHONPATH` to include custom files of this example:
```
export PYTHONPATH=${PYTHONPATH}:${PWD}
```

## 2. Start Jupyter Lab
We use [JupyterLab](https://jupyterlab.readthedocs.io) for this example.
To start JupyterLab, run
```
jupyter lab .
```
and open [task_fitting.ipynb](./task_fitting.ipynb).





### NOTES
n_clients = 3
# limiting to the proteins with sequence length<512 for embedding queries
MAX_SEQUENCE_LEN = 510  
out_dir = "/tmp/fasta/mixed_soft"

import os
import re
import numpy as np
import pandas as pd
        
# extract meta data and split
train_data = []
test_data = []

for x in proteins:
        if len(str(x.seq)) > MAX_SEQUENCE_LEN:
            continue
            
        data = {key: value for key, value in re.findall(r"([A-Z_]+)=(-?[A-z0-9]+[.0-9]*)", x.description)}
        data["sequence"] = str(x.seq)
    
        if data["SET"] == "train":
            train_data.append(data)
        elif data["SET"] == "test":
            test_data.append(data)
        
print(f"Read {len(train_data)} training and {len(test_data)} testing sequences. Total {len(train_data) + len(test_data)} valid sequences.")
        
# split training data
train_data_splits = np.array_split(train_data, n_clients)

# save split data
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# save test data (each client uses the same test data in this example)
pd.DataFrame(test_data).to_csv(os.path.join(out_dir, "test.csv"))

# save split training data
for idx, split in enumerate(train_data_splits):
    pd.DataFrame(split).to_csv(os.path.join(out_dir, f"train_site-{idx+1}.csv"))
    print(f"Saving {len(split)} training sequences for client site-{idx+1}.")
