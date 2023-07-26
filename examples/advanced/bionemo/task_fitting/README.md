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
