# Federated Deep Learning for Financial Fraud Detection
This project demonstrates how to implement federated learning for financial fraud detection using **NVIDIA FLARE (NVFlare)**. The notebooks provide an end-to-end workflow—from analyzing distributed financial datasets (using federated statistics) to training a deep learning model across multiple clients while preserving data privacy.

### 🔑 Key Components:
- **Data generation**
    Generate synthetic data on each client using a NVFlare job.

- **Federated Data Statistics**
    Compute distributed statistics across client datasets without exposing raw data.
        - Supported measures: count, mean, sum, standard deviation, histogram, quantiles
        - Interactive visualization for exploratory data analysis
- **Federated Learning Setup**
    - Uses NVFlare’s FedAvg (Federated Averaging) recipe to train a SimpleNetwork model for binary fraud classification.
    - Configurable to run in both simulation (local prototyping) and production (multi-client deployment) environments.
- **Experiment Tracking**
    -Integrated with MLflow for tracking of training metrics
- **Financial Fraud Detection Use Case**
    - Tailored for fraud detection scenarios where data privacy is critical.
    - Enables collaborative model training across financial institutions without sharing sensitive raw data.

### 📌 Workflow Overview
1. We generate synthetic data on each client.
2. Define and compute federated dataset statistics.
3. Visualize federated statistics and training performance.
4. Configure the federated learning recipe with NVFlare.
5. Train the deep learning model across multiple clients using FedAvg.
6. Track and analyze results with MLflow.

This notebook series demonstrates the full pipeline from recipe definition to execution, providing a practical example of how federated learning can be applied in real-world financial systems using NVFlare.

## Prerequisit

Install the dependencies and start a Jupyter Lab instance. We recommend doing this in a fresh virtual environment.

```
python -m venv .venv
source .venv/bin/activate

pip install jupyterlab ipywidgets
pip install -r src/stats/requirements.txt
```

## PyTorch Users
```
pip install -r src/pt/requirements.txt
```

## TensorFlow Users
```
pip install -r src/tf/requirements.txt
```

Start Jupyter Lab
```
jupyter lab .
```

## Data

We assume a CSV dataset with numerical features is available to provide data on each client. In this example, we run first a job to generate synthetic data

[01_generate_data.ipynb](./01_generate_data.ipynb)

This will generate a `data.csv` file under each client's `/worspace/dataset/{outname}` folder.

We use the feature names from "[Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)" from Kaggle to illustrate NVFlare's functionalities. You can download that public data zipfile as well, unzip it and mount it under

`/workspace/dataset/paysim1/PS_20174392719_1491204439457_log.csv`

for further experimentation.

We use the following five numerical features for training a multi-layer perceptron (MLP) deep neural network to classify each transaction as fraud or not.

| Column | Description |
|-------|-------------|
| amount | Amount of the transaction in local currency |
| oldbalanceOrg | Initial balance before the transaction |
| newbalanceOrig | New balance after the transaction |
| oldbalanceDest | Initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants) |
| newbalanceDest | New balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants) |

The **isFraud** column describes wether the transactions made by fraudulent agents. 

## End-to-end Example Notebooks

To get started run these notebooks.

1. [Data Generation](./01_generate_data.ipynb)
2. [Federated Statistics](./02_compute_fed_stats.ipynb)
3. 
    a) PyTorch version: [Training a Deep Learning Model for Fraud Detection](./03a_train_pytorch_model.ipynb)
    
    b) Tensorflow version: [Training a Deep Learning Model for Fraud Detection](./03b_train_tensorflow_model.ipynb)

> Note that running the pytorch or tensorflow examples in real production requires the corresponding dependencies to be installed on the server and client machines.

