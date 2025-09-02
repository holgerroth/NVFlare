# Federated Deep Learning for Financial Fraud Detection
The following notebooks demonstrate how to implement federated learning for financial fraud detection using NVIDIA FLARE (NVFlare). The notebooks show a complete workflow starting from federated data statistics, to training a deep learning model across multiple clients while maintaining data privacy.

### Key Components:

1. *Federated Learning Setup:* Uses NVFlare's FedAvg (Federated Averaging) recipe to train a SimpleNetwork model for binary classification (fraud detection) across multiple clients.
Experiment Tracking: Integrates MLflow for comprehensive experiment tracking, including model artifacts, training metrics, and federated learning statistics.
Environment Flexibility: Supports both simulation and production environments, allowing developers to test locally before deploying to production.
Financial Application: Specifically designed for financial fraud detection, where data privacy is crucial and federated learning enables collaborative model training without sharing raw data.
The notebook demonstrates the complete pipeline from recipe definition to execution, making it a practical example of how to implement federated learning in production financial systems using NVFlare.

### Prerequisit

Install the dependencies and start a Jupyter Lab instance. We recommend doing this in a fresh virtual environment.

```
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Start Jupyter Lab
```
jupyter lab .
```

### Data

We assume a CSV dataset with numerical features is available to provide data on each client. In this example, we use the "[Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)" from Kaggle to illustrate NVFlare's functionalities. You can download the data zipfile, unzip it and mount it under

`/workspace/dataset/paysim1/PS_20174392719_1491204439457_log.csv`

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

After making the data available on your client, you can get started with one of the following notebooks:

1. [Federated Statistics](./compute_fed_stats.ipynb)
2. [Training a Deep Learning Model for Fraud Detection](./train_dl_model.ipynb)
