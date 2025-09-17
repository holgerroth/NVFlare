# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleNetwork
from utils import MLflowCallback, compute_shapley_values, load_csv_data

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.client.tracking import MLflowWriter

PATH = "pt_model.weights.pth"


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NVFlare Deep Learning Client for Financial Fraud Detection")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the CSV dataset dir or file (default: None, meaning randomly generated data)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    # (2) initializes NVFlare client API
    flare.init()

    # Load CSV data using the utility function
    print(f"Loading data from {args.data_path}")
    feature_columns = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    (train_features, train_labels), (test_features, test_labels) = load_csv_data(
        data_path=args.data_path, feature_columns=feature_columns, label_column="isFraud"
    )

    # Get the number of features for model input shape
    n_features = train_features.shape[1]
    n_classes = len(np.unique(train_labels))

    print("Loaded data:")
    print("train_features: ", train_features.shape)
    print("train_labels: ", train_labels.shape)
    print("test_features: ", test_features.shape)
    print("test_labels: ", test_labels.shape)

    # Convert numpy arrays to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_features_tensor = torch.FloatTensor(train_features).to(device)
    train_labels_tensor = torch.LongTensor(train_labels).to(device)
    test_features_tensor = torch.FloatTensor(test_features).to(device)
    test_labels_tensor = torch.LongTensor(test_labels).to(device)

    # Create model, optimizer, and loss function
    model = SimpleNetwork(input_size=n_features, num_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model created with {n_features} input features and {n_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    mlflow = MLflowWriter()

    # Create the callback to log training metrics to MLflow
    mlflow_callback = MLflowCallback(mlflow)

    # (3) gets FLModel from NVFlare
    while flare.is_running():
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # (optional) print system info
        system_info = flare.system_info()
        print(f"NVFlare system info: {system_info}")

        # (4) loads model from NVFlare
        for name, param in model.named_parameters():
            if name in input_model.params:
                param.data = torch.tensor(input_model.params[name], device=device)

        # (5) evaluate aggregated/received model
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_features_tensor)
            test_pred = torch.argmax(test_outputs, dim=1)
            test_global_acc = (test_pred == test_labels_tensor).float().mean().item()
        
        print(
            f"Accuracy of the received model on round {input_model.current_round} on the {len(test_features)} test samples: {test_global_acc * 100} %"
        )

        # Training loop
        model.train()
        train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        for epoch in range(args.epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            # Log training metrics
            train_acc = correct / total
            avg_loss = total_loss / len(train_loader)
            mlflow_callback.log_metrics(avg_loss, train_acc, test_global_acc)

        print("Finished Training")
        # get current job_id
        job_id = flare.system_info().get("job_id")
        torch.save(model.state_dict(), os.path.join(job_id, PATH))

        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_features_tensor)
            test_pred = torch.argmax(test_outputs, dim=1)
            test_acc = (test_pred == test_labels_tensor).float().mean().item()
        
        print(f"Accuracy of the model on the {len(test_features)} test samples: {test_acc * 100} %")
        metrics = {"accuracy": test_global_acc}

        # Compute Shapley values for feature importance
        print("Computing Shapley values...")

        plot_prefix = os.path.join(job_id, f"round{input_model.current_round}")
        shap_metrics = compute_shapley_values(
            model, test_features, test_labels, n_samples=100, plot_prefix=plot_prefix, feature_names=feature_columns
        )
        if shap_metrics:
            print(f"SHAP computation completed. Used {shap_metrics['shap_samples_used']} samples.")
        else:
            print("SHAP computation failed. Skipping SHAP metrics.")
        metrics["shap_metrics"] = shap_metrics

        # (6) construct trained FL model (A dict of {parameter name: parameter weights} from the PyTorch model)
        # Combine accuracy and SHAP metrics
        model_params = {name: param.cpu().detach().numpy() for name, param in model.named_parameters()}
        output_model = flare.FLModel(params=model_params, metrics=metrics)
        
        # (7) send model back to NVFlare
        flare.send(output_model)

        mlflow.log_metric("test_global_acc", test_global_acc, input_model.current_round)


if __name__ == "__main__":
    main()
