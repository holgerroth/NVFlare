# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import tensorflow as tf
from model import SimpleNetwork
from utils import MLflowCallback, compute_shapley_values, load_csv_data

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.client.tracking import MLflowWriter

PATH = "tf_model.weights.h5"


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

    model = SimpleNetwork(input_shape=(None, n_features), num_classes=n_classes)
    model.build(input_shape=(None, n_features))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )
    model.summary()

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
        for k, v in input_model.params.items():
            model.get_layer(k).set_weights(v)

        # (5) evaluate aggregated/received model
        _, test_global_acc = model.evaluate(test_features, test_labels, verbose=2)
        print(
            f"Accuracy of the received model on round {input_model.current_round} on the {len(test_features)} test samples: {test_global_acc * 100} %"
        )

        # Use the callback in model.fit()
        model.fit(
            train_features,
            train_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(test_features, test_labels),
            callbacks=[mlflow_callback],
        )

        print("Finished Training")
        # get current job_id
        job_id = flare.system_info().get("job_id")
        model.save_weights(os.path.join(job_id, PATH))

        _, test_acc = model.evaluate(test_features, test_labels, verbose=2)
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

        # (6) construct trained FL model (A dict of {layer name: layer weights} from the keras model)
        # Combine accuracy and SHAP metrics

        output_model = flare.FLModel(
            params={layer.name: layer.get_weights() for layer in model.layers}, metrics=metrics
        )
        # (7) send model back to NVFlare
        flare.send(output_model)

        mlflow.log_metric("test_global_acc", test_global_acc, input_model.current_round)


if __name__ == "__main__":
    main()
