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


import tensorflow as tf
import numpy as np
from model import SimpleNetwork
from utils import load_csv_data, compute_shapley_values
import json
# (1) import nvflare client API
import nvflare.client as flare

PATH = "./tf_model.weights.h5"


def main():
    # (2) initializes NVFlare client API
    flare.init()

    # Load CSV data using the utility function
    # Example 1: 
    
    # Example 1: Specify specific columns
    (train_features, train_labels), (test_features, test_labels) = load_csv_data(
         file_path='/home/hroth/Code2/nvflare/jpm_demo/examples/advanced/finance-deep-learning/archive/PS_20174392719_1491204439457_log.csv',
         feature_columns=['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'],
         label_column='isFraud'
    )

    # debug
    train_features = train_features[:1000]
    train_labels = train_labels[:1000]
    test_features = test_features[:1000]
    test_labels = test_labels[:1000]

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
        optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )
    model.summary()

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

        model.fit(train_features, train_labels, epochs=1, batch_size=32, validation_data=(test_features, test_labels))

        print("Finished Training")

        model.save_weights(PATH)

        _, test_acc = model.evaluate(test_features, test_labels, verbose=2)
        print(f"Accuracy of the model on the {len(test_features)} test samples: {test_acc * 100} %")

        # Compute Shapley values for feature importance
        print("Computing Shapley values...")
        shap_metrics = compute_shapley_values(model, test_features, test_labels, n_samples=100, plot_prefix=f'round{input_model.current_round}')
        print(f"SHAP computation completed. Used {shap_metrics['shap_samples_used']} samples.")

        # (6) construct trained FL model (A dict of {layer name: layer weights} from the keras model)
        # Combine accuracy and SHAP metrics
        metrics = {"accuracy": test_global_acc}
        metrics.update(shap_metrics)
        
        output_model = flare.FLModel(
            params={layer.name: layer.get_weights() for layer in model.layers}, metrics=metrics
        )
        # (7) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
