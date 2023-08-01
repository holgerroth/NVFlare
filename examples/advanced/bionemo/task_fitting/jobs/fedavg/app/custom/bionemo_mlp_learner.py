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

import copy
import os
from typing import Union
import pickle

import numpy as np
import pandas as pd
from nvflare.apis.fl_constant import FLMetaKey, ReturnCode
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from torch.utils.tensorboard import SummaryWriter
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import sklearn


class BioNeMoMLPLearner(ModelLearner):  # does not support CIFAR10ScaffoldLearner
    def __init__(
        self,
        data_root: str = "/tmp/fasta/mixed_soft",
        aggregation_epochs: int = 1,
        lr: float = 1e-3,
        fedproxloss_mu: float = 0.0,
        central: bool = False,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        """Simple CIFAR-10 Trainer.

        Args:
            data_root: data file root.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            lr: local learning rate. Float number. Defaults to 1e-2.
            fedproxloss_mu: weight for FedProx loss. Float number. Defaults to 0.0 (no FedProx).
            central: Bool. Whether to simulate central training. Default False.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            batch_size: batch size for training and validation.
            num_workers: number of workers for data loaders.

        Returns:
            an FLModel with the updated local model differences after running `train()`, the metrics after `validate()`,
            or the best local model depending on the specified task.
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.data_root = data_root
        self.aggregation_epochs = aggregation_epochs
        self.lr = lr
        self.fedproxloss_mu = fedproxloss_mu
        self.best_acc = 0.0
        self.central = central
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.analytic_sender_id = analytic_sender_id

        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0

        # following will be created in initialize() or later
        self.local_model_file = None
        self.best_local_model_file = None
        self.model = None
        self.epoch_len = None
        self.writer = None
        self.X_train = list()
        self.y_train = list()
        self.X_test = list()
        self.y_test = list()

    def initialize(self):
        """
        Note: this code assumes a FL simulation setting
        Datasets will be initialized in train() and validate() when calling self._create_datasets()
        as we need to make sure that the server has already downloaded and split the data.
        """

        # when the run starts, this is where the actual settings get initialized for trainer
        self.info(
            f"Client {self.site_name} initialized at \n {self.app_root} \n with args: {self.args}",
        )

        self.local_model_file = os.path.join(self.app_root, "local_model.pt")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")

        # Select local TensorBoard writer or event-based writer for streaming
        self.writer = self.get_component(
            self.analytic_sender_id
        )  # user configured config_fed_client.json for streaming
        if not self.writer:  # use local TensorBoard writer only
            self.writer = SummaryWriter(self.app_root)

        # Read embeddings
        data_filename = os.path.join(self.data_root, f"data_{self.site_name}.pkl")
        protein_embeddings = pickle.load(open(data_filename, "rb"))
        self.info(f"Loaded {len(protein_embeddings)} embeddings")

        # Read labels
        labels_filename = os.path.join(self.data_root, f"data_{self.site_name}.csv")
        labels = pd.read_csv(labels_filename).astype(str)

        # Prepare the data for training
        for embedding  in protein_embeddings:
            # get label entry from pandas dataframe
            label = labels.loc[labels["id"]==str(embedding["id"])]
            if label['SET'].item() == 'train':
                self.X_train.append(embedding["embeddings"])
                self.y_train.append(label['TARGET'].item())
            elif label['SET'].item() == 'test':
                self.X_test.append(embedding["embeddings"])
                self.y_test.append(label['TARGET'].item())

        assert len(self.X_train) > 0
        assert len(self.X_test) > 0
        self.info(f"There are {len(self.X_train)} training samples and {len(self.X_test)} testing samples.")

        self.epoch_len = int(len(self.X_train)/self.batch_size)

        self.model = MLPClassifier(solver='adam', hidden_layer_sizes=(32,), batch_size=self.batch_size, max_iter=self.aggregation_epochs,
                                   learning_rate_init=self.lr,
                                   verbose=True, warm_start=True)

        # run fit to initialize the model
        unique_labels, unique_idx = np.unique(self.y_train, return_index=True)
        _X_train = list()
        _y_train = list()
        for i in unique_idx:
            _X_train.append(self.X_train[i])
            _y_train.append(self.y_train[i])
        self.info(f"Found {len(unique_labels)} unique class labels in training data.")
        self.model.fit(_X_train, _y_train)

    def finalize(self):
        # collect threads, close files here
        pass

    def save_model(self, is_best=False):
        # save model
        model_weights = self.model.coefs_
        save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
        if is_best:
            save_dict.update({"best_acc": self.best_acc})
            pickle.dump(save_dict, open(self.best_local_model_file, "wb"))
        else:
            pickle.dump(save_dict, open(self.local_model_file, "wb"))

    def load_weights(self, weights):
        weights = copy.deepcopy(weights)
        coefs = []
        intercepts = []
        for name in weights:
            if "coef" in name:
                coefs.append(weights[name])
            elif "intercept" in name:
                intercepts.append(weights[name])
            else:
                raise ValueError(f"Expected to name to contain either `coef` or `intercept` but it was `{name}`.")
        self.model.coefs_ = coefs
        self.model.intercepts_ = intercepts

    def compute_weights_diff(self, global_weights):
        local_weights = {}
        for i, w in enumerate(self.model.coefs_):
            local_weights[f"coef_{i}"] = w
        for i, w in enumerate(self.model.intercepts_):
            local_weights[f"intercept_{i}"] = w

        # compute delta model, global model has the primary key set
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = np.subtract(local_weights[name], global_weights[name], dtype=np.float32)
            if np.any(np.isnan(model_diff[name])):
                raise ValueError(f"{name} weights became NaN...")
        return model_diff

    def train(self, model: FLModel) -> Union[str, FLModel]:
        # get round information
        self.info(f"Current/Total Round: {self.current_round + 1}/{self.total_rounds} (epoch_len={self.epoch_len})")
        self.info(f"Client identity: {self.site_name}")

        # update local model weights with received weights
        global_weights = model.params
        self.load_weights(global_weights)

        # local steps
        self.model.fit(self.X_train, self.y_train)

        # compute weight differences
        model_diff = self.compute_weights_diff(global_weights)

        # return an FLModel containing the model differences
        fl_model = FLModel(params_type=ParamsType.DIFF, params=model_diff)

        FLModelUtils.set_meta_prop(fl_model, FLMetaKey.NUM_STEPS_CURRENT_ROUND, self.epoch_len)
        self.info("Local epochs finished. Returning FLModel")

        self.epoch_of_start_time += self.aggregation_epochs

        return fl_model

    def get_model(self, model_name: str) -> Union[str, FLModel]:
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            try:
                # load model to cpu as server might or might not have a GPU
                model_weights = pickle.load(self.best_local_model_file)
            except Exception as e:
                raise ValueError("Unable to load best model") from e

            # Create FLModel from model data.
            if model_weights:
                return FLModel(params_type=ParamsType.FULL, params=model_weights)
            else:
                # Set return code.
                self.error(f"best local model not found at {self.best_local_model_file}.")
                return ReturnCode.EXECUTION_RESULT_ERROR
        else:
            raise ValueError(f"Unknown model_type: {model_name}")  # Raised errors are caught in LearnerExecutor class.

    def validate(self, model: FLModel) -> Union[str, FLModel]:
        # get validation information
        self.info(f"Client identity: {self.site_name}")

        # update local model weights with received weights
        self.load_weights(model.params)

        # get validation meta info
        validate_type = FLModelUtils.get_meta_prop(
            model, FLMetaKey.VALIDATE_TYPE, ValidateType.MODEL_VALIDATE
        )
        model_owner = self.get_shareable_header(AppConstants.MODEL_OWNER)

        # perform valid
        predicted_testing_labels = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predicted_testing_labels)
        self.info(f"Model (owner={model_owner}) has an accuracy of {(accuracy * 100):.2f}%")
        self.info("Evaluation finished. Returning result")

        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.save_model(is_best=True)

        # write to tensorboard
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            self.writer.add_scalar("accuracy", accuracy, self.epoch_of_start_time)

            classifcation_report = sklearn.metrics.classification_report(self.y_test, predicted_testing_labels,
                                                                         labels=list(set(predicted_testing_labels)),
                                                                         output_dict=True)
            for category, metrics in classifcation_report.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"{category} {k}", v, self.epoch_of_start_time)
                else:
                    self.writer.add_scalar(f"{category}", metrics, self.epoch_of_start_time)

        val_results = {"accuracy": accuracy}
        return FLModel(metrics=val_results, params_type=None)
