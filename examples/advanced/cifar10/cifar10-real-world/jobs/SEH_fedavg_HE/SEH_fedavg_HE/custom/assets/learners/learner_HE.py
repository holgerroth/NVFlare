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
from typing import Optional

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_opt.sklearn.data_loader import load_data_for_range
import pandas as pd
DATASET_PATH = "/tmp/data"

class LinearLearner(Learner):
    def __init__(self, aggregation_epochs: int = 1, lr: float = 1e-2, random_state: int = None):
            super().__init__()
            self.aggregation_epochs = aggregation_epochs
            self.lr = lr
            self.random_state = random_state
    
            # runtime fields
            self.app_root = None
            self.client_id = None
            self.train_data = None
            self.valid_data = None
            self.n_samples = None
            self.n_features = None
            self.local_model = None
            self.validation_results = {}

    def load_data(self, fl_ctx: FLContext) -> dict:
        self.client_id = fl_ctx.get_identity_name()

        # Load training data
        if self.train_data is None:
            train_path = DATASET_PATH + f'/{self.client_id}/train.table'
            train_data = pd.read_table(train_path)

            drop_cols = ['Outcome']
            x_train = train_data.drop(columns=[col for col in drop_cols if col in train_data.columns])
            y_train = train_data['Outcome']

            train_size = len(x_train)

            self.train_data = (x_train, y_train, train_size)

        # Load validation data
        if self.valid_data is None:
            val_path = DATASET_PATH + f'/{self.client_id}/val.table'
            valid_data = pd.read_table(val_path)

            drop_cols = ['Outcome']
            x_val = valid_data.drop(columns=[col for col in drop_cols if col in valid_data.columns])
            y_val = valid_data['Outcome']

            self.valid_data = (x_val, y_val, len(x_val))


        print("train_data")
        print(self.train_data)
        print("valid_data")
        print(self.valid_data)

        return {"train": self.train_data, "valid": self.valid_data}

    def initialize(self, parts: dict, fl_ctx: FLContext):
        #self.log_info(fl_ctx, f"Loading data from {self.data_path}")
        data = self.load_data(fl_ctx)
        self.train_data = data["train"]
        self.valid_data = data["valid"]
        # train data size, to be used for setting
        # NUM_STEPS_CURRENT_ROUND for potential aggregation
        self.n_samples = data["train"][-1]
        self.n_features = data["train"][0].shape[1]
        # model will be created after receiving global parameters

    def set_parameters(self, params):
        print("@@@@@@@@@ Setting parameters", params)
        self.local_model.coef_ = np.array(params["coef"])
        if self.local_model.fit_intercept:
            self.local_model.intercept_ = np.array(params["intercept"])

    def train(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext) -> tuple[dict, dict]:
        (x_train, y_train, train_size) = self.train_data
        if curr_round == 0:
            # initialize model with global_param
            # and set to all zero
            fit_intercept = bool(global_param["fit_intercept"])
            self.local_model = SGDClassifier(
                loss=global_param["loss"],
                penalty=global_param["penalty"],
                fit_intercept=fit_intercept,
                learning_rate=global_param["learning_rate"],
                eta0=global_param["eta0"],
                max_iter=1,
                warm_start=True,
                random_state=self.random_state,
            )
            n_classes = global_param["n_classes"]
            self.local_model.classes_ = np.array(list(range(n_classes)))
            self.local_model.coef_ = np.zeros((1, self.n_features))
            if fit_intercept:
                self.local_model.intercept_ = np.zeros((1,))
        # Training starting from global model
        # Note that the parameter update using global model has been performed
        # during global model evaluation
        self.local_model.fit(x_train, y_train)
        if self.local_model.fit_intercept:
            params = {
                "coef": self.local_model.coef_,
                "intercept": self.local_model.intercept_,
            }
        else:
            params = {"coef": self.local_model.coef_}
        return copy.deepcopy(params), self.local_model

    def validate(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext) -> tuple[dict, dict]:
        # set local model with global parameters
        self.set_parameters(global_param)
        # perform validation
        (x_valid, y_valid, valid_size) = self.valid_data
        y_pred = self.local_model.predict(x_valid)
        auc = roc_auc_score(y_valid, y_pred)
        self.log_info(fl_ctx, f"AUC {auc:.4f}")
        metrics = {"AUC": auc}
        return metrics, self.local_model

    def finalize(self, fl_ctx: FLContext):
        # freeing resources in finalize
        del self.train_data
        del self.valid_data
        self.log_info(fl_ctx, "Freed training resources")