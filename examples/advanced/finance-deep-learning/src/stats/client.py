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

from typing import Dict, Optional

import pandas as pd

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.statistics.df.df_core_statistics import DFStatisticsCore
from .utils import load_csv_data_from_path, validate_data_features, split_data_for_statistics


class FinancialStatistics(DFStatisticsCore):
    def __init__(
        self,
        data_path,
        data_features=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"],
    ):
        super().__init__()
        self.data_path = data_path
        self.data: Optional[Dict[str, pd.DataFrame]] = None
        self.data_features = data_features

    def load_data(self, fl_ctx: FLContext) -> Dict[str, pd.DataFrame]:
        client_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"load data for client {client_name}")
        try:
            # Load CSV data using the utility function
            df = load_csv_data_from_path(
                data_path=self.data_path,
                data_features=self.data_features
            )
            
            # Validate the loaded data
            validate_data_features(df, self.data_features)
            
            # Split data into train and test sets
            train, test = split_data_for_statistics(df, train_frac=0.8, random_state=200)
            
            self.log_info(fl_ctx, f"load data done for client {client_name}")
            return {"train": train, "test": test}

        except Exception as e:
            raise Exception(f"Load data for client {client_name} failed! {e}")

    def initialize(self, fl_ctx: FLContext):
        self.data = self.load_data(fl_ctx)
