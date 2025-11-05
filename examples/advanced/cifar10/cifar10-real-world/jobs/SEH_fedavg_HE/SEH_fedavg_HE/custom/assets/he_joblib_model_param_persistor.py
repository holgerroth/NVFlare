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

import os
from typing import Dict

import tenseal as ts
from joblib import dump, load

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.model_desc import ModelDescriptor
from nvflare.app_opt.he.homomorphic_encrypt import (
    deserialize_nested_dict,
    load_tenseal_context_from_workspace,
    serialize_nested_dict,
)


class HEJoblibModelParamPersistor(ModelPersistor):
    def __init__(self, initial_params, save_name="model_param.joblib", tenseal_context_file="server_context.tenseal"):
        """
        Persist global model parameters from a dict to a joblib file with support for HE (CKKSVector).
        
        Args:
            initial_params: Initial model parameters
            save_name: Name of the file to save the model to
            tenseal_context_file: TenSEAL context file for encryption/decryption
        """
        super().__init__()
        self.initial_params = initial_params
        self.save_name = save_name
        self.tenseal_context_file = tenseal_context_file
        self.tenseal_context = None

    def _initialize(self, fl_ctx: FLContext):
        # get save path from FLContext
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.log_dir = app_root
        self.save_path = os.path.join(self.log_dir, self.save_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        fl_ctx.sync_sticky()

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Initialize and load the Model.

        Args:
            fl_ctx: FLContext

        Returns:
            ModelLearnable object
        """
        if os.path.exists(self.save_path):
            self.logger.info("Loading server model")
            model = load(self.save_path)
            # Deserialize any CKKSVector objects that were serialized
            if self.tenseal_context is not None:
                model = deserialize_nested_dict(model, self.tenseal_context)
        else:
            self.logger.info(f"Initialization, sending global settings: {self.initial_params}")
            model = self.initial_params
        model_learnable = make_model_learnable(weights=model, meta_props=dict())

        return model_learnable

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)
            # Load TenSEAL context
            self.tenseal_context = load_tenseal_context_from_workspace(self.tenseal_context_file, fl_ctx)
        elif event == EventType.END_RUN:
            self.tenseal_context = None

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        """Persists the Model object, handling CKKSVector serialization.

        Args:
            model_learnable: ModelLearnable object
            fl_ctx: FLContext
        """
        if model_learnable:
            if fl_ctx.get_prop(AppConstants.CURRENT_ROUND) == fl_ctx.get_prop(AppConstants.NUM_ROUNDS) - 1:
                self.logger.info(f"Saving received model to {os.path.abspath(self.save_path)}")
                # save 'weights' which contains model parameters
                model = model_learnable[ModelLearnableKey.WEIGHTS]
                
                # Serialize any CKKSVector objects before saving
                model_to_save = serialize_nested_dict(model)
                
                dump(model_to_save, self.save_path, compress=1)

    def get_model_inventory(self, fl_ctx: FLContext) -> Dict[str, ModelDescriptor]:
        """Get the model inventory.

        Args:
            fl_ctx: FLContext

        Returns:
            Dict of model_name: ModelDescriptor
        """
        model_inventory = {}
        if os.path.exists(self.save_path):
            _, tail = os.path.split(self.save_name)
            model_inventory[tail] = ModelDescriptor(
                name=self.save_name,
                location=self.save_path,
                model_format="joblib",
                props={},
            )
        return model_inventory

    def get_model(self, model_file: str, fl_ctx: FLContext) -> ModelLearnable:
        """Get a specific model by its file name.

        Args:
            model_file: name of the model file
            fl_ctx: FLContext

        Returns:
            ModelLearnable object
        """
        model_inventory = self.get_model_inventory(fl_ctx)
        if model_file not in model_inventory:
            self.logger.error(f"Model {model_file} not found in inventory")
            return None
        
        descriptor = model_inventory[model_file]
        location = descriptor.location
        
        try:
            self.logger.info(f"Loading model from {location}")
            model = load(location)
            # Deserialize any CKKSVector objects that were serialized
            if self.tenseal_context is not None:
                model = deserialize_nested_dict(model, self.tenseal_context)
            model_learnable = make_model_learnable(weights=model, meta_props=dict())
            return model_learnable
        except Exception as e:
            self.logger.error(f"Error loading model from {location}: {e}")
            return None
