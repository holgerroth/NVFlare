# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from collections.abc import Mapping
from typing import Any, Optional

import jax.numpy as jnp
from flax import serialization

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_opt.jax.decomposers import JaxArrayDecomposer
from nvflare.fuel.utils import fobs


def _to_jax_tree(tree):
    if isinstance(tree, Mapping):
        return {key: _to_jax_tree(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_to_jax_tree(value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_to_jax_tree(value) for value in tree)
    if tree is None or isinstance(tree, (str, bytes)):
        return tree
    return jnp.asarray(tree)


def _resolve_model_file(fl_ctx: FLContext, model_dir: str, model_name: str) -> str:
    workspace = fl_ctx.get_workspace()
    job_id = fl_ctx.get_job_id()
    if job_id is None:
        raise RuntimeError("job_id is missing in fl_ctx.")
    run_dir = workspace.get_run_dir(job_id)
    return os.path.join(run_dir, model_dir, model_name)


def _resolve_source_ckpt(fl_ctx: FLContext, source_ckpt_file_full_name: str) -> str:
    if os.path.isabs(source_ckpt_file_full_name):
        return source_ckpt_file_full_name

    app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
    return os.path.join(app_root, WorkspaceConstants.CUSTOM_FOLDER_NAME, source_ckpt_file_full_name)


class JAXModelPersistor(ModelPersistor):
    def __init__(
        self,
        model_dir: str = "models",
        model_name: str = "server.msgpack",
        model: Optional[Any] = None,
        source_ckpt_file_full_name: Optional[str] = None,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.model_name = model_name
        self.model = model
        self.source_ckpt_file_full_name = source_ckpt_file_full_name

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            fobs.register(JaxArrayDecomposer)

    @staticmethod
    def _serialize_tree(tree: Any) -> bytes:
        return serialization.msgpack_serialize(serialization.to_state_dict(tree))

    @staticmethod
    def _deserialize_tree(serialized_tree: bytes):
        return _to_jax_tree(serialization.msgpack_restore(serialized_tree))

    def _load_from_file(self, filepath: str):
        with open(filepath, "rb") as f:
            return self._deserialize_tree(f.read())

    def _get_initial_model(self):
        if self.model is None:
            raise ValueError("JAXModelPersistor requires either model or source_ckpt_file_full_name.")
        return _to_jax_tree(serialization.to_state_dict(self.model))

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        fobs.register(JaxArrayDecomposer)
        model_path = _resolve_model_file(fl_ctx, self.model_dir, self.model_name)

        weights = None
        if self.source_ckpt_file_full_name:
            ckpt_path = _resolve_source_ckpt(fl_ctx, self.source_ckpt_file_full_name)
            if not os.path.exists(ckpt_path):
                raise ValueError(f"Source checkpoint not found: {ckpt_path}. Check that it exists at runtime.")
            self.log_info(fl_ctx, f"Loading JAX model from source checkpoint: {ckpt_path}", fire_event=False)
            weights = self._load_from_file(ckpt_path)
        elif os.path.exists(model_path):
            self.log_info(fl_ctx, f"Loaded JAX model from {model_path}", fire_event=False)
            weights = self._load_from_file(model_path)

        if weights is None:
            weights = self._get_initial_model()

        model_learnable = make_model_learnable(weights=weights, meta_props={})
        self.log_info(fl_ctx, f"Loaded initial model: {model_learnable[ModelLearnableKey.WEIGHTS]}")
        return model_learnable

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        fobs.register(JaxArrayDecomposer)
        workspace = fl_ctx.get_workspace()
        job_id = fl_ctx.get_job_id()
        model_root_dir = os.path.join(workspace.get_result_root(job_id), self.model_dir)
        if not os.path.exists(model_root_dir):
            os.makedirs(model_root_dir)

        model_path = os.path.join(model_root_dir, self.model_name)
        with open(model_path, "wb") as f:
            f.write(self._serialize_tree(model_learnable[ModelLearnableKey.WEIGHTS]))
        self.log_info(fl_ctx, f"Saved JAX model to: {model_path}")
