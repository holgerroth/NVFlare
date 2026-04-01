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

import importlib.util
from unittest.mock import MagicMock

import pytest

HAS_JAX_DEPS = all(importlib.util.find_spec(dep) is not None for dep in ("jax", "flax"))
pytestmark = pytest.mark.skipif(not HAS_JAX_DEPS, reason="JAX dependencies are not installed")

jnp = pytest.importorskip("jax.numpy")
serialization = pytest.importorskip("flax.serialization")

from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_opt.jax.decomposers import JaxArrayDecomposer
from nvflare.app_opt.jax.model_persistor import JAXModelPersistor
from nvflare.fuel.utils import fobs


def _make_fl_ctx(tmp_path):
    fl_ctx = MagicMock()
    workspace = MagicMock()
    workspace.get_run_dir.return_value = str(tmp_path)
    workspace.get_result_root.return_value = str(tmp_path)
    fl_ctx.get_workspace.return_value = workspace
    fl_ctx.get_job_id.return_value = "job-1"
    fl_ctx.get_prop.return_value = str(tmp_path)
    fl_ctx.get_peer_context.return_value = None
    fl_ctx.get_identity_name.return_value = "server"
    return fl_ctx


def test_load_model_uses_in_memory_fallback(tmp_path):
    persistor = JAXModelPersistor(model={"params": {"dense": {"kernel": jnp.asarray([1.0, 2.0])}}})

    model_learnable = persistor.load_model(_make_fl_ctx(tmp_path))

    weights = model_learnable[ModelLearnableKey.WEIGHTS]
    assert bool(jnp.allclose(weights["params"]["dense"]["kernel"], jnp.asarray([1.0, 2.0])))


def test_fobs_roundtrip_supports_runtime_jax_array_type():
    weights = {"params": {"dense": {"kernel": jnp.asarray([1.0, 2.0]), "bias": jnp.asarray([0.5])}}}
    fobs.register(JaxArrayDecomposer)

    restored = fobs.loads(fobs.dumps(weights))

    assert bool(jnp.allclose(restored["params"]["dense"]["kernel"], weights["params"]["dense"]["kernel"]))
    assert bool(jnp.allclose(restored["params"]["dense"]["bias"], weights["params"]["dense"]["bias"]))


def test_save_and_load_msgpack_checkpoint(tmp_path):
    weights = {"params": {"dense": {"kernel": jnp.asarray([1.0, 2.0]), "bias": jnp.asarray([0.5])}}}
    persistor = JAXModelPersistor()

    persistor.save_model(make_model_learnable(weights=weights, meta_props={}), _make_fl_ctx(tmp_path))

    saved_path = tmp_path / "models" / "server.msgpack"
    assert saved_path.exists()

    reloaded = JAXModelPersistor(source_ckpt_file_full_name=str(saved_path)).load_model(_make_fl_ctx(tmp_path))
    reloaded_weights = reloaded[ModelLearnableKey.WEIGHTS]

    assert bool(jnp.allclose(reloaded_weights["params"]["dense"]["kernel"], weights["params"]["dense"]["kernel"]))
    assert bool(jnp.allclose(reloaded_weights["params"]["dense"]["bias"], weights["params"]["dense"]["bias"]))

    with open(saved_path, "rb") as f:
        state_dict = serialization.msgpack_restore(f.read())
    assert "params" in state_dict
