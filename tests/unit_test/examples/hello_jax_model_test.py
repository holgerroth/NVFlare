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
import os
import sys
from collections.abc import Mapping

import numpy as np
import pytest

HAS_JAX_DEPS = all(importlib.util.find_spec(dep) is not None for dep in ("jax", "flax", "optax"))
pytestmark = pytest.mark.skipif(not HAS_JAX_DEPS, reason="JAX example dependencies are not installed")


def _load_hello_jax_module(file_name: str, module_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    example_dir = os.path.join(repo_root, "examples", "hello-world", "hello-jax")
    module_path = os.path.join(example_dir, file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, example_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _assert_tree_allclose(module, expected, actual):
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert set(expected.keys()) == set(actual.keys())
        for key in expected.keys():
            _assert_tree_allclose(module, expected[key], actual[key])
        return

    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual):
            _assert_tree_allclose(module, expected_item, actual_item)
        return

    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual):
            _assert_tree_allclose(module, expected_item, actual_item)
        return

    expected_array = module.jnp.asarray(expected)
    actual_array = module.jnp.asarray(actual)
    assert expected_array.shape == actual_array.shape
    assert bool(module.jnp.allclose(expected_array, actual_array, rtol=1e-6, atol=1e-6))


def test_jax_param_state_dict_roundtrip():
    model_module = _load_hello_jax_module("model.py", "hello_jax_model")
    params = model_module.create_initial_params()

    state_dict = model_module.params_to_state_dict(params)
    restored_params = model_module.params_from_state_dict(state_dict)
    restored_state_dict = model_module.params_to_state_dict(restored_params)

    assert isinstance(state_dict, dict)
    assert state_dict
    _assert_tree_allclose(model_module, state_dict, restored_state_dict)


def test_jax_train_state_uses_same_param_structure():
    model_module = _load_hello_jax_module("model.py", "hello_jax_model")
    params = model_module.create_initial_params()
    state = model_module.create_train_state(params, learning_rate=0.05, momentum=0.9)

    state_dict = model_module.params_to_state_dict(params)
    state_state_dict = model_module.params_to_state_dict(state.params)
    _assert_tree_allclose(model_module, state_dict, state_state_dict)


def test_jax_state_dict_rejects_missing_keys():
    model_module = _load_hello_jax_module("model.py", "hello_jax_model")
    params = model_module.create_initial_params()
    state_dict = model_module.params_to_state_dict(params)
    missing_key = next(iter(state_dict.keys()))
    state_dict.pop(missing_key)

    with pytest.raises(ValueError):
        model_module.params_from_state_dict(state_dict)


def test_jax_train_epoch_rejects_empty_data():
    client_module = _load_hello_jax_module("client.py", "hello_jax_client")
    model_module = _load_hello_jax_module("model.py", "hello_jax_model")
    params = model_module.create_initial_params()
    state = model_module.create_train_state(params, learning_rate=0.05, momentum=0.9)
    empty_images = np.zeros((0, 28, 28, 1), dtype=np.float32)
    empty_labels = np.zeros((0,), dtype=np.int32)

    with pytest.raises(ValueError, match="No training data available"):
        client_module.train_epoch(state, empty_images, empty_labels, 128, client_module.jax.random.PRNGKey(0))


def test_jax_evaluate_rejects_empty_data():
    client_module = _load_hello_jax_module("client.py", "hello_jax_client")
    model_module = _load_hello_jax_module("model.py", "hello_jax_model")
    params = model_module.create_initial_params()
    empty_images = np.zeros((0, 28, 28, 1), dtype=np.float32)
    empty_labels = np.zeros((0,), dtype=np.int32)

    with pytest.raises(ValueError, match="No evaluation data available"):
        client_module.evaluate(params, empty_images, empty_labels, 128)
