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

import sys

import numpy as np


def test_msgpack_roundtrip_preserves_numpy_tree_without_importing_jax():
    import importlib

    jax_was_loaded = "jax" in sys.modules
    serialization = importlib.import_module("nvflare.app_opt.jax.serialization")

    tree = {
        "params": {
            "dense": {
                "kernel": np.arange(6, dtype=np.float32).reshape(2, 3),
                "bias": np.asarray([0.5], dtype=np.float32),
            }
        },
        "step": np.asarray(3, dtype=np.int32),
    }

    restored = serialization.msgpack_restore(serialization.msgpack_serialize(tree))

    if not jax_was_loaded:
        assert "jax" not in sys.modules
    np.testing.assert_allclose(restored["params"]["dense"]["kernel"], tree["params"]["dense"]["kernel"])
    np.testing.assert_allclose(restored["params"]["dense"]["bias"], tree["params"]["dense"]["bias"])
    assert restored["step"] == tree["step"]


def test_msgpack_roundtrip_restores_chunked_arrays(monkeypatch):
    from nvflare.app_opt.jax import serialization

    monkeypatch.setattr(serialization, "MAX_CHUNK_SIZE", 16)
    tree = {"params": {"dense": {"kernel": np.arange(32, dtype=np.float32)}}}

    restored = serialization.msgpack_restore(serialization.msgpack_serialize(tree))

    np.testing.assert_allclose(restored["params"]["dense"]["kernel"], tree["params"]["dense"]["kernel"])
