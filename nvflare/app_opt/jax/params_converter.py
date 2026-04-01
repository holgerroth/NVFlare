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

from collections.abc import Mapping

import numpy as np

from nvflare.app_common.abstract.params_converter import ParamsConverter


def _tree_convert(data, leaf_fn):
    if isinstance(data, Mapping):
        return {key: _tree_convert(value, leaf_fn) for key, value in data.items()}
    if isinstance(data, list):
        return [_tree_convert(value, leaf_fn) for value in data]
    if isinstance(data, tuple):
        return tuple(_tree_convert(value, leaf_fn) for value in data)
    return leaf_fn(data)


class NumpyToJAXParamsConverter(ParamsConverter):
    def convert(self, params, fl_ctx):
        _ = fl_ctx
        import jax.numpy as jnp

        return _tree_convert(params, lambda x: jnp.asarray(x))


class JAXToNumpyParamsConverter(ParamsConverter):
    def convert(self, params, fl_ctx):
        _ = fl_ctx
        return _tree_convert(params, lambda x: np.asarray(x))
