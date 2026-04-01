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

"""JAX/Flax model utilities for the hello-jax MNIST example."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import serialization
from flax.training import train_state


class ConvNet(nn.Module):
    """Small CNN for MNIST classification."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


MODEL = ConvNet()
INPUT_SHAPE = (1, 28, 28, 1)


@lru_cache(maxsize=1)
def _template_params():
    params = MODEL.init(jax.random.PRNGKey(0), jnp.ones(INPUT_SHAPE, dtype=jnp.float32))["params"]
    return params


def create_initial_params():
    return _template_params()


def params_to_state_dict(params):
    return serialization.to_state_dict(params)


def params_from_state_dict(state_dict):
    return serialization.from_state_dict(_template_params(), state_dict)


def create_train_state(params, learning_rate: float, momentum: float) -> train_state.TrainState:
    tx = optax.sgd(learning_rate=learning_rate, momentum=momentum)
    return train_state.TrainState.create(apply_fn=MODEL.apply, params=params, tx=tx)
