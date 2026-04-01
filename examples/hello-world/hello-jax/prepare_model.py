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

"""Prepare an initial JAX state-dict checkpoint for hello-jax."""

import argparse
import os

from flax import serialization
from model import create_initial_params, params_to_state_dict

DEFAULT_OUTPUT = "/tmp/nvflare/data/hello-jax/initial_model.msgpack"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.output.endswith(".msgpack"):
        raise ValueError("hello-jax checkpoints must use the .msgpack extension.")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    initial_state_dict = params_to_state_dict(create_initial_params())
    with open(args.output, "wb") as f:
        f.write(serialization.msgpack_serialize(initial_state_dict))


if __name__ == "__main__":
    main()
