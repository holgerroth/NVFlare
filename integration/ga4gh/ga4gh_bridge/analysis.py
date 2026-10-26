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

"""Synthetic hello-numpy calculation; also staged into the ordinary TES container."""

import json
import math
import sys
from pathlib import Path


def calculate(document):
    def increment(value):
        if isinstance(value, list) and value:
            return [increment(x) for x in value]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Expected finite numeric weights")
        return value + 1

    return {"weights": increment(document["weights"])}


def run_file(source, destination):
    result = calculate(json.loads(Path(source).read_text()))
    Path(destination).write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    run_file(*sys.argv[1:])
