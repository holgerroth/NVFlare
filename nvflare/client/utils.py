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

from collections.abc import Mapping
from typing import Any, Dict

from .config import ExchangeFormat


def _diff_values(original: Any, new: Any, path: str):
    if isinstance(original, Mapping):
        if not isinstance(new, Mapping):
            raise RuntimeError(f"parameter tree mismatch at {path or '<root>'}: expected Mapping")

        diff_dict = {}
        for key in original:
            if key not in new:
                continue
            child_path = f"{path}/{key}" if path else str(key)
            diff_dict[key] = _diff_values(original[key], new[key], child_path)
        if diff_dict == {}:
            raise RuntimeError(f"no common keys between original and new dict at {path or '<root>'}")
        return diff_dict

    if isinstance(original, list):
        if not isinstance(new, list):
            raise RuntimeError(f"parameter tree mismatch at {path or '<root>'}: expected list")
        if len(original) != len(new):
            raise RuntimeError(f"parameter list length mismatch at {path or '<root>'}")
        return [_diff_values(o, n, f"{path}[{idx}]") for idx, (o, n) in enumerate(zip(original, new))]

    if isinstance(original, tuple):
        if not isinstance(new, tuple):
            raise RuntimeError(f"parameter tree mismatch at {path or '<root>'}: expected tuple")
        if len(original) != len(new):
            raise RuntimeError(f"parameter tuple length mismatch at {path or '<root>'}")
        return tuple(_diff_values(o, n, f"{path}[{idx}]") for idx, (o, n) in enumerate(zip(original, new)))

    if isinstance(new, list) and isinstance(original, list):
        return [new[i] - original[i] for i in range(len(new))]

    return new - original


def numerical_params_diff(original: Dict, new: Dict) -> Dict:
    """Calculates the numerical parameter difference.

    Args:
        original: A dict of numerical values.
        new: A dict of numerical values.

    Returns:
        A dict with common keys that exist in both original dict and new dict,
        values are the difference between original and new.
    """
    return _diff_values(original=original, new=new, path="")


DIFF_FUNCS = {
    ExchangeFormat.PYTORCH: numerical_params_diff,
    ExchangeFormat.NUMPY: numerical_params_diff,
    ExchangeFormat.JAX: numerical_params_diff,
}
