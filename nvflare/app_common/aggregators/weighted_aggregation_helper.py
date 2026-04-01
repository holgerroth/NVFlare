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

import re
import threading
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Set


def _is_aggregatable_metric_value(v: Any) -> bool:
    """Return True if the metric value supports weighted aggregation (v * weight and addition).

    Boolean values are considered aggregatable and treated as binary values
    (`True=1.0`, `False=0.0`) when averaged.
    """
    if v is None:
        return False
    if isinstance(v, (dict, list, set, tuple, str)):
        return False
    # Bool metrics are treated as binary values (True=1, False=0) and averaged.
    if isinstance(v, (int, float, bool)):
        return True
    try:
        _ = v * 1.0
        _ = v + v
        return True
    except (TypeError, ValueError, AttributeError):
        return False


def filter_aggregatable_metrics(
    metrics: Optional[Dict[str, Any]],
    warn_skipped: Optional[Callable[[str, str], None]] = None,
    warned_metric_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Return metric entries that support weighted aggregation.

    Note:
        Boolean metric values are included and aggregate as binary rates.

    Args:
        metrics: Dict of metric name -> value.
        warn_skipped: Optional callback invoked as warn_skipped(key, type_name) for skipped metrics.
        warned_metric_keys: Optional set of keys already warned about. If provided, warnings are emitted
            at most once per key and newly warned keys are added to this set.
    """
    if not metrics:
        return {}

    filtered = {}
    for key, value in metrics.items():
        if _is_aggregatable_metric_value(value):
            filtered[key] = value
            continue
        if warn_skipped is None:
            continue
        if warned_metric_keys is None or key not in warned_metric_keys:
            warn_skipped(key, type(value).__name__)
            if warned_metric_keys is not None:
                warned_metric_keys.add(key)
    return filtered


class WeightedAggregationHelper(object):
    def __init__(self, exclude_vars: Optional[str] = None, weigh_by_local_iter: bool = True):
        """Perform weighted aggregation.

        Args:
            exclude_vars (str, optional): regex string to match excluded vars during aggregation. Defaults to None.
            weigh_by_local_iter (bool, optional): Whether to weight the contributions by the number of iterations
                performed in local training in the current round. Defaults to `True`.
                Setting it to `False` can be useful in applications such as homomorphic encryption to reduce
                the number of computations on encrypted ciphertext.
                The aggregated sum will still be divided by the provided weights and `aggregation_weights` for the
                resulting weighted sum to be valid.
        """
        super().__init__()
        self.lock = threading.Lock()
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.weigh_by_local_iter = weigh_by_local_iter
        self.reset_stats()
        self.total = dict()
        self.counts = dict()
        self.history = list()

    def reset_stats(self):
        self.total = dict()
        self.counts = dict()
        self.history = list()

    @staticmethod
    def _is_pytorch_tensor(tensor):
        """Check if tensor is a PyTorch tensor with in-place operation support."""
        return hasattr(tensor, "add_") and hasattr(tensor, "mul_") and hasattr(tensor, "clone")

    @staticmethod
    def _join_path(path: str, child: str) -> str:
        return child if not path else f"{path}/{child}"

    @staticmethod
    def _is_sequence_node(value: Any) -> bool:
        return isinstance(value, (list, tuple))

    @staticmethod
    def _materialize_leaf(value: Any) -> Any:
        materialize_fn = getattr(value, "materialize", None)
        if callable(materialize_fn):
            return materialize_fn()
        return value

    def _init_leaf(self, value: Any, weight: float):
        if self._is_pytorch_tensor(value):
            if self.weigh_by_local_iter:
                return value.mul(weight)
            return value.clone()

        if self.weigh_by_local_iter:
            return value * weight

        try:
            return value.copy() if hasattr(value, "copy") else value
        except (ValueError, RuntimeError):
            return value

    def _accumulate_leaf(self, current_total: Any, value: Any, weight: float):
        if self._is_pytorch_tensor(value) and self._is_pytorch_tensor(current_total):
            if self.weigh_by_local_iter:
                current_total.add_(value, alpha=weight)
            else:
                current_total.add_(value)
            return current_total

        if self.weigh_by_local_iter:
            return current_total + value * weight
        return current_total + value

    def _divide_leaf(self, value: Any, count: float):
        if self._is_pytorch_tensor(value):
            return value.div_(count)
        return value * (1.0 / count)

    def _add_value(self, current_total: Any, current_count: Any, value: Any, weight: float, path: str):
        if self.exclude_vars is not None and self.exclude_vars.search(path):
            return None, None, True

        value = self._materialize_leaf(value)

        if isinstance(value, Mapping):
            if current_total is not None and not isinstance(current_total, Mapping):
                raise ValueError(f"Aggregation structure mismatch at {path}: expected Mapping accumulator")
            if current_count is not None and not isinstance(current_count, Mapping):
                raise ValueError(f"Aggregation count mismatch at {path}: expected Mapping accumulator")

            total_result = {}
            count_result = {}
            seen_keys = set()
            for key, child_value in value.items():
                child_path = self._join_path(path, str(key))
                existing_total = None if current_total is None else current_total.get(key)
                existing_count = None if current_count is None else current_count.get(key)
                if current_total is not None and key not in current_total:
                    raise ValueError(f"Aggregation structure mismatch at {child_path}: unexpected key")

                child_total, child_count, skipped = self._add_value(
                    existing_total,
                    existing_count,
                    child_value,
                    weight,
                    child_path,
                )
                if skipped:
                    continue
                total_result[key] = child_total
                count_result[key] = child_count
                seen_keys.add(key)

            if current_total is not None:
                missing_keys = set(current_total.keys()) - seen_keys
                if missing_keys:
                    raise ValueError(
                        f"Aggregation structure mismatch at {path}: missing keys {sorted(str(k) for k in missing_keys)}"
                    )
            if not total_result:
                return None, None, True
            return total_result, count_result, False

        if self._is_sequence_node(value):
            if current_total is not None and not self._is_sequence_node(current_total):
                raise ValueError(f"Aggregation structure mismatch at {path}: expected sequence accumulator")
            if current_count is not None and not self._is_sequence_node(current_count):
                raise ValueError(f"Aggregation count mismatch at {path}: expected sequence accumulator")

            if current_total is not None and len(current_total) != len(value):
                raise ValueError(f"Aggregation sequence length mismatch at {path}")

            total_items = []
            count_items = []
            for idx, child_value in enumerate(value):
                child_path = f"{path}[{idx}]"
                existing_total = None if current_total is None else current_total[idx]
                existing_count = None if current_count is None else current_count[idx]
                child_total, child_count, skipped = self._add_value(
                    existing_total,
                    existing_count,
                    child_value,
                    weight,
                    child_path,
                )
                if skipped:
                    raise ValueError(f"exclude_vars is not supported for sequence children at {child_path}")
                total_items.append(child_total)
                count_items.append(child_count)

            if isinstance(value, tuple):
                return tuple(total_items), tuple(count_items), False
            return total_items, count_items, False

        if current_total is None:
            return self._init_leaf(value, weight), weight, False

        return self._accumulate_leaf(current_total, value, weight), current_count + weight, False

    def _get_result_value(self, total_value: Any, count_value: Any):
        if isinstance(total_value, Mapping):
            return {key: self._get_result_value(total_value[key], count_value[key]) for key in total_value.keys()}

        if self._is_sequence_node(total_value):
            values = [self._get_result_value(v, c) for v, c in zip(total_value, count_value)]
            if isinstance(total_value, tuple):
                return tuple(values)
            return values

        return self._divide_leaf(total_value, count_value)

    def add(self, data, weight, contributor_name, contribution_round):
        """Compute weighted sum and sum of weights."""
        with self.lock:
            for k, v in data.items():
                total_value, count_value, skipped = self._add_value(
                    self.total.get(k, None),
                    self.counts.get(k, None),
                    v,
                    weight,
                    str(k),
                )
                if skipped:
                    continue
                self.total[k] = total_value
                self.counts[k] = count_value

            self.history.append(
                {
                    "contributor_name": contributor_name,
                    "round": contribution_round,
                    "weight": weight,
                }
            )

    def get_result(self):
        """Divide weighted sum by sum of weights."""
        with self.lock:
            aggregated_dict = {
                key: self._get_result_value(total_value, self.counts[key]) for key, total_value in self.total.items()
            }

            self.reset_stats()
            return aggregated_dict

    def get_history(self):
        return self.history

    def get_len(self):
        return len(self.get_history())
