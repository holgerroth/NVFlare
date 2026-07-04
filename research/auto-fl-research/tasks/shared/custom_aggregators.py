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

"""Shared custom aggregator implementations for the Auto-FL NVFlare starter.

These are NVFlare-oriented aggregation variants intended for bounded autoresearch experiments.
The repo-level loop is inspired by the public karpathy/autoresearch operating model, but the
aggregation code itself is adapted to NVFlare's FLModel / ModelAggregator interfaces.
"""

from __future__ import annotations

from collections import deque

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.app_constant import AlgorithmConstants

SCAFFOLD_CTRL_DIFF = AlgorithmConstants.SCAFFOLD_CTRL_DIFF
SCAFFOLD_CTRL_GLOBAL = AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL


def _as_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_output_type(value, reference):
    if torch is not None and isinstance(reference, torch.Tensor):
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if isinstance(reference, np.ndarray):
        return np.asarray(value, dtype=reference.dtype)
    return np.asarray(value)


def _to_meta_tensor(value, reference):
    # Persisted checkpoints reload with torch.load(weights_only=True), which
    # rejects pickled numpy objects; meta arrays must be CPU torch tensors.
    result = np.asarray(value)
    if reference is not None:
        ref_array = _as_numpy(reference)
        result = result.astype(ref_array.dtype, copy=False)
    if torch is not None:
        return torch.from_numpy(np.ascontiguousarray(result))
    return result


def _raise_empty_aggregation(aggregator_name: str, client_weights):
    raise ValueError(
        f"{aggregator_name} cannot aggregate because total client weight is zero; "
        f"received client weights={list(client_weights)}. Check NUM_STEPS_CURRENT_ROUND "
        "and client training output before continuing."
    )


class WeightedAggregator(ModelAggregator):
    def __init__(self):
        super().__init__()
        self.weighted_sum = {}
        self.total_weight = 0.0
        self.client_weights = []
        self.params_type = None

    def accept_model(self, model: FLModel):
        weight = model.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
        self.client_weights.append(weight)

        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}.")

        for key, value in model.params.items():
            if key not in self.weighted_sum:
                self.weighted_sum[key] = value * weight
            else:
                self.weighted_sum[key] += value * weight
        self.total_weight += weight

    def aggregate_model(self) -> FLModel:
        if self.total_weight == 0:
            _raise_empty_aggregation(type(self).__name__, self.client_weights)

        aggregated_params = {key: val / self.total_weight for key, val in self.weighted_sum.items()}
        return FLModel(params=aggregated_params, params_type=self.params_type)

    def reset_stats(self):
        self.weighted_sum = {}
        self.total_weight = 0.0
        self.client_weights = []
        self.params_type = None


class FedAvgAggregator(WeightedAggregator):
    """Explicit FedAvg alias for benchmark readability."""


class FedOptAggregator(ModelAggregator):
    """Server-side optimizer over weighted client DIFFs.

    This keeps the FL contract intact: clients still send model DIFFs with
    NUM_STEPS_CURRENT_ROUND, and the server returns a DIFF update with the same
    parameter keys and params_type.
    """

    def __init__(
        self,
        optimizer: str = "sgdm",
        server_lr: float = 1.0,
        server_momentum: float = 0.6,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        window_avg: int = 0,
        window_avg_tail_rounds: int = 0,
        total_rounds: int = 0,
        fednova_norm: bool = False,
    ):
        super().__init__()
        if optimizer not in {"sgdm", "adam"}:
            raise ValueError(f"Unsupported FedOpt optimizer: {optimizer}")
        if server_lr <= 0.0:
            raise ValueError("server_lr must be > 0")
        if not 0.0 <= server_momentum < 1.0:
            raise ValueError("server_momentum must be in [0, 1)")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if tau <= 0.0:
            raise ValueError("tau must be > 0")
        if window_avg < 0:
            raise ValueError("window_avg must be >= 0")
        if window_avg_tail_rounds < 0:
            raise ValueError("window_avg_tail_rounds must be >= 0")
        if window_avg > 1 and window_avg_tail_rounds > 0:
            raise ValueError("window_avg (feedback) and window_avg_tail_rounds (tail-only) are mutually exclusive")
        if window_avg_tail_rounds > 0 and total_rounds <= 0:
            raise ValueError("window_avg_tail_rounds requires total_rounds > 0")

        self.optimizer = optimizer
        self.server_lr = server_lr
        self.server_momentum = server_momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.window_avg = window_avg
        self.window_avg_tail_rounds = window_avg_tail_rounds
        self.fednova_norm = fednova_norm

        self.first_moment = {}
        self.second_moment = {}
        self.adam_step = 0
        # Round-model coordinates relative to the initial global model; the
        # broadcast state is reconstructed as initial + cumulative emitted
        # updates, so window averaging needs no absolute weights.
        self.cum_emitted = {}
        if window_avg > 1:
            self.round_history = deque(maxlen=window_avg)
        elif window_avg_tail_rounds > 0:
            self.round_history = deque(maxlen=window_avg_tail_rounds)
        else:
            self.round_history = None
        self.round_index = 0
        self.total_rounds = total_rounds
        self.reset_stats()

    def accept_model(self, model: FLModel):
        weight = float(model.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0))
        self.client_weights.append(weight)

        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}.")

        for key, value in model.params.items():
            diff = _as_numpy(value).astype(np.float64, copy=False)
            self.references.setdefault(key, value)
            if key not in self.weighted_sum:
                self.weighted_sum[key] = diff * weight
            else:
                self.weighted_sum[key] += diff * weight
            if self.fednova_norm:
                if key not in self.plain_sum:
                    self.plain_sum[key] = diff.copy()
                else:
                    self.plain_sum[key] += diff
        self.total_weight += weight
        self.total_weight_sq += weight * weight

    def aggregate_model(self) -> FLModel:
        if self.total_weight == 0:
            _raise_empty_aggregation(type(self).__name__, self.client_weights)

        if self.fednova_norm:
            # FedNova normalized averaging with data-proportional steps:
            # tau_eff * sum_i(p_i * d_i / tau_i) with p_i = tau_i / T reduces to
            # (sum tau_i^2 / T^2) * sum_i d_i [src: Wang20 arXiv:2007.07481].
            scale = self.total_weight_sq / (self.total_weight * self.total_weight)
            mean_diff = {key: val * scale for key, val in self.plain_sum.items()}
        else:
            mean_diff = {key: val / self.total_weight for key, val in self.weighted_sum.items()}
        if self.optimizer == "sgdm":
            update = self._sgdm_update(mean_diff)
        else:
            update = self._adam_update(mean_diff)

        if self.round_history is not None:
            if self.window_avg > 1:
                update = self._window_average_update(update)
            else:
                update = self._tail_average_update(update)

        aggregated_params = {key: _to_output_type(update[key], self.references[key]) for key in update}
        return FLModel(params=aggregated_params, params_type=self.params_type)

    def _record_round_coord(self, update):
        round_coord = {}
        for key, val in update.items():
            previous = self.cum_emitted.get(key)
            if previous is None:
                previous = np.zeros_like(val)
                self.cum_emitted[key] = previous
            round_coord[key] = previous + val
        self.round_history.append(round_coord)
        return round_coord

    def _window_average_update(self, update):
        """Window-average of round-wise global models, fed back as the broadcast
        state [src: Pu21 arXiv:2103.11619, WiMA arXiv:2310.01366]."""
        self._record_round_coord(update)

        emitted = {}
        for key in update:
            target = sum(coord[key] for coord in self.round_history) / len(self.round_history)
            emitted[key] = target - self.cum_emitted[key]
            self.cum_emitted[key] = target
        return emitted

    def _tail_average_update(self, update):
        """SWA-style tail average: rounds proceed unmodified, but the FINAL
        persisted global model is the mean of the last W round models
        [src: Izmailov18 SWA arXiv:1803.05407, Pu21 arXiv:2103.11619]."""
        round_coord = self._record_round_coord(update)
        self.round_index += 1

        if self.round_index < self.total_rounds:
            for key in update:
                self.cum_emitted[key] = round_coord[key]
            return update

        emitted = {}
        for key in update:
            target = sum(coord[key] for coord in self.round_history) / len(self.round_history)
            emitted[key] = target - self.cum_emitted[key]
            self.cum_emitted[key] = target
        return emitted

    def reset_stats(self):
        self.weighted_sum = {}
        self.plain_sum = {}
        self.total_weight = 0.0
        self.total_weight_sq = 0.0
        self.client_weights = []
        self.params_type = None
        self.references = {}

    def _sgdm_update(self, mean_diff):
        updates = {}
        for key, diff in mean_diff.items():
            previous = self.first_moment.get(key)
            if previous is None:
                previous = np.zeros_like(diff)
            velocity = self.server_momentum * previous + diff
            self.first_moment[key] = velocity
            updates[key] = self.server_lr * velocity
        return updates

    def _adam_update(self, mean_diff):
        updates = {}
        self.adam_step += 1
        first_bias_correction = 1.0 - self.beta1**self.adam_step
        second_bias_correction = 1.0 - self.beta2**self.adam_step

        for key, diff in mean_diff.items():
            first = self.first_moment.get(key)
            if first is None:
                first = np.zeros_like(diff)
            second = self.second_moment.get(key)
            if second is None:
                second = np.zeros_like(diff)

            first = self.beta1 * first + (1.0 - self.beta1) * diff
            second = self.beta2 * second + (1.0 - self.beta2) * np.square(diff)
            self.first_moment[key] = first
            self.second_moment[key] = second
            first_hat = first / first_bias_correction
            second_hat = second / second_bias_correction
            updates[key] = self.server_lr * first_hat / (np.sqrt(second_hat) + self.tau)
        return updates


class FedAvgMAggregator(FedOptAggregator):
    def __init__(
        self,
        server_lr: float = 1.0,
        server_momentum: float = 0.6,
        window_avg: int = 0,
        window_avg_tail_rounds: int = 0,
        total_rounds: int = 0,
        fednova_norm: bool = False,
    ):
        super().__init__(
            optimizer="sgdm",
            server_lr=server_lr,
            server_momentum=server_momentum,
            window_avg=window_avg,
            window_avg_tail_rounds=window_avg_tail_rounds,
            total_rounds=total_rounds,
            fednova_norm=fednova_norm,
        )


class FedAdamAggregator(FedOptAggregator):
    def __init__(
        self,
        server_lr: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
    ):
        super().__init__(
            optimizer="adam",
            server_lr=server_lr,
            beta1=beta1,
            beta2=beta2,
            tau=tau,
        )


class ScaffoldAggregator(ModelAggregator):
    """SCAFFOLD aggregation over DIFF params plus control-variate metadata.

    Control deltas are step-weighted by NUM_STEPS_CURRENT_ROUND to match NVFlare's
    built-in scaffold workflow aggregation.
    """

    def __init__(self):
        super().__init__()
        self.global_controls = {}
        self.reset_stats()

    def accept_model(self, model: FLModel):
        weight = float(model.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0))
        self.client_weights.append(weight)

        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}.")

        for key, value in model.params.items():
            diff = _as_numpy(value).astype(np.float64, copy=False)
            self.references.setdefault(key, value)
            if key not in self.weighted_sum:
                self.weighted_sum[key] = diff * weight
            else:
                self.weighted_sum[key] += diff * weight

        ctrl_diff = model.meta.get(SCAFFOLD_CTRL_DIFF)
        if not ctrl_diff:
            client_name = model.meta.get("site_name", "unknown")
            raise ValueError(
                f"Client '{client_name}' did not return required "
                f"FLModel.meta['{SCAFFOLD_CTRL_DIFF}'] for SCAFFOLD aggregation."
            )
        for key, value in ctrl_diff.items():
            diff = _as_numpy(value).astype(np.float64, copy=False)
            self.control_references.setdefault(key, value)
            if key not in self.control_weighted_sum:
                self.control_weighted_sum[key] = diff * weight
            else:
                self.control_weighted_sum[key] += diff * weight

        self.total_weight += weight

    def aggregate_model(self) -> FLModel:
        if self.total_weight == 0:
            _raise_empty_aggregation(type(self).__name__, self.client_weights)

        aggregated_params = {
            key: _to_output_type(val / self.total_weight, self.references[key])
            for key, val in self.weighted_sum.items()
        }

        for key, val in self.control_weighted_sum.items():
            delta = val / self.total_weight
            previous = self.global_controls.get(key)
            if previous is None:
                previous = np.zeros_like(delta)
            self.global_controls[key] = previous + delta

        global_controls = {
            key: _to_meta_tensor(value, self.control_references.get(key)) for key, value in self.global_controls.items()
        }
        return FLModel(
            params=aggregated_params,
            params_type=self.params_type,
            meta={SCAFFOLD_CTRL_GLOBAL: global_controls},
        )

    def reset_stats(self):
        self.weighted_sum = {}
        self.control_weighted_sum = {}
        self.total_weight = 0.0
        self.client_weights = []
        self.params_type = None
        self.references = {}
        self.control_references = {}


class MedianAggregator(ModelAggregator):
    def __init__(self):
        super().__init__()
        self.client_models = []
        self.params_type = None

    def accept_model(self, model: FLModel):
        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}.")
        self.client_models.append(model.params)

    def aggregate_model(self) -> FLModel:
        if not self.client_models:
            raise ValueError(
                "MedianAggregator cannot aggregate because no client models were accepted. "
                "Check client training output before continuing."
            )

        aggregated_params = {}
        param_keys = self.client_models[0].keys()
        for key in param_keys:
            stacked = np.stack([_as_numpy(m[key]) for m in self.client_models], axis=0)
            aggregated_params[key] = _to_output_type(np.median(stacked, axis=0), self.client_models[0][key])

        return FLModel(params=aggregated_params, params_type=self.params_type)

    def reset_stats(self):
        self.client_models = []
        self.params_type = None
