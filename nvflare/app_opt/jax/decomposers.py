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

from io import BytesIO
from typing import Tuple

import jax.numpy as jnp
import numpy as np

import nvflare.fuel.utils.fobs.dots as dots
from nvflare.app_common.np.np_downloader import ArrayDownloadable, download_arrays
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.download_service import Downloadable
from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.decomposers.via_downloader import ViaDownloaderDecomposer


def _arrays_to_jax(arrays: dict[str, np.ndarray], **kwargs):
    _ = kwargs
    return {key: jnp.asarray(value) for key, value in arrays.items()}


def _supported_jax_array_type():
    # FOBS matches decomposers by the exact runtime class name, so we must register
    # the concrete array implementation rather than the abstract jax.Array alias.
    return type(jnp.asarray(0))


class JaxArrayDecomposer(ViaDownloaderDecomposer):
    def __init__(self):
        ViaDownloaderDecomposer.__init__(self, 1024 * 1024 * 2, "jax_")

    def supported_type(self):
        return _supported_jax_array_type()

    def get_download_dot(self) -> int:
        return dots.JAX_DOWNLOAD

    def to_downloadable(self, items: dict, max_chunk_size: int, fobs_ctx: dict) -> Downloadable:
        return ArrayDownloadable(items, max_chunk_size)

    def download(
        self,
        from_fqcn: str,
        ref_id: str,
        per_request_timeout: float,
        cell: Cell,
        secure=False,
        optional=False,
        abort_signal=None,
    ) -> Tuple[str, dict]:
        return download_arrays(
            from_fqcn=from_fqcn,
            ref_id=ref_id,
            per_request_timeout=per_request_timeout,
            cell=cell,
            secure=secure,
            optional=optional,
            abort_signal=abort_signal,
            arrays_received_cb=_arrays_to_jax,
        )

    def native_decompose(self, target, manager: DatumManager = None) -> bytes:
        stream = BytesIO()
        np.save(stream, np.asarray(target), allow_pickle=False)
        return stream.getvalue()

    def native_recompose(self, data: bytes, manager: DatumManager = None):
        stream = BytesIO(data)
        return jnp.asarray(np.load(stream, allow_pickle=False))
