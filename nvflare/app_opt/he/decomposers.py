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
"""Decomposers for HE related classes"""
import zlib
from typing import Any

import tenseal as ts

from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import Datum, DatumManager


class _SerializedCKKS:
    """Wrapper class to prevent FOBS from externalizing large CKKSVector data.
    By wrapping the data in a custom class, FOBS will decompose it through this decomposer
    rather than treating it as raw bytes and externalizing it."""
    
    def __init__(self, vec_data: bytes, ctx_data: bytes):
        self.vec_data = vec_data
        self.ctx_data = ctx_data


class _SerializedCKKSDecomposer(fobs.Decomposer):
    """Decomposer for the wrapper class that keeps data inline."""
    
    def supported_type(self):
        return _SerializedCKKS
    
    def decompose(self, target: _SerializedCKKS, manager: DatumManager = None) -> Any:
        # Return as list of smaller chunks to avoid externalization
        # Split data into chunks to stay under FOBS externalization threshold
        vec_chunks = self._chunk_data(target.vec_data)
        ctx_chunks = self._chunk_data(target.ctx_data)
        
        return {
            "v": vec_chunks,
            "c": ctx_chunks
        }
    
    def recompose(self, data: Any, manager: DatumManager = None) -> _SerializedCKKS:
        vec_chunks = data["v"]
        ctx_chunks = data["c"]
        
        # Reassemble chunks
        vec_data = self._reassemble_chunks(vec_chunks)
        ctx_data = self._reassemble_chunks(ctx_chunks)
        
        return _SerializedCKKS(vec_data, ctx_data)
    
    def _chunk_data(self, data: bytes, chunk_size: int = 5 * 1024 * 1024) -> list:
        """Split data into chunks smaller than FOBS externalization threshold."""
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    def _reassemble_chunks(self, chunks: list) -> bytes:
        """Reassemble chunks, handling memoryview conversion."""
        result = b''
        for chunk in chunks:
            if isinstance(chunk, memoryview):
                chunk = bytes(chunk)
            result += chunk
        return result


class CKKSVectorDecomposer(fobs.Decomposer):
    def supported_type(self):
        return ts.CKKSVector

    def decompose(self, target: ts.CKKSVector, manager: DatumManager = None) -> Any:
        # Serialize and compress the data
        vec_data = target.serialize()
        ctx_data = target.context().serialize()
        
        # Compress to reduce size
        compressed_vec = zlib.compress(vec_data, level=1)
        compressed_ctx = zlib.compress(ctx_data, level=1)
        
        # Wrap in custom class to prevent externalization
        return _SerializedCKKS(compressed_vec, compressed_ctx)

    def recompose(self, data: Any, manager: DatumManager = None) -> ts.CKKSVector:
        if isinstance(data, _SerializedCKKS):
            vec_data = data.vec_data
            ctx_data = data.ctx_data
        else:
            # Fallback for old format (shouldn't happen after update)
            vec_data = data["vec"]
            ctx_data = data["ctx"]
        
        # Convert memoryview to bytes if necessary
        if isinstance(vec_data, memoryview):
            vec_data = bytes(vec_data)
        if isinstance(ctx_data, memoryview):
            ctx_data = bytes(ctx_data)
        
        # Decompress the data
        vec_data = zlib.decompress(vec_data)
        ctx_data = zlib.decompress(ctx_data)
        
        context = ts.context_from(ctx_data)
        return ts.ckks_vector_from(context, vec_data)


def register():
    if register.registered:
        return

    fobs.register(_SerializedCKKSDecomposer)
    fobs.register(CKKSVectorDecomposer)

    register.registered = True


register.registered = False
